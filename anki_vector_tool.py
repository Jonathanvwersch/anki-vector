import concurrent.futures
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import chromadb
import click
import requests
from chromadb.errors import ChromaError
from chromadb.utils import embedding_functions

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def sanitize_collection_name(deck_name: str) -> str:
    """
    Sanitize the deck name to be a valid ChromaDB collection name.
    Non-alphanumeric characters (except underscores and hyphens) become underscores.
    Ensures the name length is between 3 and 63 characters.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", deck_name)
    sanitized = re.sub(r"^[^A-Za-z0-9]+", "", sanitized)
    sanitized = re.sub(r"[^A-Za-z0-9]+$", "", sanitized)
    if len(sanitized) < 3:
        sanitized = sanitized.ljust(3, "_")
    if len(sanitized) > 63:
        sanitized = sanitized[:63]
    return sanitized


# -----------------------------------------------------------------------------
# Anki Vector Manager
# -----------------------------------------------------------------------------
class AnkiVectorManager:
    """
    Manages synchronization between Anki (via AnkiConnect) and a local ChromaDB vector DB.
    Each deck is stored in its own collection. This version performs incremental sync,
    only adding new notes and deleting ones that were removed in Anki.
    """

    def __init__(self, db_path: str = "./vector_db"):
        self.db_path = db_path
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        try:
            self.chroma_client = chromadb.PersistentClient(path=db_path)
        except ChromaError as e:
            logging.error("Failed to initialize ChromaDB client: %s", e)
            raise
        # Use the default embedding function for better performance
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

    def get_collection(self, deck_name: str) -> Any:
        """
        Retrieve or create the ChromaDB collection for a deck.
        """
        sanitized = sanitize_collection_name(deck_name)
        collection_name = f"anki_cards_{sanitized}"
        try:
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name, embedding_function=self.embedding_function
            )
            return collection
        except ChromaError as e:
            logging.error(
                "Failed to create/retrieve collection for deck '%s': %s", deck_name, e
            )
            raise

    def invoke_anki_connect(
        self, action: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send a JSON request to AnkiConnect and return its response.
        """
        endpoint = "http://localhost:8765"
        payload = {"action": action, "version": 6, "params": params}
        try:
            response = requests.post(endpoint, json=payload, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error("AnkiConnect request failed: %s", e)
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            logging.error("Invalid JSON received from AnkiConnect: %s", e)
            return {"error": "Invalid JSON from AnkiConnect"}

    def get_deck_cards(self, deck_name: str) -> List[Dict[str, Any]]:
        """
        Retrieve all notes for a given deck from Anki.
        """
        query = f'deck:"{deck_name}"'
        logging.info("Retrieving notes for deck '%s'.", deck_name)
        find_resp = self.invoke_anki_connect("findNotes", {"query": query})
        if find_resp.get("error"):
            logging.error("findNotes error: %s", find_resp["error"])
            return []
        note_ids = find_resp.get("result", [])
        if not note_ids:
            logging.info("No notes found in deck '%s'.", deck_name)
            return []
        info_resp = self.invoke_anki_connect("notesInfo", {"notes": note_ids})
        if info_resp.get("error"):
            logging.error("notesInfo error: %s", info_resp["error"])
            return []
        return info_resp.get("result", [])

    def process_card_batch(self, cards: List[Dict[str, Any]]) -> tuple:
        """Process a batch of cards and return their documents, metadata, and IDs."""
        documents = []
        metadatas = []
        ids = []

        for card in cards:
            try:
                note_id = str(card["noteId"])
                front = card["fields"]["Front"]["value"]
                back = card["fields"]["Back"]["value"]

                # Add front text
                documents.append(front)
                metadatas.append(
                    {
                        "note_id": note_id,
                        "type": "front",
                        "front": front,
                        "back": back,
                    }
                )
                ids.append(f"{note_id}_front")

                # Add back text
                documents.append(back)
                metadatas.append(
                    {
                        "note_id": note_id,
                        "type": "back",
                        "front": front,
                        "back": back,
                    }
                )
                ids.append(f"{note_id}_back")
            except Exception as e:
                logging.error("Error processing card: %s", e)
                continue

        return documents, metadatas, ids

    def incremental_sync_deck(self, deck_name: str) -> int:
        """
        Truly incremental sync - only process new or changed cards.
        """
        anki_cards = self.get_deck_cards(deck_name)
        if not anki_cards:
            logging.info("No cards to sync for deck '%s'.", deck_name)
            return 0

        collection = self.get_collection(deck_name)

        # Get existing cards from vector DB
        try:
            existing_docs = collection.get(
                where={"type": "front"}, include=["metadatas"]
            )
            existing_note_ids = (
                {meta["note_id"] for meta in existing_docs["metadatas"]}
                if existing_docs["ids"]
                else set()
            )
        except Exception as e:
            logging.error("Error getting existing cards: %s", e)
            existing_note_ids = set()

        # Find new cards
        current_note_ids = {str(card["noteId"]) for card in anki_cards}
        new_note_ids = current_note_ids - existing_note_ids
        removed_note_ids = existing_note_ids - current_note_ids

        # Remove deleted cards
        if removed_note_ids:
            try:
                ids_to_remove = []
                for note_id in removed_note_ids:
                    ids_to_remove.extend([f"{note_id}_front", f"{note_id}_back"])
                collection.delete(ids=ids_to_remove)
                logging.info("Removed %d deleted cards", len(removed_note_ids))
            except Exception as e:
                logging.error("Error removing deleted cards: %s", e)

        if not new_note_ids:
            logging.info("No new cards to add for deck '%s'", deck_name)
            return 0

        # Process only new cards in parallel batches
        new_cards = [card for card in anki_cards if str(card["noteId"]) in new_note_ids]
        batch_size = 20
        batches = [
            new_cards[i : i + batch_size] for i in range(0, len(new_cards), batch_size)
        ]
        total_batches = len(batches)
        added_count = 0

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_batch = {
                executor.submit(self.process_card_batch, batch): i
                for i, batch in enumerate(batches)
            }

            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    documents, metadatas, ids = future.result()
                    if documents:
                        try:
                            collection.add(
                                documents=documents,
                                metadatas=metadatas,
                                ids=ids,
                            )
                            batch_size = (
                                len(documents) // 2
                            )  # Each card has front and back
                            added_count += batch_size
                            logging.info(
                                "Added batch %d/%d - %d new cards (total: %d/%d)",
                                batch_idx + 1,
                                total_batches,
                                batch_size,
                                added_count,
                                len(new_cards),
                            )
                        except ChromaError as e:
                            logging.error(
                                "Failed to add batch %d/%d: %s",
                                batch_idx + 1,
                                total_batches,
                                e,
                            )
                except Exception as e:
                    logging.error(
                        "Error processing batch %d/%d: %s",
                        batch_idx + 1,
                        total_batches,
                        e,
                    )

        logging.info(
            "Sync complete for deck '%s': %d new cards added, %d removed",
            deck_name,
            added_count,
            len(removed_note_ids),
        )
        return added_count

    def find_similar_cards(
        self, front: str, back: str, deck_name: str, threshold: float = 0.9
    ) -> List[Dict[str, Any]]:
        """
        Query the deck's collection for cards with a front text similar to the provided text.
        Returns any results whose similarity (computed as 1 - distance) exceeds the threshold.
        """
        collection = self.get_collection(deck_name)
        try:
            results = collection.query(
                query_texts=[front],
                n_results=5,
                include=["metadatas", "distances", "documents"],
                where={"type": "front"},
            )
        except ChromaError as e:
            logging.error("Query failed for deck '%s': %s", deck_name, e)
            return []

        similar = []
        distances = results.get("distances", [[]])
        metadatas = results.get("metadatas", [[]])
        for idx, distance in enumerate(distances[0]):
            similarity = 1 - distance  # assuming distance ∈ [0,1]
            if similarity >= threshold:
                meta = metadatas[0][idx]
                similar.append(
                    {
                        "similarity": similarity,
                        "front": meta.get("front", ""),
                        "back": meta.get("back", ""),
                        "note_id": meta.get("note_id", ""),
                        "match_type": "front",
                    }
                )
                logging.info(
                    "Found similar card (note %s) with similarity %.2f",
                    meta.get("note_id", ""),
                    similarity,
                )
        return similar

    def add_single_card_to_vector_db(self, note_id: int, deck_name: str) -> bool:
        """
        Retrieve note details for a given note_id and add/update its entries in the vector DB.
        """
        resp = self.invoke_anki_connect("notesInfo", {"notes": [note_id]})
        if resp.get("error") or not resp.get("result"):
            logging.error(
                "No note info for note_id %s in deck '%s'.", note_id, deck_name
            )
            return False

        note_data = resp["result"][0]
        front = note_data["fields"]["Front"]["value"]
        back = note_data["fields"]["Back"]["value"]
        collection = self.get_collection(deck_name)
        try:
            collection.add(
                documents=[front, back],
                metadatas=[
                    {
                        "note_id": str(note_id),
                        "type": "front",
                        "front": front,
                        "back": back,
                    },
                    {
                        "note_id": str(note_id),
                        "type": "back",
                        "front": front,
                        "back": back,
                    },
                ],
                ids=[f"{note_id}_front", f"{note_id}_back"],
            )
            logging.info("Added/Updated note_id %s in vector DB.", note_id)
            return True
        except ChromaError as e:
            logging.error("Failed to add note_id %s: %s", note_id, e)
            return False

    def add_card_to_anki(self, deck_name: str, front: str, back: str) -> bool:
        """
        Create a new Basic card in Anki with the provided front/back text and add it to the vector DB.
        """
        payload = {
            "note": {
                "deckName": deck_name,
                "modelName": "Basic",
                "fields": {"Front": front, "Back": back},
                "options": {"allowDuplicate": True},
                "tags": [],
            }
        }
        resp = self.invoke_anki_connect("addNote", payload)
        if resp.get("error"):
            logging.error("Failed to add note to Anki: %s", resp["error"])
            return False
        if "result" not in resp:
            logging.error("Unexpected response from Anki: %s", resp)
            return False

        note_id = resp["result"]
        return self.add_single_card_to_vector_db(note_id, deck_name)


# -----------------------------------------------------------------------------
# CLI Commands
# -----------------------------------------------------------------------------
@click.group()
def cli():
    """CLI tool to manage Anki decks with a local vector DB for duplicate detection."""
    pass


def prompt_for_deck(
    manager: AnkiVectorManager, deck_name: Optional[str] = None
) -> Optional[str]:
    """
    If a deck name is not provided, prompt the user to choose one from Anki.
    """
    if deck_name:
        return deck_name

    resp = manager.invoke_anki_connect("deckNames", {})
    decks = resp.get("result", [])
    if not decks:
        click.echo("No decks found in Anki.")
        return None

    click.echo("\nAvailable decks:")
    for idx, d in enumerate(decks, 1):
        click.echo(f"{idx}. {d}")
    try:
        choice = click.prompt("Choose deck number", type=int)
        if 1 <= choice <= len(decks):
            return decks[choice - 1]
    except click.Abort:
        return None
    return None


@cli.command(name="sync")
@click.argument("deck_name", required=False)
def sync(deck_name: str):
    """
    Incrementally synchronize the vector DB for a given deck.
    """
    manager = AnkiVectorManager()
    deck = prompt_for_deck(manager, deck_name)
    if not deck:
        click.echo("No deck selected.")
        return
    new_count = manager.incremental_sync_deck(deck)
    click.echo(f"Synced deck '{deck}': {new_count} new note(s) added.")


@cli.command(name="sync-all")
def sync_all():
    """
    Incrementally sync all decks from Anki into their vector DB collections.
    """
    manager = AnkiVectorManager()
    resp = manager.invoke_anki_connect("deckNames", {})
    decks = resp.get("result", [])
    if not decks:
        click.echo("No decks found in Anki.")
        return
    total_new = 0
    for deck in decks:
        click.echo(f"Syncing deck: {deck}")
        new_count = manager.incremental_sync_deck(deck)
        click.echo(f"  {new_count} new note(s) added.\n")
        total_new += new_count
    click.echo(f"All decks synced. Total new notes added: {total_new}")


@cli.command(name="add-card")
@click.argument("deck_name", required=False)
def add_card(deck_name: str):
    """
    Add a new Basic card to Anki. Before adding, check for similar cards in the deck.
    """
    manager = AnkiVectorManager()
    deck = prompt_for_deck(manager, deck_name)
    if not deck:
        click.echo("No deck selected.")
        return

    click.echo("\nEnter FRONT text (finish with Ctrl+D/Ctrl+Z):")
    front_lines = []
    try:
        while True:
            front_lines.append(input())
    except EOFError:
        pass
    front_text = "\n".join(front_lines).strip()
    if not front_text:
        click.echo("Front text cannot be empty.")
        return

    click.echo("\nEnter BACK text (finish with Ctrl+D/Ctrl+Z):")
    back_lines = []
    try:
        while True:
            back_lines.append(input())
    except EOFError:
        pass
    back_text = "\n".join(back_lines).strip()
    if not back_text:
        click.echo("Back text cannot be empty.")
        return

    click.secho("\nSyncing deck before duplicate check...", fg="yellow")
    manager.incremental_sync_deck(deck)

    similar = manager.find_similar_cards(front_text, back_text, deck, threshold=0.9)
    if similar:
        click.secho(f"\nFound {len(similar)} similar card(s):", fg="red")
        for idx, card in enumerate(similar, start=1):
            click.echo(f"{idx}. Similarity: {card['similarity']:.2f}")
            click.echo(f"   Front: {card['front']}")
            click.echo(f"   Back:  {card['back']}")
        click.echo("\nOptions:")
        click.echo("  0 - Add as a new card anyway")
        for idx in range(1, len(similar) + 1):
            click.echo(
                f"  {idx} - Overwrite card with note_id {similar[idx-1]['note_id']}"
            )
        click.echo("  C - Cancel")
        choice = click.prompt("Choose an option", type=str).upper()
        if choice == "C":
            click.echo("Operation cancelled.")
            return
        elif choice.isdigit():
            choice_num = int(choice)
            if choice_num == 0:
                ok = manager.add_card_to_anki(deck, front_text, back_text)
                click.echo("New card added." if ok else "Failed to add card.")
            elif 1 <= choice_num <= len(similar):
                note_id = similar[choice_num - 1]["note_id"]
                payload = {
                    "note": {
                        "id": int(note_id),
                        "fields": {"Front": front_text, "Back": back_text},
                    }
                }
                resp = manager.invoke_anki_connect("updateNoteFields", payload)
                if not resp.get("error"):
                    click.echo("Card updated successfully!")
                    manager.add_single_card_to_vector_db(note_id, deck)
                else:
                    click.echo(f"Failed to update card: {resp.get('error')}")
            else:
                click.echo("Invalid option. No changes made.")
            return
        else:
            click.echo("Invalid input. Operation cancelled.")
            return
    else:
        ok = manager.add_card_to_anki(deck, front_text, back_text)
        click.echo("Card added successfully!" if ok else "Failed to add card.")

    click.secho("\nSyncing deck after changes...", fg="yellow")
    manager.incremental_sync_deck(deck)
    click.secho("Done!", fg="green")


@cli.command(name="list-decks")
def list_decks():
    """
    List all decks from Anki via AnkiConnect.
    """
    manager = AnkiVectorManager()
    resp = manager.invoke_anki_connect("deckNames", {})
    decks = resp.get("result", [])
    if decks:
        click.echo("Available decks:")
        for d in decks:
            click.echo(f"  - {d}")
    else:
        click.echo("Could not retrieve decks from AnkiConnect.")


@cli.command(name="add-from-file")
@click.argument("file_path", type=click.Path(exists=True), required=False)
@click.argument("deck_name", required=False)
def add_from_file(file_path: str, deck_name: str):
    """
    Bulk-import cards from a text file. Cards are separated by the delimiter 'SEPARATOR'.
    The first nonempty line is treated as the FRONT text and the remaining as the BACK.
    """
    manager = AnkiVectorManager()
    if not file_path:
        file_path = click.prompt(
            "Enter the path to the file", type=click.Path(exists=True)
        )
    deck = prompt_for_deck(manager, deck_name)
    if not deck:
        click.echo("No deck selected.")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as e:
        click.echo(f"Error reading file: {e}")
        return

    raw_cards = content.split("SEPARATOR")
    cards = []
    for raw in raw_cards:
        lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
        if len(lines) >= 2:
            question = lines[0]
            answer = "\n".join(lines[1:])
            cards.append((question, answer))
    click.echo(f"Found {len(cards)} card(s) in file '{file_path}'.")

    click.secho("\nSyncing deck before import...", fg="yellow")
    manager.incremental_sync_deck(deck)

    added = 0
    skipped = 0
    with click.progressbar(cards, label="Importing cards") as bar:
        for front, back in bar:
            similar = manager.find_similar_cards(front, back, deck, threshold=0.9)
            if similar:
                click.secho(f"\nSkipping duplicate card: {front[:50]}...", fg="yellow")
                skipped += 1
                continue
            if manager.add_card_to_anki(deck, front, back):
                added += 1
            else:
                skipped += 1

    click.secho("\nImport Summary:", fg="green")
    click.echo(f"  Added:   {added}")
    click.echo(f"  Skipped: {skipped}")
    click.echo(f"  Total:   {added + skipped}")


if __name__ == "__main__":
    cli()
