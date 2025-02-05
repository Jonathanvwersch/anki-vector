import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import chromadb
import click
import requests
from chromadb.errors import ChromaError
from chromadb.utils import embedding_functions

# Configure logging (INFO level is used by default; DEBUG messages will be hidden)
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)


def sanitize_collection_name(deck_name: str) -> str:
    """
    Convert a deck name into a valid collection name for ChromaDB.
    Replaces characters that are not alphanumeric, underscores, or hyphens with underscores.
    Ensures the name is between 3 and 63 characters long.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", deck_name)
    sanitized = re.sub(r"^[^A-Za-z0-9]+", "", sanitized)
    sanitized = re.sub(r"[^A-Za-z0-9]+$", "", sanitized)
    if len(sanitized) < 3:
        sanitized = sanitized.ljust(3, "_")
    if len(sanitized) > 63:
        sanitized = sanitized[:63]
    return sanitized


class AnkiVectorManager:
    """
    Manages synchronization of Anki flashcards with a local ChromaDB vector database
    and performs similarity checks to prevent duplicates. Each deck is stored in its own collection.
    """

    def __init__(self, db_path: str = "./vector_db"):
        """
        Initialize the vector database client.
        """
        self.db_path = db_path
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        try:
            self.chroma_client = chromadb.PersistentClient(path=db_path)
        except ChromaError as e:
            logging.error("Failed to initialize ChromaDB client: %s", e)
            raise
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

    def get_collection(self, deck_name: str):
        """
        Retrieve or create a ChromaDB collection for the given deck.
        """
        sanitized_deck = sanitize_collection_name(deck_name)
        collection_name = f"anki_cards_{sanitized_deck}"
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
        Send a request to the AnkiConnect plugin.
        """
        endpoint = "http://localhost:8765"
        payload = {"action": action, "version": 6, "params": params}
        try:
            response = requests.post(endpoint, json=payload, timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error("AnkiConnect request failed: %s", e)
            return {"error": str(e)}
        try:
            return response.json()
        except json.JSONDecodeError as e:
            logging.error("Failed to parse AnkiConnect response as JSON: %s", e)
            return {"error": "Invalid JSON from AnkiConnect"}

    def get_deck_cards(self, deck_name: str) -> List[Dict[str, Any]]:
        """
        Retrieve all notes from the specified deck.
        """
        query = f'deck:"{deck_name}"'
        logging.info("Retrieving notes for deck '%s'.", deck_name)
        find_notes_resp = self.invoke_anki_connect("findNotes", {"query": query})
        if find_notes_resp.get("error"):
            logging.error("Error in findNotes response: %s", find_notes_resp["error"])
            return []
        note_ids = find_notes_resp.get("result", [])
        if not note_ids:
            logging.info("No notes found for deck '%s'.", deck_name)
            return []
        notes_info_resp = self.invoke_anki_connect("notesInfo", {"notes": note_ids})
        return notes_info_resp.get("result", [])

    def add_cards_to_vector_db(self, deck_name: str) -> int:
        """
        Synchronize cards from the specified deck to the corresponding ChromaDB collection.
        Removes cards that no longer exist in Anki and adds new ones.
        Returns the number of new cards added.
        """
        anki_cards = self.get_deck_cards(deck_name)
        anki_note_ids = {str(card["noteId"]) for card in anki_cards}
        collection = self.get_collection(deck_name)

        try:
            existing_docs = collection.get()
            existing_ids = set(existing_docs.get("ids", []))
        except ChromaError as e:
            logging.error("Failed to fetch documents for deck '%s': %s", deck_name, e)
            return 0

        ids_to_add = anki_note_ids - existing_ids
        ids_to_remove = existing_ids - anki_note_ids

        if ids_to_remove:
            try:
                collection.delete(ids=list(ids_to_remove))
                logging.info(
                    "Removed %d outdated card(s) from deck '%s'.",
                    len(ids_to_remove),
                    deck_name,
                )
            except ChromaError as e:
                logging.error(
                    "Failed to remove outdated cards for deck '%s': %s", deck_name, e
                )

        new_cards = [card for card in anki_cards if str(card["noteId"]) in ids_to_add]
        if not new_cards:
            return 0

        documents, metadatas, ids = [], [], []
        for card in new_cards:
            note_id = card["noteId"]
            front = card["fields"]["Front"]["value"]
            back = card["fields"]["Back"]["value"]
            combined_text = f"{front}\n{back}"
            documents.append(combined_text)
            metadatas.append({"front": front, "back": back, "note_id": note_id})
            ids.append(str(note_id))

        try:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logging.info(
                "Added %d new card(s) to deck '%s'.", len(new_cards), deck_name
            )
            return len(new_cards)
        except ChromaError as e:
            logging.error("Failed to add new cards for deck '%s': %s", deck_name, e)
            return 0

    def find_similar_cards(
        self, front: str, back: str, deck_name: str, threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Query the deck-specific collection for cards similar to the provided text.
        """
        combined_text = f"{front}\n{back}"
        collection = self.get_collection(deck_name)
        try:
            results = collection.query(query_texts=[combined_text], n_results=5)
        except ChromaError as e:
            logging.error("ChromaDB query failed for deck '%s': %s", deck_name, e)
            return []
        if not results.get("distances") or not results["distances"][0]:
            return []
        similar_cards = []
        for idx, distance in enumerate(results["distances"][0]):
            similarity = 1 - distance  # assuming distance in [0,1]
            if similarity >= threshold:
                metadata = results["metadatas"][0][idx]
                similar_cards.append(
                    {
                        "similarity": similarity,
                        "front": metadata["front"],
                        "back": metadata["back"],
                        "note_id": metadata["note_id"],
                    }
                )
        return similar_cards

    def add_single_card_to_vector_db(self, note_id: int, deck_name: str) -> bool:
        """
        Add or update a single note in the deck-specific collection.
        """
        resp = self.invoke_anki_connect("notesInfo", {"notes": [note_id]})
        if not resp.get("result"):
            logging.error(
                "No note info for note_id %s in deck '%s'.", note_id, deck_name
            )
            return False

        note_data = resp["result"][0]
        front = note_data["fields"]["Front"]["value"]
        back = note_data["fields"]["Back"]["value"]
        combined_text = f"{front}\n{back}"
        collection = self.get_collection(deck_name)
        try:
            collection.add(
                documents=[combined_text],
                metadatas=[{"front": front, "back": back, "note_id": note_id}],
                ids=[str(note_id)],
            )
            # Use debug level for per-note updates to keep output clean.
            logging.debug("Updated note_id %s in deck '%s'.", note_id, deck_name)
            return True
        except ChromaError as e:
            logging.error(
                "Failed to update note_id %s for deck '%s': %s", note_id, deck_name, e
            )
            return False

    def add_card_to_anki(self, deck_name: str, front: str, back: str) -> bool:
        """
        Add a new Basic card to Anki and sync it to the vector DB.
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
    """Anki Vector CLI: Manage a local vector DB of Anki cards."""
    pass


def prompt_for_deck(
    manager: AnkiVectorManager, deck_name: Optional[str] = None
) -> Optional[str]:
    """
    Prompt the user to choose a deck if one is not provided.
    """
    if deck_name:
        return deck_name

    resp = manager.invoke_anki_connect("deckNames", {})
    if "result" not in resp or not resp["result"]:
        click.echo("Couldn't retrieve deck names from Anki.")
        return None

    decks = resp["result"]
    click.echo("\nAvailable decks:")
    for i, d in enumerate(decks, 1):
        click.echo(f"{i}. {d}")

    while True:
        try:
            choice = click.prompt("\nChoose deck number", type=int)
            if 1 <= choice <= len(decks):
                return decks[choice - 1]
            click.echo("Invalid choice. Try again.")
        except click.Abort:
            return None


@cli.command(name="sync")
@click.argument("deck_name", required=False)
def sync(deck_name: str):
    """
    Synchronize the local vector DB with a given deck.
    """
    manager = AnkiVectorManager()
    deck = prompt_for_deck(manager, deck_name)
    if not deck:
        return
    new_count = manager.add_cards_to_vector_db(deck)
    click.echo(f"Synced deck '{deck}'. {new_count} new card(s) added.")


@cli.command(name="sync-all")
def sync_all():
    """
    Synchronize all decks from Anki to their respective vector DB collections.
    """
    manager = AnkiVectorManager()
    resp = manager.invoke_anki_connect("deckNames", {})
    if "result" not in resp or not resp["result"]:
        click.echo("Couldn't retrieve deck names from Anki.")
        return

    decks = resp["result"]
    total_new_cards = 0
    click.echo("Starting sync for all decks...\n")
    for deck in decks:
        click.echo(f"Syncing deck: {deck}")
        new_cards = manager.add_cards_to_vector_db(deck)
        click.echo(f" - {new_cards} new card(s) added.\n")
        total_new_cards += new_cards
    click.echo(f"Sync complete. Total new cards added: {total_new_cards}")


@cli.command(name="add-card")
@click.argument("deck_name", required=False)
def add_card(deck_name: str):
    """
    Create a new Basic card in Anki with a similarity check.
    """
    manager = AnkiVectorManager()
    deck = prompt_for_deck(manager, deck_name)
    if not deck:
        return

    click.echo("\nEnter FRONT text (Ctrl+D or Ctrl+Z to finish):")
    front_lines = []
    while True:
        try:
            line = input()
            front_lines.append(line)
        except EOFError:
            break
    front_text = "\n".join(front_lines)

    click.echo("\nEnter BACK text (Ctrl+D or Ctrl+Z to finish):")
    back_lines = []
    while True:
        try:
            line = input()
            back_lines.append(line)
        except EOFError:
            break
    back_text = "\n".join(back_lines)

    click.secho("\nSyncing deck before checking duplicates...", fg="yellow")
    manager.add_cards_to_vector_db(deck)

    similar = manager.find_similar_cards(front_text, back_text, deck, threshold=0.8)
    if similar:
        click.secho(f"\nFound {len(similar)} similar card(s):", fg="red")
        for i, card in enumerate(similar, 1):
            click.echo(f"{i}. Similarity: {card['similarity']:.2f}")
            click.echo(f"   Front: {card['front']}")
            click.echo(f"   Back:  {card['back']}")
        click.echo("\nOptions:")
        click.echo("  0 - Add as new card")
        for i in range(1, len(similar) + 1):
            click.echo(f"  {i} - Overwrite card #{i}")
        click.echo("  C - Cancel")
        choice = click.prompt("Choose option", type=str).upper()
        if choice == "C":
            click.echo("Canceled.")
            return
        elif choice.isdigit():
            choice_num = int(choice)
            if choice_num == 0:
                ok = manager.add_card_to_anki(deck, front_text, back_text)
                click.echo("New card added!" if ok else "Failed to add card.")
            elif 1 <= choice_num <= len(similar):
                note_id = similar[choice_num - 1]["note_id"]
                update_payload = {
                    "note": {
                        "id": note_id,
                        "fields": {"Front": front_text, "Back": back_text},
                    }
                }
                resp = manager.invoke_anki_connect("updateNoteFields", update_payload)
                if not resp.get("error"):
                    click.echo("Card updated successfully!")
                    manager.add_single_card_to_vector_db(note_id, deck)
                else:
                    click.echo(f"Failed to update card. Error: {resp['error']}")
            else:
                click.echo("Invalid choice. No changes made.")
    else:
        ok = manager.add_card_to_anki(deck, front_text, back_text)
        click.echo("Card added successfully!" if ok else "Failed to add card.")

    click.secho("\nSyncing deck after changes...", fg="yellow")
    manager.add_cards_to_vector_db(deck)
    click.secho("All done!", fg="green")


@cli.command(name="list-decks")
def list_decks():
    """
    List all decks from Anki via AnkiConnect.
    """
    manager = AnkiVectorManager()
    resp = manager.invoke_anki_connect("deckNames", {})
    if "result" in resp and resp["result"]:
        decks = resp["result"]
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
    Bulk-add cards from a text file.
    Cards are separated by 'SEPARATOR'. The first line of each card is the question; the rest is the answer.
    """
    manager = AnkiVectorManager()
    if not file_path:
        file_path = click.prompt(
            "Enter the path to the file", type=click.Path(exists=True)
        )
    deck = prompt_for_deck(manager, deck_name)
    if not deck:
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as e:
        click.echo(f"Error reading file: {e}")
        return

    cards_content = content.split("SEPARATOR")
    cards = []
    for card_content in cards_content:
        if not card_content.strip():
            continue
        lines = [line for line in card_content.strip().split("\n") if line.strip()]
        if not lines:
            continue
        question = lines[0].strip()
        answer = "\n".join(lines[1:]).strip()
        if question and answer:
            cards.append((question, answer))

    click.echo(f"Found {len(cards)} card(s) in file '{file_path}'.")
    click.secho("\nSyncing deck before import...", fg="yellow")
    manager.add_cards_to_vector_db(deck)

    added_count = 0
    skipped_count = 0
    with click.progressbar(cards, label="Processing cards", show_percent=True) as bar:
        for front, back in bar:
            similar = manager.find_similar_cards(front, back, deck, threshold=0.8)
            if any(s["similarity"] > 0.9 for s in similar):
                skipped_count += 1
                continue
            ok = manager.add_card_to_anki(deck, front, back)
            if ok:
                added_count += 1
            else:
                skipped_count += 1

    click.secho("\nSyncing deck after import...", fg="yellow")
    manager.add_cards_to_vector_db(deck)
    click.echo(f"\nImport complete. Added: {added_count} | Skipped: {skipped_count}")


if __name__ == "__main__":
    cli()
