"""CLI tool for managing Anki flashcards with vector similarity search and duplicate detection."""

import json
import logging
import os
from typing import Any, Dict, List

import chromadb
import click
import requests
from chromadb.utils import embedding_functions

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)


class AnkiVectorTool:
    """Tool for managing Anki cards with vector similarity search."""

    def __init__(self, db_path: str = "./vector_db"):
        """Initialize with ChromaDB path."""
        self.db_path = db_path
        os.environ["ANONYMIZED_TELEMETRY"] = "False"

        try:
            self.chroma_client = chromadb.PersistentClient(path=db_path)
        except chromadb.errors.ChromaError as e:
            logging.error("Failed to initialize ChromaDB client: %s", e)
            raise

        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            name="anki_cards", embedding_function=self.embedding_function
        )

    def invoke_anki_connect(
        self, action: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send a request to the AnkiConnect plugin.

        :param action: The AnkiConnect action, e.g. "findNotes" or "addNote".
        :param params: The parameters for that action.
        :return: The JSON response from AnkiConnect, or an error message if it fails.
        """
        endpoint = "http://localhost:8765"
        payload = {
            "action": action,
            "version": 6,  # current stable AnkiConnect version
            "params": params,
        }
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
            return {"error": "Invalid JSON response from AnkiConnect"}

    def get_deck_cards(self, deck_name: str) -> List[Dict[str, Any]]:
        """Retrieve all cards from specified deck."""
        query = f'deck:"{deck_name}"'
        logging.info("Searching for notes with query: %s", query)

        find_notes_response = self.invoke_anki_connect("findNotes", {"query": query})

        note_ids = find_notes_response.get("result", [])
        if not note_ids:
            logging.info("No notes found for deck: %s", deck_name)
            return []

        notes_info_response = self.invoke_anki_connect("notesInfo", {"notes": note_ids})
        return notes_info_response.get("result", [])

    def add_cards_to_vector_db(self, deck_name: str) -> int:
        """
        Sync cards from the specified deck to the local ChromaDB.
        Adds new cards and removes deleted ones.
        """
        # Get current cards from Anki
        anki_cards = self.get_deck_cards(deck_name)
        anki_note_ids = {str(card["noteId"]) for card in anki_cards}

        # Get existing cards from vector DB
        try:
            existing_docs = self.collection.get()
            existing_ids = set(existing_docs.get("ids", []))
        except chromadb.errors.ChromaError as e:
            logging.error("Failed to fetch documents from ChromaDB: %s", e)
            return 0

        # Find cards to add and remove
        ids_to_add = anki_note_ids - existing_ids
        ids_to_remove = existing_ids - anki_note_ids

        # Remove deleted cards from vector DB
        if ids_to_remove:
            try:
                self.collection.delete(ids=list(ids_to_remove))
                logging.info(
                    "Removed %d deleted cards from vector database.", len(ids_to_remove)
                )
            except chromadb.errors.ChromaError as e:
                logging.error("Failed to remove deleted cards from ChromaDB: %s", e)

        # Add new cards
        new_cards = [card for card in anki_cards if str(card["noteId"]) in ids_to_add]
        if not new_cards:
            return 0

        documents = []
        metadatas = []
        ids = []

        for card in new_cards:
            front_text = card["fields"]["Front"]["value"]
            back_text = card["fields"]["Back"]["value"]
            card_text = f"{front_text} {back_text}"

            documents.append(card_text)
            metadatas.append(
                {
                    "front": front_text,
                    "back": back_text,
                    "note_id": card["noteId"],
                }
            )
            ids.append(str(card["noteId"]))

        try:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logging.info("Added %d new cards to vector database.", len(new_cards))
            return len(new_cards)
        except chromadb.errors.ChromaError as e:
            logging.error("Failed to add cards to ChromaDB: %s", e)
            return 0

    def find_similar_cards(
        self, front: str, back: str, threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Find similar cards in the vector DB by comparing embeddings.

        :param front: Proposed front text.
        :param back: Proposed back text.
        :param threshold: Minimum similarity threshold to be considered a "similar" card.
        :return: List of similar cards with similarity score, front, and back.
        """
        query_text = f"{front} {back}"
        try:
            results = self.collection.query(query_texts=[query_text], n_results=5)
        except chromadb.errors.ChromaError as e:
            logging.error("ChromaDB query failed: %s", e)
            return []

        # If no distances returned, means no results.
        if not results["distances"][0]:
            return []

        similar_cards = []
        for i, distance in enumerate(results["distances"][0]):
            similarity = 1 - distance  # Convert distance to similarity score
            if similarity >= threshold:
                metadatum = results["metadatas"][0][i]
                similar_cards.append(
                    {
                        "similarity": similarity,
                        "front": metadatum["front"],
                        "back": metadatum["back"],
                        "note_id": metadatum["note_id"],
                    }
                )
        return similar_cards

    def add_single_card_to_vector_db(self, note_id: int) -> bool:
        """Add a single card to the vector database by note ID."""
        response = self.invoke_anki_connect("notesInfo", {"notes": [note_id]})
        if "result" not in response or not response["result"]:
            return False

        card = response["result"][0]
        front_text = card["fields"]["Front"]["value"]
        back_text = card["fields"]["Back"]["value"]
        card_text = f"{front_text} {back_text}"

        try:
            self.collection.add(
                documents=[card_text],
                metadatas=[
                    {"front": front_text, "back": back_text, "note_id": card["noteId"]}
                ],
                ids=[str(card["noteId"])],
            )
            return True
        except chromadb.errors.ChromaError as e:
            logging.error("Failed to add card to ChromaDB: %s", e)
            return False

    def add_card_to_anki(self, deck_name: str, front: str, back: str) -> bool:
        """Add a new Basic note/card to Anki using AnkiConnect."""
        response = self.invoke_anki_connect(
            "addNote",
            {
                "note": {
                    "deckName": deck_name,
                    "modelName": "Basic",
                    "fields": {"Front": front, "Back": back},
                    "options": {"allowDuplicate": True},
                    "tags": [],
                }
            },
        )
        if "error" in response and response["error"] is not None:
            logging.error("Failed to add card to Anki. Response: %s", response)
            return False
        if "result" not in response:
            logging.error("Unexpected response from Anki: %s", response)
            return False

        # Add just this card to vector db
        note_id = response["result"]
        if self.add_single_card_to_vector_db(note_id):
            logging.info("Added new card to vector database.")
        return True


@click.group()
def cli():
    """Anki Vector Tool - Manage Anki cards with vector similarity search."""


@cli.command()
@click.argument("deck_name", required=False)
def sync_cards(deck_name: str = None):
    """Synchronize cards from an Anki deck to the local vector database."""
    tool = AnkiVectorTool()

    if deck_name is None:
        response = tool.invoke_anki_connect("deckNames", {})
        if "result" not in response:
            click.echo("Failed to get deck list")
            return

        decks = response["result"]
        click.echo("\nAvailable decks:")
        for idx, deck in enumerate(decks, 1):
            click.echo(f"{idx}. {deck}")

        while True:
            try:
                choice = click.prompt("\nChoose deck number", type=int)
                if 1 <= choice <= len(decks):
                    deck_name = decks[choice - 1]
                    break
                click.echo("Invalid choice. Please try again.")
            except click.Abort:
                return

    new_cards_count = tool.add_cards_to_vector_db(deck_name)
    click.echo(f"Added {new_cards_count} new cards to vector database.")


@cli.command()
@click.argument("deck_name", required=False)
def add_card(deck_name: str = None):
    """Add a new card to a deck with similarity checking."""
    tool = AnkiVectorTool()

    if deck_name is None:
        response = tool.invoke_anki_connect("deckNames", {})
        if "result" not in response:
            click.echo("Failed to get deck list")
            return

        decks = response["result"]
        click.echo("\nAvailable decks:")
        for idx, deck in enumerate(decks, 1):
            click.echo(f"{idx}. {deck}")

        while True:
            try:
                choice = click.prompt("\nChoose deck number", type=int)
                if 1 <= choice <= len(decks):
                    deck_name = decks[choice - 1]
                    break
                click.echo("Invalid choice. Please try again.")
            except click.Abort:
                return

    click.echo("\nEnter the front of the card (press Ctrl+D or Ctrl+Z when done):")
    front_lines = []
    while True:
        try:
            line = input()
            front_lines.append(line)
        except EOFError:
            break
    front = "\n".join(front_lines)

    click.echo("\nEnter the back of the card (press Ctrl+D or Ctrl+Z when done):")
    back_lines = []
    while True:
        try:
            line = input()
            back_lines.append(line)
        except EOFError:
            break
    back = "\n".join(back_lines)

    click.secho("\n🔄 Syncing deck before operation...", fg="yellow", bold=True)
    tool.add_cards_to_vector_db(deck_name)
    click.secho("✅ Initial sync complete\n", fg="green")

    # Check for similar cards
    similar_cards = tool.find_similar_cards(front, back)

    if similar_cards:
        click.echo("\nSimilar cards found:")
        for idx, card in enumerate(similar_cards, 1):
            click.echo(f"\n{idx}. Similarity: {card['similarity']:.2%}")
            click.echo(f"   Front: {card['front']}")
            click.echo(f"   Back: {card['back']}")

        click.echo("\nOptions:")
        click.echo("0. Add as new card")
        for i in range(1, len(similar_cards) + 1):
            click.echo(f"{i}. Replace card #{i} shown above")
        click.echo("C. Cancel")

        choice = click.prompt("Choose action", type=str).upper()

        if choice == "C":
            click.echo("Card addition cancelled.")
            return

        if choice.isdigit():
            choice = int(choice)
            if choice == 0:
                # Add as new card
                if tool.add_card_to_anki(deck_name, front, back):
                    click.echo("Card added to Anki successfully!")
                else:
                    click.echo("Failed to add card to Anki.")
            elif 1 <= choice <= len(similar_cards):
                # Override existing card
                note_id = similar_cards[choice - 1]["note_id"]
                response = tool.invoke_anki_connect(
                    "updateNoteFields",
                    {"note": {"id": note_id, "fields": {"Front": front, "Back": back}}},
                )
                if "error" not in response or response["error"] is None:
                    click.echo("Card updated successfully!")
                    tool.add_single_card_to_vector_db(note_id)
                else:
                    click.echo("Failed to update card.")
            else:
                click.echo("Invalid choice.")
                return
    else:
        # No similar cards found, add automatically
        if tool.add_card_to_anki(deck_name, front, back):
            click.echo("Card added successfully!")
        else:
            click.echo("Failed to add card.")

    click.secho("\n🔄 Syncing deck after operation...", fg="yellow", bold=True)
    tool.add_cards_to_vector_db(deck_name)
    click.secho("✅ Final sync complete\n", fg="green")


@cli.command()
def list_decks():
    """List all available decks in Anki."""
    tool = AnkiVectorTool()
    response = tool.invoke_anki_connect("deckNames", {})
    if "result" in response:
        decks = response["result"]
        click.echo("\nAvailable decks:")
        for deck in decks:
            click.echo(f"  - {deck}")
    else:
        click.echo("Failed to get deck list")


@cli.command()
@click.argument("file_path", type=click.Path(exists=True), required=False)
@click.argument("deck_name", required=False)
def add_cards_from_file(file_path: str = None, deck_name: str = None):
    """Add multiple cards from a text file. Questions should end with '?' and be followed by their answers."""
    tool = AnkiVectorTool()

    if file_path is None:
        file_path = click.prompt(
            "Enter path to cards file", type=click.Path(exists=True)
        )

    if deck_name is None:
        response = tool.invoke_anki_connect("deckNames", {})
        if "result" not in response:
            click.echo("Failed to get deck list")
            return

        decks = response["result"]
        click.echo("\nAvailable decks:")
        for idx, deck in enumerate(decks, 1):
            click.echo(f"{idx}. {deck}")

        while True:
            try:
                choice = click.prompt("\nChoose deck number", type=int)
                if 1 <= choice <= len(decks):
                    deck_name = decks[choice - 1]
                    break
                click.echo("Invalid choice. Please try again.")
            except click.Abort:
                return

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Split content into lines
    lines = content.split("\n")
    cards = []
    current_question = None
    current_answer_lines = []

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:  # Skip empty lines
            continue

        # If line ends with ? and isn't indented, it's a new question
        if stripped_line.endswith("?") and not line.startswith(" "):
            # Save the previous card if we have one
            if current_question is not None:
                cards.append(
                    (current_question, "\n".join(current_answer_lines).strip())
                )
            # Start a new card
            current_question = stripped_line
            current_answer_lines = []
        else:
            # Add to current answer if we have a question
            if current_question is not None:
                current_answer_lines.append(line)

    # Don't forget to add the last card
    if current_question is not None and current_answer_lines:
        cards.append((current_question, "\n".join(current_answer_lines).strip()))

    click.echo(f"\nFound {len(cards)} cards in file.")
    click.secho("\n🔄 Syncing deck before processing...", fg="yellow", bold=True)
    tool.add_cards_to_vector_db(deck_name)
    click.secho("✅ Initial sync complete\n", fg="green")

    added_count = 0
    skipped_count = 0
    with click.progressbar(cards, label="Processing cards") as card_list:
        for front, back in card_list:
            # Check for similar cards
            similar_cards = tool.find_similar_cards(front, back)

            if similar_cards:
                # If very similar (>90%), skip
                if any(card["similarity"] > 0.9 for card in similar_cards):
                    skipped_count += 1
                    continue

            # Add as new card
            if tool.add_card_to_anki(deck_name, front, back):
                added_count += 1
            else:
                skipped_count += 1

    click.secho("\n🔄 Syncing deck after processing...", fg="yellow", bold=True)
    tool.add_cards_to_vector_db(deck_name)
    click.secho("✅ Final sync complete\n", fg="green")

    click.echo(f"\nProcessing complete:")
    click.echo(f"- Added: {added_count} cards")
    click.echo(f"- Skipped: {skipped_count} cards (due to duplicates or errors)")


if __name__ == "__main__":
    cli()
