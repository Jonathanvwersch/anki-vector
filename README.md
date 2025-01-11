# Anki Vector Tool

A command-line tool that enhances Anki flashcard management using vector similarity search. This tool helps prevent duplicate cards by finding semantically similar existing cards before adding new ones.

This is a tool I built for my own workflow. It may not be useful for others but I use it all the time for my Computer Science Anki Deck.

## Prerequisites

- Anki with AnkiConnect plugin installed and running
- Python 3.7+

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install click requests chromadb
```
3. Ensure Anki is running with AnkiConnect plugin installed and configured

## Setup

The tool requires AnkiConnect to be properly configured in Anki:

1. Install AnkiConnect from Anki's add-on manager (Tools -> Add-ons -> Get Add-ons...)
2. Restart Anki
3. Make sure Anki is running when using this tool

## Usage

The tool provides three main commands:

### 1. Sync Cards

Synchronizes cards from an Anki deck to the local vector database:

```bash
python anki_vector_tool.py sync-cards
```
or
```bash
python anki_vector_tool.py sync-cards "Your Deck Name"
```

If you don't specify a deck name, the tool will:
1. Show a numbered list of all your available decks
2. Let you choose a deck by entering its number
3. Sync that deck to the vector database

### 2. Add Card

Adds a new card to a deck with similarity checking:

```bash
python anki_vector_tool.py add-card
```
or
```bash
python anki_vector_tool.py add-card "Your Deck Name"
```

If you don't specify a deck name, the tool will:
1. Show a numbered list of all your available decks
2. Let you choose a deck by entering its number
3. Prompt for the card's front and back content
4. Check for similar cards and show options

### 3. List Decks

Shows all available decks in your Anki:

```bash
python anki_vector_tool.py list-decks
```

### 4. Add Cards from File

Add multiple cards from a text file:

```bash
python anki_vector_tool.py add-cards-from-file
```
or
```bash
python anki_vector_tool.py add-cards-from-file ./decks-to-upload/cards.txt
```
or
```bash
python anki_vector_tool.py add-cards-from-file cards.txt "Your Deck Name"
```

If you don't provide arguments, the tool will:
1. Prompt for the file path
2. Show a numbered list of available decks
3. Let you choose a deck by number

The text file should have one card per line in the format:
```
front|||back
```

If you don't specify a deck name, the tool will:
1. Show a numbered list of all your available decks
2. Let you choose a deck by entering its number
3. Process each card in the file, showing:
   - Similar cards if found
   - Options to add, replace, skip, or quit
   - Progress through the file

Options for each card:
- 0: Add as new card
- 1-N: Replace existing similar card
- S: Skip this card
- Q: Quit processing remaining cards

## Features

- Interactive deck selection - no need to type deck names
- Vector similarity search using ChromaDB
- Persistent storage of card embeddings
- Integration with AnkiConnect for seamless Anki interaction
- Similarity threshold customization
- Duplicate detection based on semantic meaning rather than exact text matching
- Command-line interface with interactive prompts
- Automatic synchronization of new cards to the vector database

## Technical Details

- Uses ChromaDB's default embedding function
- Stores embeddings in a local ChromaDB database
- Communicates with Anki through the AnkiConnect API
- Default similarity threshold of 0.8 (configurable)
- Supports Basic note type cards

## Troubleshooting

1. If you get connection errors, ensure:
   - Anki is running
   - AnkiConnect is properly installed
   - AnkiConnect is listening on port 8765 (default)

2. If cards aren't being found:
   - Run the `sync-cards` command to update the vector database
   - Check if the deck name exactly matches your Anki deck name
