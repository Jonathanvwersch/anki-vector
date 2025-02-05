# Anki Vector Tool

A command-line tool that enhances Anki flashcard management using vector similarity search. This tool helps prevent duplicate cards by finding semantically similar existing cards before adding new ones.

This is a tool I built for my own workflow. It may not be useful for others, but I use it all the time for my Computer Science Anki Deck.

## Prerequisites

- Anki with the AnkiConnect plugin installed and running
- Python 3.7+

## Installation

1. Clone this repository (or place the script somewhere on your system)
2. Install the required packages:
```bash
pip install click requests chromadb
```
3. Ensure Anki is running with the AnkiConnect plugin installed and configured

## Setup

The tool requires AnkiConnect to be properly configured in Anki:

1. Install AnkiConnect from Anki's add-on manager (Tools → Add-ons → Get Add-ons...)
2. Restart Anki
3. Make sure Anki is running when using this tool

## Usage

The tool provides four main commands:

### 1. Sync
Synchronizes cards from an Anki deck to the local vector database:

```bash
python anki_vector_tool.py sync
```
or
```bash
python anki_vector_tool.py sync "Your Deck Name"
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
3. Prompt for the card's front and back content (supports multiline input)
4. Check for similar cards and show options to add, replace, or cancel

### 3. List Decks
Shows all available decks in your Anki:

```bash
python anki_vector_tool.py list-decks
```

### 4. Add From File
Add multiple cards from a text file:

```bash
python anki_vector_tool.py add-from-file
```
or
```bash
python anki_vector_tool.py add-from-file ./decks-to-upload/cards.txt
```
or
```bash
python anki_vector_tool.py add-from-file cards.txt "Your Deck Name"
```

If you don't provide arguments, the tool will:
1. Prompt for the file path
2. Show a numbered list of available decks
3. Let you choose a deck by number

File format:
- Cards are separated by the word "SEPARATOR" on its own line
- First line of each section is the question
- Everything after the first line until SEPARATOR is the answer
- Empty lines are allowed within questions and answers

Example:
```
What is a binary search?
A search algorithm that finds the position of a target value within a sorted array.
It works by repeatedly dividing the search space in half.

SEPARATOR

What is its time complexity?
O(log n) for sorted arrays.
Best case: O(1) when middle element is the target.
Worst case: O(log n) when target is at the end of a partition.

SEPARATOR

What are the requirements for binary search?
1. Array must be sorted
2. Random access to elements (array or similar data structure)
3. Clear ordering relationship between elements

SEPARATOR
```

## Features

- Interactive deck selection – no need to manually type deck names
- Vector similarity search using ChromaDB to find semantically similar cards
- Persistent storage of card embeddings in a local ChromaDB database
- Integration with AnkiConnect for seamless Anki interaction
- Similarity threshold customization (default: 0.8)
- Duplicate detection based on semantic meaning rather than exact text matching
- Command-line interface with interactive prompts
- Automatic synchronization of new cards to the vector database

## Technical Details

- Uses ChromaDB's default embedding function
- Stores embeddings in a local ChromaDB database (./vector_db by default)
- Communicates with Anki through the AnkiConnect API
- Default similarity threshold is 0.8 (configurable)
- Supports Basic note type for new cards

## Troubleshooting

1. Connection errors:
   - Ensure Anki is running
   - Confirm AnkiConnect is properly installed
   - Make sure AnkiConnect is listening on port 8765 (default)

2. Cards aren't being found:
   - Run the `sync` command to update the vector database
   - Check if the deck name matches exactly