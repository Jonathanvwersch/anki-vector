# Anki Vector Tool

A command-line tool that enhances Anki flashcard management using vector similarity search. This tool helps prevent duplicate cards by finding semantically similar existing cards before adding new ones. I mainly built this to match my own personal use case.

## Features

- **Smart Duplicate Detection**: Uses vector similarity search to find semantically similar cards, not just exact matches
- **Fast Incremental Sync**: Only processes new or changed cards, not the entire deck each time
- **Parallel Processing**: Uses multi-threading to speed up card processing
- **Interactive Interface**: Easy-to-use CLI with deck selection and card management
- **Bulk Import**: Import multiple cards from text files with duplicate detection
- **Efficient Storage**: Uses ChromaDB for fast vector storage and similarity search

## Prerequisites

- Anki with the AnkiConnect plugin installed and running
- Python 3.7+

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd anki-vector
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Ensure Anki is running with AnkiConnect plugin installed and configured

## Setup

1. Install AnkiConnect from Anki's add-on manager (Tools → Add-ons → Get Add-ons...)
2. Restart Anki
3. Make sure Anki is running when using this tool

## Commands

### Sync a Deck
Incrementally synchronize cards from an Anki deck to the vector database:
```bash
python anki_vector_tool.py sync [deck-name]
```
- Only processes new or changed cards
- Removes cards that were deleted from Anki
- Much faster than full sync

### Add a Single Card
Add a new card with duplicate detection:
```bash
python anki_vector_tool.py add-card [deck-name]
```
- Checks for similar existing cards before adding
- Shows similarity scores and card content
- Options to add anyway, update existing, or cancel

### List Decks
Show all available Anki decks:
```bash
python anki_vector_tool.py list-decks
```

### Bulk Import from File
Add multiple cards from a text file:
```bash
python anki_vector_tool.py add-from-file [file-path] [deck-name]
```

File format:
```
What is a binary search?
A search algorithm that finds the position of a target value within a sorted array.
It works by repeatedly dividing the search space in half.

SEPARATOR

What is its time complexity?
O(log n) for sorted arrays.
Best case: O(1) when middle element is the target.
Worst case: O(log n) when target is at the end of a partition.
```

- Cards are separated by 'SEPARATOR'
- First line is the front of the card
- Remaining lines until SEPARATOR are the back
- Automatic duplicate detection for each card

### Sync All Decks
Sync all your Anki decks at once:
```bash
python anki_vector_tool.py sync-all
```

## Performance

- Uses incremental sync to minimize processing time
- Parallel processing for batch operations
- Efficient vector storage and retrieval with ChromaDB
- Only processes new or modified cards

## Technical Details

- Uses ChromaDB's default embedding function for vector similarity
- Stores embeddings in a local ChromaDB database (`./vector_db` by default)
- Communicates with Anki through the AnkiConnect API
- Default similarity threshold: 0.9 (configurable)
- Supports Basic note type
- Parallel processing with 4 worker threads

## Troubleshooting

1. Connection errors:
   - Ensure Anki is running
   - Confirm AnkiConnect is properly installed
   - Check if AnkiConnect is listening on port 8765 (default)

2. Performance issues:
   - The first sync might take longer as it needs to process all cards
   - Subsequent syncs are much faster as they only process new/changed cards
   - Try adjusting batch size if needed (default: 20 cards per batch)

3. Duplicate detection issues:
   - Adjust similarity threshold (default: 0.9) if getting too many/few matches
   - Check if cards are being properly added to the vector database
   - Try running a full sync if vector DB seems out of sync