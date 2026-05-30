"""Text splitting utilities for chunking documents."""


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: The text to split.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    # Clean the text
    text = text.strip()

    # If text is shorter than chunk_size, return as single chunk
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        # Calculate end position
        end = start + chunk_size

        # If this is not the last chunk, try to break at a sentence or word boundary
        if end < len(text):
            # Look for sentence boundaries (., !, ?) within the last 100 characters
            search_start = max(end - 100, start)
            last_period = text.rfind(".", search_start, end)
            last_exclaim = text.rfind("!", search_start, end)
            last_question = text.rfind("?", search_start, end)
            last_newline = text.rfind("\n", search_start, end)

            # Find the best break point
            break_point = max(last_period, last_exclaim, last_question, last_newline)

            if break_point > start:
                end = break_point + 1
            else:
                # Fall back to word boundary
                last_space = text.rfind(" ", search_start, end)
                if last_space > start:
                    end = last_space

        # Extract chunk and add to list
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position, accounting for overlap
        start = end - chunk_overlap if end < len(text) else len(text)

    return chunks
