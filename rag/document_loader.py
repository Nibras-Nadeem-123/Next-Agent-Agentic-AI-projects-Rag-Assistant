"""Document loading utilities for PDF and TXT files."""

from typing import BinaryIO
from PyPDF2 import PdfReader
import io


def load_pdf(file: BinaryIO) -> str:
    """Extract text content from a PDF file.

    Args:
        file: A file-like object containing PDF data.

    Returns:
        Extracted text content from all pages.
    """
    reader = PdfReader(file)
    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    return "\n\n".join(text_parts)


def load_txt(file: BinaryIO) -> str:
    """Read text content from a plain text file.

    Args:
        file: A file-like object containing text data.

    Returns:
        The text content of the file.
    """
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    return content


def load_document(file: BinaryIO, filename: str) -> str:
    """Load a document based on its file extension.

    Args:
        file: A file-like object containing the document data.
        filename: The name of the file (used to determine type).

    Returns:
        Extracted text content from the document.

    Raises:
        ValueError: If the file type is not supported.
    """
    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf"):
        return load_pdf(file)
    elif filename_lower.endswith(".txt"):
        return load_txt(file)
    else:
        raise ValueError(f"Unsupported file type: {filename}. Supported types: .pdf, .txt")
