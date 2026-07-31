"""Extract text from PDF files and split into tokenized chunks."""

from io import BytesIO

from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text content from a PDF file."""
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks for embedding.

    Uses word-based tokenization with configurable chunk size and overlap.
    """
    if not text.strip():
        return []

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap

    return chunks


def process_pdf(file_bytes: bytes, filename: str) -> list[dict]:
    """Extract text from PDF and return tokenized chunks with metadata."""
    text = extract_text_from_pdf(file_bytes)
    if not text.strip():
        raise ValueError("No text could be extracted from the PDF.")

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("PDF content could not be tokenized into chunks.")

    return [
        {
            "text": chunk,
            "metadata": {
                "source": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
            },
        }
        for i, chunk in enumerate(chunks)
    ]
