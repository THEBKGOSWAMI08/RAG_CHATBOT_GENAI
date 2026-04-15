import re
from langchain.schema import Document


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove special characters but keep punctuation
    text = re.sub(r"[^\w\s.,;:!?'\"-/()\[\]]", "", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def remove_headers_footers(text: str) -> str:
    """Remove common header/footer patterns like page numbers."""
    # Remove standalone page numbers
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    # Remove "Page X of Y" patterns
    text = re.sub(r"[Pp]age\s+\d+\s*(of\s+\d+)?", "", text)
    return text


def preprocess_documents(documents: list[Document]) -> list[Document]:
    """Apply all preprocessing steps to a list of documents."""
    processed = []
    for doc in documents:
        text = doc.page_content
        text = remove_headers_footers(text)
        text = clean_text(text)
        if len(text) > 10:  # skip near-empty pages
            processed.append(
                Document(page_content=text, metadata=doc.metadata)
            )
    return processed
