from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of text strings."""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query string."""
        embedding = self.model.encode([query])
        return embedding[0].tolist()

    def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        """Generate embeddings for a list of Document objects."""
        texts = [doc.page_content for doc in documents]
        return self.embed_texts(texts)
