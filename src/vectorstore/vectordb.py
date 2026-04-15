import os
import pickle
import numpy as np
import faiss
from langchain.schema import Document

from src.embedding.embedder import Embedder


class VectorDB:
    def __init__(self, persist_directory: str = "./faiss_db"):
        self.persist_directory = persist_directory
        self.embedder = Embedder()
        self.index = None
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self.dimension = 384  # all-MiniLM-L6-v2 output dimension

        os.makedirs(persist_directory, exist_ok=True)
        self._load()

    def _index_path(self):
        return os.path.join(self.persist_directory, "index.faiss")

    def _docs_path(self):
        return os.path.join(self.persist_directory, "docs.pkl")

    def _load(self):
        """Load existing FAISS index and documents from disk if available."""
        if os.path.exists(self._index_path()) and os.path.exists(self._docs_path()):
            self.index = faiss.read_index(self._index_path())
            with open(self._docs_path(), "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.metadatas = data["metadatas"]
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

    def _save(self):
        """Persist FAISS index and documents to disk."""
        faiss.write_index(self.index, self._index_path())
        with open(self._docs_path(), "wb") as f:
            pickle.dump(
                {"documents": self.documents, "metadatas": self.metadatas}, f
            )

    def add_documents(self, documents: list[Document]) -> None:
        """Embed and add documents to the FAISS index."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings = self.embedder.embed_texts(texts)

        vectors = np.array(embeddings, dtype="float32")
        self.index.add(vectors)
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)
        self._save()

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Search the FAISS index for the most similar documents."""
        if self.index.ntotal == 0:
            return []

        query_embedding = np.array(
            [self.embedder.embed_query(query_text)], dtype="float32"
        )
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append({
                    "content": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "distance": float(dist),
                })
        return results

    def reset(self) -> None:
        """Clear the FAISS index and all stored documents."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadatas = []
        self._save()
