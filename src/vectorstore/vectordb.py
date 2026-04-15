import chromadb
from langchain.schema import Document

from src.embedding.embedder import Embedder


class VectorDB:
    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_directory: str = "./chroma_db",
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = Embedder()

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents and their embeddings to the vector store."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings = self.embedder.embed_texts(texts)
        ids = [f"doc_{i}" for i in range(len(texts))]

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Query the vector store and return the most similar documents."""
        query_embedding = self.embedder.embed_query(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = []
        for i in range(len(results["documents"][0])):
            docs.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return docs

    def reset(self) -> None:
        """Delete all documents from the collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
