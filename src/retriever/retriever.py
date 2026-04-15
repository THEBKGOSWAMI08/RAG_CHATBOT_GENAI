from src.vectorstore.vectordb import VectorDB


class Retriever:
    def __init__(self, vector_db: VectorDB, top_k: int = 5):
        self.vector_db = vector_db
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve the most relevant documents for a query."""
        results = self.vector_db.query(query, top_k=self.top_k)
        return results

    def get_context(self, query: str) -> str:
        """Retrieve documents and format them as a context string."""
        results = self.retrieve(query)
        context_parts = []
        for i, doc in enumerate(results, 1):
            context_parts.append(f"[{i}] {doc['content']}")
        return "\n\n".join(context_parts)
