from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[dict], top_k: int = 3) -> list[dict]:
        """Rerank retrieved documents using a cross-encoder model."""
        if not documents:
            return []

        pairs = [[query, doc["content"]] for doc in documents]
        scores = self.model.predict(pairs)

        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])

        ranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]
