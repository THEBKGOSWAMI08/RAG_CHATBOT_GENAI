from src.loaders.loader import load_documents
from src.preprocessing.preprocess import preprocess_documents
from src.chunking.chunking import chunk_documents
from src.vectorstore.vectordb import VectorDB
from src.retriever.retriever import Retriever
from src.reranker.reranker import Reranker
from src.llm.llm import LLM


class RAGPipeline:
    def __init__(self, groq_api_key: str):
        self.vector_db = VectorDB()
        self.retriever = Retriever(self.vector_db, top_k=5)
        self.reranker = Reranker()
        self.llm = LLM(api_key=groq_api_key)

    def ingest(self, path: str) -> int:
        """Load, preprocess, chunk, and store documents from a file or directory."""
        documents = load_documents(path)
        documents = preprocess_documents(documents)
        chunks = chunk_documents(documents)
        self.vector_db.add_documents(chunks)
        return len(chunks)

    def query(self, question: str, use_reranker: bool = True) -> str:
        """Run the full RAG pipeline: retrieve, rerank, and generate."""
        retrieved = self.retriever.retrieve(question)

        if use_reranker and retrieved:
            retrieved = self.reranker.rerank(question, retrieved, top_k=3)

        context_parts = []
        for i, doc in enumerate(retrieved, 1):
            context_parts.append(f"[{i}] {doc['content']}")
        context = "\n\n".join(context_parts)

        answer = self.llm.generate(question, context)
        return answer

    def reset(self) -> None:
        """Clear the vector store."""
        self.vector_db.reset()
