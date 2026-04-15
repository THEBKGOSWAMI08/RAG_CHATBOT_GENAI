from src.loaders.loader import load_documents
from src.preprocessing.preprocess import preprocess_documents
from src.chunking.chunking import chunk_documents
from src.vectorstore.vectordb import VectorDB
from src.retriever.retriever import Retriever
from src.reranker.reranker import Reranker
from src.llm.llm import LLM
from src.memory.memory import ConversationMemory


class RAGPipeline:
    def __init__(self, groq_api_key: str):
        self.vector_db = VectorDB()
        self.retriever = Retriever(self.vector_db, top_k=5)
        self.reranker = Reranker()
        self.llm = LLM(api_key=groq_api_key)
        self.memory = ConversationMemory(max_turns=10)

    def ingest(self, path: str) -> int:
        """Load, preprocess, chunk, and store documents from a file or directory."""
        documents = load_documents(path)
        documents = preprocess_documents(documents)
        chunks = chunk_documents(documents)
        self.vector_db.add_documents(chunks)
        return len(chunks)

    def query(self, question: str, use_reranker: bool = True) -> str:
        """Run the full RAG pipeline: retrieve, rerank, augment, and generate."""
        # Retrieval
        retrieved = self.retriever.retrieve(question)

        # Re-ranking
        if use_reranker and retrieved:
            retrieved = self.reranker.rerank(question, retrieved, top_k=3)

        # Augmentation: build context from retrieved docs
        context_parts = []
        for i, doc in enumerate(retrieved, 1):
            context_parts.append(f"[{i}] {doc['content']}")
        context = "\n\n".join(context_parts)

        # Memory: inject conversation history
        history = self.memory.get_history_text()

        # Output generation
        answer = self.llm.generate(question, context, history=history)

        # Save turn to memory
        self.memory.add(question, answer)
        return answer

    def reset(self) -> None:
        """Clear the vector store and conversation memory."""
        self.vector_db.reset()
        self.memory.clear()
