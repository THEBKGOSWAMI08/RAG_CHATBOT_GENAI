import os
from groq import Groq


class LLM:
    def __init__(self, api_key: str = None, model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.client = Groq(api_key=self.api_key)

    def generate(self, query: str, context: str) -> str:
        """Generate a response using retrieved context and the user query."""
        prompt = (
            "You are a helpful assistant. Answer the user's question based only on "
            "the provided context. If the context does not contain enough information, "
            "say so clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content
