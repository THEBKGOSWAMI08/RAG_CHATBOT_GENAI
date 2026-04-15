class ConversationMemory:
    def __init__(self, max_turns: int = 10):
        self.history: list[dict] = []
        self.max_turns = max_turns

    def add(self, question: str, answer: str) -> None:
        """Store a question-answer turn."""
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_history_text(self) -> str:
        """Format conversation history as a string for the LLM prompt."""
        if not self.history:
            return ""
        lines = []
        for turn in self.history:
            lines.append(f"User: {turn['question']}")
            lines.append(f"Assistant: {turn['answer']}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset conversation history."""
        self.history = []
