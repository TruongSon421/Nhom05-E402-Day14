"""Fallback LLM response for local/prod smoke tests."""


def ask(question: str) -> str:
    cleaned = question.strip()
    return (
        "Mock answer: "
        f"I received your question '{cleaned}'. "
        "Set OPENAI_API_KEY to integrate a real model."
    )
