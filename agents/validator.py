"""
agents/validator.py
Strands Validator Agent — responsible for generating the final answer
from retrieved context and scoring its own response quality.
"""

from strands import Agent, tool


@tool
def score_answer(answer: str, context: str, question: str) -> str:
    """
    Score the faithfulness and relevance of an answer against retrieved context.

    Args:
        answer: The generated answer to evaluate
        context: The retrieved context used to generate the answer
        question: The original user question

    Returns:
        JSON string with faithfulness and relevance scores (0.0 - 1.0)
    """
    import json

    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    overlap = len(answer_words & context_words)
    faithfulness = min(overlap / max(len(answer_words), 1), 1.0)

    question_words = set(question.lower().split()) - {"how", "what", "why", "when", "where", "do", "i", "a", "the"}
    answer_relevance = len(question_words & answer_words) / max(len(question_words), 1)
    answer_relevance = min(answer_relevance, 1.0)

    return json.dumps({
        "faithfulness": round(faithfulness, 3),
        "answer_relevance": round(answer_relevance, 3),
        "overall": round((faithfulness + answer_relevance) / 2, 3)
    })


VALIDATOR_SYSTEM_PROMPT = """You are a Validator agent for Amazon Seller Support.

You receive:
1. A user question
2. Retrieved context from the seller knowledge base

Your job is to:
1. Generate a clear, accurate, helpful answer using ONLY the provided context
2. If the context doesn't contain enough information, say so honestly
3. Use the score_answer tool to evaluate your answer's quality
4. Return both the answer and the quality scores

Format your response as:
ANSWER: <your answer here>
SOURCES: <comma-separated source names>
SCORES: <call score_answer tool and include results>

Be concise and direct. Sellers need quick, actionable answers."""


def create_validator_agent(model_id: str = "mistral") -> Agent:
    """Create and return a configured Validator agent."""
    return Agent(
        model=f"ollama/{model_id}",
        system_prompt=VALIDATOR_SYSTEM_PROMPT,
        tools=[score_answer],
    )


def run_validator(question: str, context: str, model_id: str = "mistral") -> dict:
    """Run the validator agent to generate and score a final answer."""
    agent = create_validator_agent(model_id)

    prompt = f"""Question: {question}

Retrieved Context:
{context}

Generate a helpful answer using only the context above, then score it."""

    response = agent(prompt)

    return {
        "question": question,
        "response": str(response),
        "agent": "validator"
    }