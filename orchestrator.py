"""
orchestrator.py
LangGraph state machine orchestrating the Retriever → Validator pipeline.

Flow:
  START → retrieve → validate → END
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from agents.retriever import run_retriever
from agents.validator import run_validator


class AgentState(TypedDict):
    """State passed between nodes in the LangGraph pipeline."""
    question: str
    retrieved_context: str
    final_answer: str
    sources: list[str]
    validation_score: float
    error: str


def retrieve_node(state: AgentState) -> AgentState:
    """Node 1: Retriever agent fetches relevant context."""
    try:
        result = run_retriever(state["question"])
        return {
            **state,
            "retrieved_context": result["retrieved_context"],
            "error": ""
        }
    except Exception as e:
        return {
            **state,
            "retrieved_context": "",
            "error": f"Retrieval failed: {str(e)}"
        }


def validate_node(state: AgentState) -> AgentState:
    """Node 2: Validator agent generates and scores the final answer."""
    if state.get("error"):
        return {
            **state,
            "final_answer": "Sorry, I couldn't retrieve relevant information for your question.",
            "validation_score": 0.0
        }

    try:
        result = run_validator(
            question=state["question"],
            context=state["retrieved_context"]
        )
        response_text = result["response"]

        answer = response_text
        score = 0.75

        if "ANSWER:" in response_text:
            answer = response_text.split("ANSWER:")[-1].split("SCORES:")[0].strip()

        import re
        score_match = re.search(r'"overall":\s*([\d.]+)', response_text)
        if score_match:
            score = float(score_match.group(1))

        return {
            **state,
            "final_answer": answer,
            "validation_score": score,
            "error": ""
        }
    except Exception as e:
        return {
            **state,
            "final_answer": "An error occurred while generating the answer.",
            "validation_score": 0.0,
            "error": str(e)
        }


def build_graph() -> StateGraph:
    """Build and compile the LangGraph state machine."""
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("validate", validate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "validate")
    graph.add_edge("validate", END)

    return graph.compile()


def run_pipeline(question: str) -> dict:
    """
    Run the full multi-agent pipeline for a seller question.

    Args:
        question: The seller's question

    Returns:
        dict with answer, validation_score, and any errors
    """
    graph = build_graph()

    initial_state: AgentState = {
        "question": question,
        "retrieved_context": "",
        "final_answer": "",
        "sources": [],
        "validation_score": 0.0,
        "error": ""
    }

    result = graph.invoke(initial_state)

    return {
        "question": result["question"],
        "answer": result["final_answer"],
        "validation_score": result["validation_score"],
        "error": result.get("error", "")
    }


if __name__ == "__main__":
    question = "How do I create a product listing on Amazon?"
    print(f"Question: {question}\n")
    result = run_pipeline(question)
    print(f"Answer: {result['answer']}")
    print(f"Validation Score: {result['validation_score']}")