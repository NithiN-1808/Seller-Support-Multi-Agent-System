"""
tests/test_agents.py
Basic tests for the multi-agent pipeline.
Run: pytest tests/
"""

import pytest
from unittest.mock import patch, MagicMock


def test_retriever_returns_context():
    """Retriever should return non-empty context for a valid question."""
    with patch("agents.retriever.retrieve_with_scores") as mock_retrieve:
        mock_doc = MagicMock()
        mock_doc.page_content = "FBA fulfillment fees start at $3.22 per unit."
        mock_doc.metadata = {"source": "local_faq"}
        mock_retrieve.return_value = [(mock_doc, 0.85)]

        from agents.retriever import run_retriever
        with patch("agents.retriever.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.return_value = "Retrieved: FBA fees context"
            MockAgent.return_value = mock_agent_instance

            result = run_retriever("What are FBA fees?")
            assert result["query"] == "What are FBA fees?"
            assert result["agent"] == "retriever"


def test_orchestrator_pipeline_structure():
    """Pipeline should return expected keys."""
    with patch("orchestrator.run_retriever") as mock_ret, \
         patch("orchestrator.run_validator") as mock_val:

        mock_ret.return_value = {
            "query": "test question",
            "retrieved_context": "Some context about FBA fees.",
            "agent": "retriever"
        }
        mock_val.return_value = {
            "question": "test question",
            "response": "ANSWER: FBA fees start at $3.22.\nSCORES: {\"overall\": 0.85}",
            "agent": "validator"
        }

        from orchestrator import run_pipeline
        result = run_pipeline("What are FBA fees?")

        assert "question" in result
        assert "answer" in result
        assert "validation_score" in result


def test_api_health(client=None):
    """Health endpoint should return ok."""
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_root():
    """Root endpoint should list agents."""
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "agents" in data
    assert "orchestrator" in data


def test_benchmark_csv_exists():
    """Benchmark CSV should have 50 questions."""
    import pandas as pd
    df = pd.read_csv("evaluation/benchmark.csv")
    assert len(df) == 50
    assert "question" in df.columns
    assert "ground_truth" in df.columns