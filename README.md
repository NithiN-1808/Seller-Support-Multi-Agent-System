# Seller Support Multi-Agent System

A multi-agent system for answering Amazon Seller Central queries using **Strands Agents**, **LangGraph**, **MCP**, **ChromaDB**, and **Ollama** for fully local LLM inference.

## Architecture

```
User Query
    │
    ▼
FastAPI REST API
    │
    ▼
LangGraph Orchestrator (state machine)
    ├──► Retriever Agent (Strands) ──► MCP Tool Server ──► ChromaDB
    └──► Validator Agent (Strands) ──► RAGAS Evaluation
    │
    ▼
Final Answer
```

## Tech Stack

| Component | Tool |
|---|---|
| Agent framework | Strands Agents |
| Orchestration | LangGraph |
| Tool protocol | MCP (Model Context Protocol) |
| Vector store | ChromaDB |
| LLM backend | Ollama (Mistral 7B Q4) |
| RAG pipeline | LangChain |
| Evaluation | RAGAS |
| API | FastAPI |
| Containerization | Docker |

## Evaluation Results

| Metric | Score |
|---|---|
| Answer Relevance |  run `make evaluate` to generate  |
| Faithfulness | run `make evaluate` to generate |
| Context Precision | run `make evaluate` to generate |
| Task Completion Rate |  run `make evaluate` to generate |

> Fill in after running evaluation on the 50-question benchmark.

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed
- Docker (optional, for containerized run)

### 1. Pull the model
```bash
ollama pull mistral
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Ingest Seller Central docs
```bash
python rag/ingest.py
```

### 4. Start the MCP server
```bash
python mcp_server/server.py
```

### 5. Start the API
```bash
uvicorn main:app --reload
```

### 6. Run evaluation
```bash
python evaluation/evaluate.py
```

### Docker
```bash
docker build -t seller-support-agent .
docker run -p 8000:8000 seller-support-agent
```

## API Usage

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I create a new product listing?"}'
```

Response:
```json
{
  "answer": "To create a new product listing...",
  "sources": ["listing-creation.txt"],
  "validation_score": 0.91,
  "retrieval_faithfulness": 0.88
}
```

## Project Structure

```
seller-support-agent/
├── main.py                  # FastAPI entrypoint
├── agents/
│   ├── retriever.py         # Strands Retriever agent
│   └── validator.py         # Strands Validator agent
├── mcp_server/
│   └── server.py            # MCP tool server
├── rag/
│   ├── ingest.py            # Document ingestion + ChromaDB
│   └── retriever.py         # RAG retrieval logic
├── evaluation/
│   ├── evaluate.py          # RAGAS evaluation runner
│   └── benchmark.csv        # 50-question QA benchmark
├── data/
│   └── seller_central/      # Scraped FAQ documents
├── tests/
│   └── test_agents.py
├── requirements.txt
├── Dockerfile
└── Makefile

```
