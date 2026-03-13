"""
evaluation/evaluate.py
Runs RAGAS evaluation over the 50-question benchmark.
Measures: answer_relevancy, faithfulness, context_precision, context_recall

Run: python evaluation/evaluate.py
Results saved to: evaluation/results.json
"""

import json
import time
import pandas as pd
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)
from langchain_ollama import OllamaEmbeddings, ChatOllama
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from rag.retriever import retrieve
from orchestrator import run_pipeline

BENCHMARK_PATH = Path("evaluation/benchmark.csv")
RESULTS_PATH = Path("evaluation/results.json")


def run_evaluation(sample_size: int = 20):
    """
    Run RAGAS evaluation on benchmark questions.

    Args:
        sample_size: Number of questions to evaluate (default 20 to save time)
    """
    print(f"Loading benchmark from {BENCHMARK_PATH}...")
    df = pd.read_csv(BENCHMARK_PATH)
    df = df.head(sample_size)
    print(f"Evaluating {len(df)} questions...")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, row in df.iterrows():
        question = row["question"]
        gt = row["ground_truth"]

        print(f"  [{i+1}/{len(df)}] Running pipeline for: {question[:60]}...")
        start = time.time()

        result = run_pipeline(question)
        docs = retrieve(question, k=4)

        questions.append(question)
        answers.append(result["answer"])
        contexts.append([doc.page_content for doc in docs])
        ground_truths.append(gt)

        elapsed = round(time.time() - start, 2)
        print(f"    Done in {elapsed}s | Score: {result['validation_score']}")

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    print("\nRunning RAGAS evaluation...")
    llm = LangchainLLMWrapper(ChatOllama(model="mistral"))
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="mistral"))

    result = evaluate(
        dataset=dataset,
        metrics=[
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
    )

    scores = {
        "answer_relevancy": round(float(result["answer_relevancy"]), 4),
        "faithfulness": round(float(result["faithfulness"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
        "context_recall": round(float(result["context_recall"]), 4),
        "sample_size": sample_size,
    }

    RESULTS_PATH.write_text(json.dumps(scores, indent=2))

    print("\n" + "="*50)
    print("RAGAS Evaluation Results")
    print("="*50)
    for metric, score in scores.items():
        if metric != "sample_size":
            print(f"  {metric:<25} {score:.4f}")
    print(f"\n  Questions evaluated: {sample_size}")
    print(f"  Results saved to: {RESULTS_PATH}")
    print("="*50)

    return scores


if __name__ == "__main__":
    import sys
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_evaluation(sample_size=sample)