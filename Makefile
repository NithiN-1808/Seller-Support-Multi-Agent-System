.PHONY: install ingest serve evaluate test docker-build docker-run clean

install:
	pip install -r requirements.txt

ingest:
	python rag/ingest.py

serve:
	uvicorn main:app --reload --port 8000

evaluate:
	python evaluation/evaluate.py

test:
	pytest tests/ -v

docker-build:
	docker build -t seller-support-agent .

docker-run:
	docker run -p 8000:8000 seller-support-agent

clean:
	rm -rf data/chroma_db data/seller_central/__pycache__ __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete