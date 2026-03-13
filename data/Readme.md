# data/

This directory is populated by running the ingestion pipeline:

```bash
python rag/ingest.py
```

## What gets created

### `seller_central/`
Raw FAQ documents scraped from Amazon Seller Central help pages,
covering topics including:
- Product listing creation
- FBA fees and fulfillment
- Account health metrics
- Returns and refunds
- Brand Registry
- Inventory management
- Payments and disbursements
- Shipping options
- Listing suppression and policy violations
- SEO and product visibility

### `chroma_db/`
ChromaDB vector store containing ~200 embedded chunks
from the seller_central documents. Used by the Retriever agent
for semantic similarity search at query time.

Both directories are excluded from version control via `.gitignore`.
Run `make ingest` to regenerate them locally.