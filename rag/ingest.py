"""
rag/ingest.py
Scrapes Amazon Seller Central FAQ pages and ingests them into ChromaDB.
Run once before starting the system: python rag/ingest.py
"""

import os
import re
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document

DATA_DIR = Path("data/seller_central")
CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "seller_central_faq"

SELLER_CENTRAL_URLS = [
    "https://sellercentral.amazon.com/help/hub/reference/G200421970",  # List products
    "https://sellercentral.amazon.com/help/hub/reference/G1791",       # Product detail pages
    "https://sellercentral.amazon.com/help/hub/reference/G200386250",  # Fees
    "https://sellercentral.amazon.com/help/hub/reference/G69117",      # Shipping
    "https://sellercentral.amazon.com/help/hub/reference/G200545940",  # Returns
    "https://sellercentral.amazon.com/help/hub/reference/G200142030",  # Account health
    "https://sellercentral.amazon.com/help/hub/reference/G200633530",  # Brand registry
    "https://sellercentral.amazon.com/help/hub/reference/G200417110",  # Inventory management
    "https://sellercentral.amazon.com/help/hub/reference/G201750520",  # FBA overview
    "https://sellercentral.amazon.com/help/hub/reference/G200270400",  # Payments
]

# Fallback: local FAQ documents if scraping fails
FALLBACK_FAQS = [
    {
        "title": "How to create a product listing",
        "content": """To create a new product listing on Amazon Seller Central:
1. Go to Inventory > Add a Product
2. Search for the product by UPC, EAN, ISBN, or ASIN if it already exists on Amazon
3. If the product is new, click 'Create a new product listing'
4. Select the appropriate product category
5. Fill in required fields: product title, brand, manufacturer, description
6. Set your price, quantity, and fulfillment method (FBM or FBA)
7. Add product images (minimum 1, recommended 7+)
8. Submit the listing for review
Note: New listings may take 15 minutes to several hours to appear on Amazon."""
    },
    {
        "title": "Amazon FBA fees explained",
        "content": """Amazon FBA (Fulfillment by Amazon) fees include:
- Fulfillment fees: Based on product size and weight. Small standard items start at $3.22 per unit.
- Monthly storage fees: $0.87 per cubic foot (Jan-Sep), $2.40 per cubic foot (Oct-Dec)
- Long-term storage fees: Applied to items stored over 365 days
- Referral fees: Category-based percentage of sale price (typically 8-15%)
- Selling plan: Individual ($0.99/item) or Professional ($39.99/month)
Use the FBA Revenue Calculator to estimate fees before sending inventory."""
    },
    {
        "title": "Account health and performance metrics",
        "content": """Amazon monitors seller performance through Account Health metrics:
- Order Defect Rate (ODR): Must stay below 1%. Includes negative feedback, A-to-z claims, chargebacks.
- Cancellation Rate: Must stay below 2.5% for seller-fulfilled orders.
- Late Shipment Rate: Must stay below 4%.
- Valid Tracking Rate: Must be above 95%.
- Customer Service Dissatisfaction Rate: Must stay below 25%.
Check your Account Health dashboard regularly. Violations can result in listing suppression or account suspension."""
    },
    {
        "title": "How to handle customer returns",
        "content": """Amazon return policy for sellers:
- FBA sellers: Amazon handles returns automatically. Items are returned to fulfillment centers.
- FBM sellers: Must match or exceed Amazon's return policy (30 days from delivery).
To process a return:
1. Go to Orders > Manage Returns
2. Review the return request and reason
3. Authorize or close the return
4. Issue refund once item is received
For FBA: Returns are automatically refunded. You may receive reimbursement if items are lost/damaged.
Tip: High return rates can negatively impact your account health score."""
    },
    {
        "title": "Brand Registry requirements and benefits",
        "content": """Amazon Brand Registry protects your brand and unlocks additional features:
Requirements:
- Active registered trademark (word mark or design mark)
- Trademark must be registered in countries where you want protection
- Must match the brand name on your products and packaging
Benefits:
- Proactive brand protection using ML to detect violations
- Access to A+ Content (enhanced product descriptions)
- Brand Analytics dashboard
- Sponsored Brands advertising
- Amazon Stores (free multi-page brand storefront)
- Virtual product bundles
To enroll: Visit brandservices.amazon.com and submit trademark details."""
    },
    {
        "title": "Inventory management best practices",
        "content": """Managing FBA inventory effectively:
- Monitor inventory levels in Inventory > Manage FBA Inventory
- Set reorder alerts to avoid stockouts
- Use the Inventory Performance Index (IPI) to track efficiency (target: 450+)
- Remove slow-moving inventory to avoid long-term storage fees
- Create removal orders for items approaching 365-day storage threshold
- Use the Restock Inventory tool for demand forecasting
- Send inventory in correct packaging: poly bags, bubble wrap, or boxes per Amazon guidelines
- Label requirements: FNSKU labels required unless using stickerless commingled inventory"""
    },
    {
        "title": "How to get paid as an Amazon seller",
        "content": """Amazon payment disbursement process:
- Payments are disbursed every 14 days to your bank account
- Amazon holds funds for 3-5 business days after delivery confirmation
- Disbursement schedule: Every 14 days based on your registration date
- Minimum disbursement: $1 (below this, balance carries to next cycle)
- You can request early disbursement once per settlement period
To set up payments:
1. Go to Settings > Account Info > Deposit Methods
2. Add your bank account details
3. Verify your bank account
Note: Amazon deducts fees, refunds, and other charges before disbursement."""
    },
    {
        "title": "Shipping and fulfillment options",
        "content": """Amazon offers two main fulfillment methods:
FBA (Fulfillment by Amazon):
- Send inventory to Amazon warehouses
- Amazon picks, packs, ships, and handles customer service
- Products eligible for Prime shipping
- Higher fees but less operational burden

FBM (Fulfilled by Merchant):
- You store and ship orders yourself
- Lower fees but requires logistics infrastructure
- Must meet Amazon shipping speed requirements
- Seller Fulfilled Prime available for qualified sellers

Seller Fulfilled Prime requirements:
- Same-day handling for orders placed by cutoff time
- Ship with Amazon-approved carriers
- Maintain on-time delivery rate above 93.5%"""
    },
    {
        "title": "How to improve product visibility and ranking",
        "content": """Amazon SEO and visibility strategies:
Title optimization:
- Include primary keyword near the beginning
- Add brand, key features, size, quantity
- Keep under 200 characters

Backend keywords:
- Add search terms in Seller Central backend
- No punctuation, no repeated words
- Include synonyms and alternate spellings

Bullet points:
- Lead with key benefits
- Include secondary keywords naturally
- Focus on customer pain points solved

A+ Content (Brand Registry required):
- Enhanced images and text modules
- Can increase conversion by 3-10%

Advertising:
- Sponsored Products for keyword targeting
- Automatic campaigns to discover new keywords
- Optimize bids based on ACoS (Advertising Cost of Sale) targets"""
    },
    {
        "title": "Dealing with listing suppression and policy violations",
        "content": """Common reasons for listing suppression:
- Missing required attributes (brand, manufacturer, etc.)
- Images not meeting Amazon guidelines (white background, 1000px+ minimum)
- Price violation (price too high compared to other channels)
- ASIN mismatch or incorrect category
- Restricted product sold without approval

How to fix suppressed listings:
1. Go to Inventory > Fix Stranded Inventory or Manage All Inventory
2. Filter by 'Suppressed' status
3. Click the listing issue to see the specific reason
4. Edit the listing to fix the issue
5. Resubmit for review

For policy violations:
- Review the notification in Performance > Account Health
- Submit a Plan of Action (POA) if requested
- POA should include: root cause, corrective actions, preventive measures"""
    },
]


def scrape_url(url: str) -> str:
    """Attempt to scrape a Seller Central URL. Returns empty string if blocked."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            main = soup.find("main") or soup.find("article") or soup.body
            if main:
                text = main.get_text(separator="\n", strip=True)
                text = re.sub(r"\n{3,}", "\n\n", text)
                return text[:3000]
    except Exception:
        pass
    return ""


def load_documents() -> list[Document]:
    """Load documents from scraping or fallback FAQs."""
    docs = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Attempting to load from Seller Central URLs...")
    scraped_count = 0
    for url in SELLER_CENTRAL_URLS:
        content = scrape_url(url)
        if content and len(content) > 200:
            slug = url.split("/")[-1]
            fpath = DATA_DIR / f"{slug}.txt"
            fpath.write_text(content)
            docs.append(Document(
                page_content=content,
                metadata={"source": url, "filename": fpath.name}
            ))
            scraped_count += 1
            print(f"  Scraped: {url}")
            time.sleep(1)

    if scraped_count < 3:
        print(f"Only scraped {scraped_count} pages. Using local FAQ documents...")
        for faq in FALLBACK_FAQS:
            fpath = DATA_DIR / f"{faq['title'].lower().replace(' ', '_')}.txt"
            fpath.write_text(f"{faq['title']}\n\n{faq['content']}")
            docs.append(Document(
                page_content=f"{faq['title']}\n\n{faq['content']}",
                metadata={"source": "local_faq", "filename": fpath.name, "title": faq["title"]}
            ))

    print(f"Loaded {len(docs)} documents total.")
    return docs


def ingest():
    """Main ingestion pipeline."""
    print("Starting ingestion pipeline...")

    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    print("Loading embedding model via Ollama...")
    embeddings = OllamaEmbeddings(model="mistral")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR)
    )

    print(f"Ingested {len(chunks)} chunks into ChromaDB at {CHROMA_DIR}")
    print("Ingestion complete.")
    return vectorstore


if __name__ == "__main__":
    ingest()