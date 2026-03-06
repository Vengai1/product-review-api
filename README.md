# Product Review Insights API (RAG Engine)

A high-performance RAG (Retrieval-Augmented Generation) pipeline that transforms thousands of raw Amazon reviews into structured, actionable insights. 

This project uses K-Means clustering to identify top themes, analyzes sentiment per aspect, and extracts supporting evidence using a hybrid local-vector/cloud-LLM architecture.

## Key Features

*   Intelligent Aspect Extraction: Uses K-Means clustering to summarize large review sets (up to 10k+) into representative themes before LLM processing.
*   Hybrid Sentiment Engine: Categorizes aspects as Pro, Con, or Mixed based on real customer star-ratings metadata.
*   Evidence Selection: Automatically retrieves multiple relevant supporting sentences for every insight, ensuring zero hallucinations.
*   Dynamic Scaling: Automatically adjusts the depth of analysis based on the number of available reviews.
*   Professional Dashboard: Interactive Streamlit interface with a split-screen layout for summaries and categorized insights.

## Tech Stack

- NLP: spaCy (Sentence Segmentation)
- Vector DB: ChromaDB (Persistent Storage)
- Embeddings: BAAI/bge-small-en-v1.5
- Intelligence: OpenRouter (GPT-4o-mini)
- Backend: FastAPI & Uvicorn
- Frontend: Streamlit
- Analytics: scikit-learn (K-Means Clustering)

## Repository Structure

```text
product-review-api/
├── main.py                     # Client-side API (Proxies requests to RAG Engine)
├── RAG/
│   ├── main.py                 # Core RAG Server (Handles analysis logic)
│   ├── insight_engine.py       # Insight Generation Logic (LLM + Evidence)
│   ├── cluster_aspect_extractor.py # K-Means Theme Discovery
│   ├── vector_store.py         # ChromaDB Management & Sub-batching
│   ├── data_manager.py         # spaCy Preprocessing & Sentence Splitting
│   ├── index_data.py           # Script to populate the Vector Database
│   └── search_reviews.py       # CLI tool for manual review searching
├── front_end/
│   └── dashboard.py            # Streamlit Dashboard UI
├── data/
│   ├── Reviews.csv             # Raw Dataset (Input)
│   └── Clean_reviews.csv       # Preprocessed Data (Punctuation-preserved)
├── k.py                        # Data Standardization & Cleaning Script
└── .env                        # Configuration (OPENROUTER_API_KEY)
```

## Installation & Setup

1. Setup Environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. Configure API Key:
   Create a .env file in the root:
   ```text
   OPENROUTER_API_KEY=your_openrouter_key_here
   ```

3. Index the Data (Required before first run):
   ```bash
   python k.py
   python RAG/index_data.py
   ```

## Running the Application

This system consists of two backend servers and one frontend dashboard.

### 1. Start the RAG Engine (Core Server)
```bash
python -m RAG.main
```
*Default: http://localhost:8000*

### 2. Start the Client API (Gateway Server)
```bash
python main.py
```
*Default: http://localhost:8001*

### 3. Start the Dashboard (Frontend)
```bash
streamlit run front_end/dashboard.py
```

## API Endpoints

### 1. Client Gateway API (main.py)
This is the primary entry point for the frontend. It proxies requests to the RAG engine.
* **GET `/items/{item_id}`**: Retrieves the processed insights for a product.
    * **Response**: A structured JSON containing `summary`, `top_aspects`, `status`, and `confidence`.

### 2. RAG Engine API (RAG/main.py)
The core analysis server.
* **GET `/`**: Health check.
* **GET `/items/{item_id}`**: Performs the actual RAG analysis (Clustering -> LLM -> Vector Search).
* **Response**: Raw analysis data used by the Gateway.

## License
MIT License
