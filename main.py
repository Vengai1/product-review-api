from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn
import pandas as pd
import os
import logging

# --- ROBUST LOGGING CONFIGURATION ---
os.makedirs("logs", exist_ok=True)
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File Handler for Gateway
file_handler = logging.FileHandler("logs/gateway.log")
file_handler.setFormatter(log_format)

# Console Handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_format)

# Root Logger Setup
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)

logger = logging.getLogger(__name__)
logger.info("Gateway Logging initialized - writing to logs/gateway.log")

# Configuration
RAG_ENGINE_URL = "http://172.16.44.58:8000" #Replace the given IP with the IP address of the remote machine where RAG Engine is running
DATA_PATH = "data/Clean_reviews.csv"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize a shared HTTP client
    app.state.http_client = httpx.AsyncClient()
    
    # Load unique product list from dataset
    logger.info(f" Scanning dataset for unique products: {DATA_PATH}")
    if os.path.exists(DATA_PATH):
        try:
            # Matches the current RAG indexing limit
            df = pd.read_csv(DATA_PATH, nrows=20000)
            
            # Count reviews per product
            counts = df['product_id'].value_counts()
            
            # Create a list of formatted strings: "ID (X reviews)"
            product_list = []
            for pid in sorted(counts.index):
                count = counts[pid]
                product_list.append(f"{pid} ({count} reviews)")
            
            app.state.products = product_list
            logger.info(f" Loaded {len(app.state.products)} unique products with counts.")
        except Exception as e:
            logger.error(f" Failed to parse dataset: {e}")
            app.state.products = []
    else:
        logger.warning(f" Dataset not found at {DATA_PATH}. Product list will be empty.")
        app.state.products = []
        
    yield
    await app.state.http_client.aclose()

app = FastAPI(lifespan=lifespan, title="Gateway API")

# --- NETWORK ACCESS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/products")
async def get_products():
    """Returns a list of all unique product IDs in the dataset."""
    return {"products": app.state.products}

@app.get("/items/{item_id}")
async def read_item(item_id: str):
    """
    Proxies requests to the RAG Engine on the remote machine.
    """
    try:
        target_url = f"{RAG_ENGINE_URL}/items/{item_id}"
        
        logger.info(f" Proxying request: {item_id} -> {RAG_ENGINE_URL}")
        response = await app.state.http_client.get(target_url, timeout=45.0)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f" RAG Engine returned error {response.status_code}: {response.text}")
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"RAG Engine error: {response.text}"
            )
            
    except httpx.RequestError as exc:
        logger.error(f" Connection to RAG Engine failed: {exc}")
        raise HTTPException(
            status_code=503, 
            detail=f"Could not reach RAG Engine at {RAG_ENGINE_URL}."
        )

# Run this on port 8001
if __name__ == "__main__":
    logger.info(" Starting Gateway API on http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
