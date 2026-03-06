from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn
import pandas as pd
import os

# Configuration
RAG_ENGINE_URL = "http://10.68.198.184:8000"
DATA_PATH = "data/Clean_reviews.csv"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize a shared HTTP client
    app.state.http_client = httpx.AsyncClient()
    
    # Load unique product list from dataset
    print(f"📂 Loading unique products from {DATA_PATH}...")
    if os.path.exists(DATA_PATH):
        try:
            # Matches the current RAG indexing limit of 20,000
            df = pd.read_csv(DATA_PATH, nrows=20000)
            
            # Count reviews per product
            counts = df['product_id'].value_counts()
            
            # Create a list of formatted strings: "ID (X reviews)"
            # We sort by the ID for consistency
            product_list = []
            for pid in sorted(counts.index):
                count = counts[pid]
                product_list.append(f"{pid} ({count} reviews)")
            
            app.state.products = product_list
            print(f"✅ Loaded {len(app.state.products)} unique products with counts.")
        except Exception as e:
            print(f"❌ Error loading products: {e}")
            app.state.products = []
    else:
        print(f"⚠️ Warning: Dataset not found at {DATA_PATH}. Product list will be empty.")
        app.state.products = []
        
    yield
    await app.state.http_client.aclose()

app = FastAPI(lifespan=lifespan, title="Gateway API")

# --- NETWORK ACCESS: Allow any device on the local network ---
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
        print(f"📡 Forwarding request for {item_id} to RAG Engine at {RAG_ENGINE_URL}...")
        response = await app.state.http_client.get(target_url, timeout=45.0)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"RAG Engine error: {response.text}"
            )
            
    except httpx.RequestError as exc:
        print(f"❌ Connection to RAG Engine failed: {exc}")
        raise HTTPException(
            status_code=503, 
            detail="Could not reach RAG Engine. Check if the remote machine is online."
        )

if __name__ == "__main__":
    # Gateway API runs on 8001
    print("🚀 Gateway API listening on http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
