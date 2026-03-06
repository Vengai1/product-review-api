from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn

# Configuration
RAG_ENGINE_URL = "http://10.68.198.184:8000"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize a shared HTTP client for better performance
    app.state.http_client = httpx.AsyncClient()
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

@app.get("/items/{item_id}")
async def read_item(item_id: str):
    """
    Proxies requests to the RAG Engine on the remote machine.
    """
    try:
        # Calling the RAG Engine at the specified IP
        target_url = f"{RAG_ENGINE_URL}/items/{item_id}"
        
        print(f"Forwarding request for {item_id} to RAG Engine at {RAG_ENGINE_URL}...")
        response = await app.state.http_client.get(target_url, timeout=45.0)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"RAG Engine error: {response.text}"
            )
            
    except httpx.RequestError as exc:
        print(f"Connection to RAG Engine failed: {exc}")
        raise HTTPException(
            status_code=503, 
            detail=f"Could not reach RAG Engine at {RAG_ENGINE_URL}. Check if the remote machine is online."
        )

# Run this on port 8001 to keep port 8000 free for the RAG engine (if needed locally)
if __name__ == "__main__":
    print("🚀 Gateway API listening on http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
