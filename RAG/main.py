from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .insight_engine import InsightEngine
import uvicorn

# 1. Initialize FastAPI
app = FastAPI(
    title="Product Review",
    description="Analyzes Amazon reviews to extract aspects, sentiment, and evidence.",
    version="1.0.0"
)

# 2. Add CORS Middleware (Crucial for Team Access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows your teammate to connect from any URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Initialize the Engine
# This will load the VectorStore and LLM Client once on startup
engine = InsightEngine()

@app.get("/")
def read_root():
    return {
        "status": "ONLINE",
        "message": "Product Review Insights API is running.",
        "usage": "GET /insights/{product_id}"
    }

@app.get("/items/{product_id}")
async def get_insights(product_id: str):
    """
    Fetches full RAG insights for a specific product.
    Returns: JSON with summary, aspects, sentiment, and evidence.
    """
    try:
        print(f"Received request for Product ID: {product_id}")
        
        # Run the full pipeline
        insights = engine.get_full_insights(product_id)
        
        # Handle Task 5 "Not enough data" requirement
        if "error" in insights:
            if "Not enough data" in insights["error"]:
                return {
                    "product_id": product_id,
                    "status": "INSUFFICIENT_DATA",
                    "message": insights["error"],
                    "top_aspects": [],
                    "summary": "Not enough reviews available for analysis.",
                    "confidence": 0.0
                }
            # Handle other errors (API failures, etc.)
            raise HTTPException(status_code=500, detail=insights["error"])
        
        # Successful result
        insights["status"] = "SUCCESS"
        return insights

    except Exception as e:
        print(f"Error processing {product_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 4. To run: python -m RAG.main
if __name__ == "__main__":
    print("Starting Server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
