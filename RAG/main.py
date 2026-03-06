from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .insight_engine import InsightEngine
import uvicorn
import logging

import os

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Define format
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File Handler
file_handler = logging.FileHandler("logs/rag_engine.log")
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
logger.info("Logging initialized - writing to logs/rag_engine.log")

# 1. Initialize FastAPI
app = FastAPI(
    title="RAG Engine API",
    description="Analyzes Amazon reviews using RAG pipeline.",
    version="1.0.0"
)

# 2. Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Initialize the Engine
try:
    engine = InsightEngine()
    logger.info("RAG Engine initialized.")
except Exception as e:
    logger.critical(f"Failed to start RAG Engine: {e}")
    raise

@app.get("/")
def read_root():
    return {"status": "ONLINE", "server": "RAG_CORE"}

@app.get("/items/{product_id}")
async def get_insights(product_id: str):
    """
    Core RAG analysis endpoint.
    """
    try:
        logger.info(f" Analysis request for: {product_id}")
        insights = engine.get_full_insights(product_id)
        
        if "error" in insights:
            logger.warning(f" Issue processing {product_id}: {insights['error']}")
        
        return insights
    except Exception as e:
        logger.error(f" Critical error in RAG Engine for {product_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal RAG processing error.")

if __name__ == "__main__":
    logger.info(" Starting RAG Core Server on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
