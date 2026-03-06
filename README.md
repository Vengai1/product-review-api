# Product Review API
A powerful, AI-driven Product Review API built with Python. This project leverages **Retrieval-Augmented Generation (RAG)** to intelligently process, query, and serve product reviews and related data. The repository also includes a dedicated frontend interface for seamless interaction.

## Features

* **RAG-Powered Insights:** Uses Retrieval-Augmented Generation to provide accurate, context-aware answers and summaries about product reviews.
* **RESTful API:** Robust endpoints defined in `main.py` to handle data ingestion and client requests.
* **Frontend Included:** A user-friendly interface (`front_end/` directory) to interact directly with the API.
* **LLM Integration:** Tools to manage and list various models (`list_models.py`).
* **JSON Data Handling:** Structured product review data management (`data.json`).

## Repository Structure

```text
product-review-api/
│
├── RAG/                # Core Retrieval-Augmented Generation logic and embeddings
├── front_end/          # Client-side UI to interact with the API
├── main.py             # Main entry point and API application routing
├── requirements.txt    # Python dependencies
├── data.json           # Sample/Storage dataset for product reviews
├── list_models.py      # Utility for configuring and listing available AI models
├── gra.py              # Additional script / Gradio UI (if applicable)
├── i.py / k.py         # Utility/Helper scripts
├── .gitignore          # Git ignore configuration
└── LICENSE             # MIT License
```

## **Prerequisites**
* Python 3.9+, 
* pip, 
* virtual environment tool (optional)

### **Installation and Setup**

## 1. Clone The Repository:
git clone [https://github.com/Vengai1/product-review-api.git](https://github.com/Vengai1/product-review-api.git)
cd product-review-api

## 2. Create and activate a virtual environment:

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

## 3. Install Dependencies:
pip install -r requirements.txt

## 4. Environment Variables:
If your application requires API keys (e.g., OpenAI, HuggingFace, etc.) for the RAG pipeline, make sure to set them up in a .env file in the root directory.

### **Running the application**

## 1. Start the API server

Run the main python script to launch the API:
python main.py
(If you are using FastAPI/Uvicorn or Flask, the console will output the local host address, typically http://localhost:8000 or http://127.0.0.1:5000)

## 2. View Available Models:

To check which LLM models are currently configured or available:
python list_models.py

## 3. Start the Frontend:

Navigate to the frontend directory and follow its specific setup instructions, or if it is integrated into the Python backend (e.g., Gradio/Streamlit), it may launch automatically via main.py or gra.py.

### **API Endpoints:**
(Note: Update these based on your actual main.py routing)

* GET / - Health check and API status.

* POST /reviews - Submit a new product review.

* GET /reviews/{product_id} - Retrieve reviews for a specific product.

* POST /ask - RAG-based query endpoint. Send a question about a product and receive an AI-generated answer.