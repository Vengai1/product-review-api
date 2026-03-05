from fastapi import FastAPI, HTTPException
import json
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/welcome")
async def welcome():
    return {"message": "Welcome to our API"}

@app.get("/items/{item_id}")
async def read_item(item_id: str):
    try:
        with open("data.json", "r") as f:
            data = json.load(f)
            for item in data:
                if item["product_id"] == item_id:
                    return item
    except FileNotFoundError:
        print("data.json file not found. Please make sure it exists in the same directory as main.py.")
    raise HTTPException(status_code=404, detail="Item not found")