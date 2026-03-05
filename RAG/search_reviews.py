from vector_store import VectorStore
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python search_reviews.py '<search_query>' [product_id]")
        return

    query = sys.argv[1]
    product_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    vs = VectorStore()
    
    print(f"Searching for: '{query}'" + (f" in product {product_id}" if product_id else ""))
    results = vs.search(query, product_id=product_id, top_k=10)
    
    print("\nTop 10 Results:")
    print("-" * 50)
    
    # ChromaDB results are returned in a nested list structure
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    for i in range(len(documents)):
        text = documents[i]
        meta = metadatas[i]
        dist = distances[i]
        
        print(f"{i+1}. [Score: {1-dist:.4f}] [Product: {meta['product_id']}] [Rating: {meta['rating']}/5]")
        print(f"   {text}")
        print("-" * 50)

if __name__ == "__main__":
    main()
