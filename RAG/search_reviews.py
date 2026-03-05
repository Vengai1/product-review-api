from vector_store import VectorStore
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python RAG/search_reviews.py '<search_query>' [product_id]")
        return

    query = sys.argv[1]
    product_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Initialize VectorStore
    vs = VectorStore()
    
    print(f"\n🔍 Searching for: '{query}'" + (f" (Product: {product_id})" if product_id else ""))
    print("=" * 60)
    
    results = vs.search(query, product_id=product_id, top_k=10)
    
    # Display results
    if not results or not results['documents'][0]:
        print("No matches found.")
        return

    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    seen_texts = set()
    displayed_count = 0
    
    for i in range(len(documents)):
        text = documents[i]
        meta = metadatas[i]
        dist = distances[i]
        
        # Simple deduplication (many reviews have the exact same sentence)
        if text.strip().lower() in seen_texts:
            continue
        seen_texts.add(text.strip().lower())
        
        # Convert distance to a 0-100% similarity score
        similarity = max(0, 1 - dist) * 100
        
        print(f"[{displayed_count + 1}] Similarity: {similarity:.1f}%")
        print(f"    Product ID: {meta.get('product_id', 'N/A')}")
        print(f"    Rating:     {'★' * int(meta.get('rating', 0))}{'☆' * (5 - int(meta.get('rating', 0)))}")
        print(f"    Sentence:   \"{text}\"")
        print("-" * 60)
        
        displayed_count += 1
        if displayed_count >= 10:
            break

if __name__ == "__main__":
    main()
