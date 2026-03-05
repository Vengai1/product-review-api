import pandas as pd
from data_manager import chunk_text_into_sentences
from vector_store import VectorStore
import time
import os

def main():
    print("🚀 Starting Fast Indexing Test (10,000 reviews)...")
    start_time = time.time()
    
    # Calculate paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "Clean_reviews.csv")
    
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found. Please run k.py first.")
        return
    
    # 1. Initialize Vector Store (Uses BAAI/bge-small-en-v1.5)
    vs = VectorStore()
    
    # 2. Process exactly 10,000 reviews
    limit = 10000
    batch_size = 2500 # 4 batches total
    total_indexed = 0
    
    print(f"Reading first {limit} reviews from {data_path}...")
    
    # Use pandas chunking
    for chunk_df in pd.read_csv(data_path, chunksize=batch_size, nrows=limit):
        # Map k.py columns
        if 'rating(out of 5)' in chunk_df.columns:
            chunk_df = chunk_df.rename(columns={'rating(out of 5)': 'rating'})
            
        # 3. Chunk reviews into sentences
        chunks = chunk_text_into_sentences(chunk_df)
        
        if chunks:
            # 4. Add to Vector Store
            # Note: We'll leverage standard multi-threading in the encode step
            vs.add_to_index(chunks)
            
            total_indexed += len(chunk_df)
            elapsed = time.time() - start_time
            print(f"✅ Indexed {total_indexed}/{limit} reviews. Elapsed: {elapsed:.2f}s")

    end_time = time.time()
    print(f"\n✨ Fast test completed in {end_time - start_time:.2f} seconds.")
    print(f"Total reviews indexed: {total_indexed}")

if __name__ == "__main__":
    main()
