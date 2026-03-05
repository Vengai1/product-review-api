from data_manager import load_and_clean_data, chunk_text_into_sentences
from vector_store import VectorStore
import time

def main():
    print("Starting indexing process...")
    start_time = time.time()
    
    # 1. Load data
    df = load_and_clean_data('data/Reviews.csv')
    print(f"Total reviews in CSV: {len(df)}")
    
    # Limit for demonstration to first 100 rows
    limit = 100
    df_sample = df.head(limit)
    print(f"Processing first {limit} reviews...")
    
    # 2. Chunk text
    chunks = chunk_text_into_sentences(df_sample)
    print(f"Total sentences to index: {len(chunks)}")
    
    # 3. Initialize and add to Vector Store
    vs = VectorStore()
    vs.add_to_index(chunks)
    
    end_time = time.time()
    print(f"Indexing completed in {end_time - start_time:.2f} seconds.")
    
    # 4. Quick verification search
    print("\nVerification Search for 'quality':")
    results = vs.search("quality", top_k=3)
    for i, doc in enumerate(results['documents'][0]):
        print(f"Result {i+1}: {doc}")

if __name__ == "__main__":
    main()
