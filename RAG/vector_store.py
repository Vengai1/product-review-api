import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import uuid
import os

class VectorStore:
    def __init__(self, collection_name: str = "product_reviews"):
        """Initializes ChromaDB client and embedding model."""
        # Calculate the project root (assuming this file is in project_root/RAG/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        chroma_path = os.path.join(project_root, "chroma_db")
        
        # Persistent storage locally in a folder named 'chroma_db'
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create or get existing collection
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_to_index(self, chunked_data: List[Tuple[str, str, int]]):
        """
        Embeds each sentence and stores it in the collection with metadata.
        chunked_data is a list of (sentence, product_id, rating) tuples.
        """
        sentences = [item[0] for item in chunked_data]
        ids = [str(uuid.uuid4()) for _ in range(len(sentences))]
        
        # Generate embeddings
        embeddings = self.model.encode(sentences).tolist()
        
        # Prepare metadata
        metadatas = [
            {
                "text": item[0],
                "product_id": item[1],
                "rating": item[2]
            }
            for item in chunked_data
        ]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=sentences # Storing sentence also in document field for easy retrieval
        )

    def search(self, query: str, product_id: str = None, top_k: int = 5):
        """Search the collection with a natural language query."""
        query_embedding = self.model.encode([query]).tolist()
        
        where_filter = None
        if product_id:
            where_filter = {"product_id": product_id}
            
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter
        )
        return results

if __name__ == "__main__":
    # Test initialization
    vs = VectorStore()
    print("VectorStore initialized.")
