import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from typing import List, Dict
from .vector_store import VectorStore

class ClusterAspectExtractor:
    def __init__(self):
        self.vs = VectorStore()

    def get_representative_sentences(self, product_id: str) -> List[str]:
        """
        Dynamically clusters sentences:
        - Max 25 sentences for large review sets
        - Min 5 sentences (if available)
        - If < 5, returns all sentences
        """
        results = self.vs.get_all_for_product(product_id)
        sentences = results.get('documents', [])
        
        if not sentences:
            return []

        num_sentences = len(sentences)

        # 1. Determine number of clusters
        if num_sentences <= 5:
            return sentences # Return all if very few
        
        # Scale between 5 and 25 clusters
        # Logic: 1 cluster per ~20 sentences, capped at 25, floor at 5
        n_clusters = min(max(5, num_sentences // 20), 25)
        
        print(f" Product {product_id}: Found {num_sentences} sentences. Extracting {n_clusters} clusters.")

        # 2. Generate Embeddings
        embeddings = self.vs.model.encode(sentences)
        
        # 3. K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        
        # 4. Find closest to centroids
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
        
        representative_sentences = []
        for idx in closest:
            sent = sentences[idx].strip()
            representative_sentences.append(sent)
        
        return list(dict.fromkeys(representative_sentences))

if __name__ == "__main__":
    extractor = ClusterAspectExtractor()
    test_ids = ["B003VXFK44", "B002BCD2OG", "B004NE2E9O"]
    
    for tid in test_ids:
        print(f"\n" + "="*80)
        print(f" TESTING PRODUCT: {tid}")
        reps = extractor.get_representative_sentences(tid)
        print(f" Generated {len(reps)} representative sentences:")
        for i, sent in enumerate(reps):
            print(f"  [{i+1}] \"{sent}\"")
