import pandas as pd
import re
from typing import List, Tuple

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Loads Amazon Reviews CSV and drops rows with missing Text or ProductId."""
    df = pd.read_csv(file_path)
    # Map the CSV columns to the expected names
    df = df.rename(columns={'Text': 'review_text', 'ProductId': 'product_id', 'Score': 'rating'})
    # Drop rows where critical info is missing
    df = df.dropna(subset=['review_text', 'product_id'])
    return df

def chunk_text_into_sentences(df: pd.DataFrame) -> List[Tuple[str, str, int]]:
    """
    Splits each review into sentences using regex.
    Returns a list of tuples: (sentence, product_id, rating)
    """
    chunked_data = []
    # Simple regex to split on . ! or ? followed by a space or end of string
    sentence_pattern = re.compile(r'[^.!?]+[.!?]*')
    
    for _, row in df.iterrows():
        text = str(row['review_text'])
        product_id = row['product_id']
        rating = row['rating']
        
        # Split into sentences
        sentences = sentence_pattern.findall(text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 5: # Basic filter for noise
                chunked_data.append((sentence, product_id, int(rating)))
                
    return chunked_data

if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "Reviews.csv")
    
    df = load_and_clean_data(data_path)
    print(f"Loaded {len(df)} reviews.")
    chunks = chunk_text_into_sentences(df.head(5))
    print(f"Sample chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i+1}: {chunk}")
