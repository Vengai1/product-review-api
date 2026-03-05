import pandas as pd
import spacy
from typing import List, Tuple

# Load spaCy model for sentence splitting
# We disable components we don't need for speed
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer", "attribute_ruler"])
    # Add sentencizer if it's not already there (though en_core_web_sm usually uses dependency parser for sentences)
    if "senter" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
except Exception as e:
    print(f"Warning: Could not load spaCy model: {e}")
    nlp = None

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
    Splits each review into sentences using spaCy.
    Returns a list of tuples: (sentence, product_id, rating)
    """
    chunked_data = []
    
    for _, row in df.iterrows():
        text = str(row['review_text'])
        product_id = row['product_id']
        rating = row['rating']
        
        if nlp:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        else:
            # Fallback if spaCy is missing
            sentences = [text]
        
        for sentence in sentences:
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
