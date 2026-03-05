import pandas as pd
import string
import os

def standardize(messy):
    print(f"Loading raw data from {messy}...")
    df = pd.read_csv(messy)
    column_mapping ={
        "ProductId": "product_id",
        "Text":"review_text",
        "Score":"rating(out of 5)",
        "Time":"review_date"
    }
    #"Standardizing column names"
    df.rename(columns=column_mapping, inplace=True)
    mandatory_columns = ['product_id', 'review_text', 'rating(out of 5)', 'review_date']
    df=df[mandatory_columns]
    df=df.dropna(subset=mandatory_columns)
    df['review_date'] = pd.to_datetime(df['review_date'], unit='s').dt.strftime('%Y-%m-%d')
    #"Data standardized"
    return df

def clean_text(text):
    text=str(text)
    # Remove HTML line breaks
    text = text.replace('<br>', ' ').replace('<br/>', ' ').replace('<br />', ' ')
    # Normalize whitespace but KEEP punctuation
    text = ' '.join(text.split())
    return text

def preprocess_text(df):
    df['review_text'] = df['review_text'].apply(clean_text)
    df=df[df['review_text']!=""]
    return df

if __name__ == "__main__":
    # Define paths relative to the project root
    input_path = os.path.join("data", "Reviews.csv")
    output_path = os.path.join("data", "Clean_reviews.csv")
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    Clean_dataset = standardize(input_path)
    
    Final_Dataset = preprocess_text(Clean_dataset)
    Final_Dataset.to_csv(output_path, index=False)
    print(f"Cleaned and preprocessed data saved to {output_path}")
