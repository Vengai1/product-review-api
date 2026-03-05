import pandas as pd
import string
def standardize(messy):
    print(f"Loading raw data from {messy}...")
    df = pd.read_csv(messy)
    column_mapping ={
        "ProductId": "product_id",
        "Text":"review_text",
        "Score":"rating(out of 5)",
        "Time":"review_date"
    }
    #"Standardizing column names..."
    df.rename(columns=column_mapping, inplace=True)
    mandatory_columns = ['product_id', 'review_text', 'rating(out of 5)', 'review_date']
    df=df[mandatory_columns]
    df=df.dropna(subset=mandatory_columns)
    df['review_date'] = pd.to_datetime(df['review_date'], unit='s').dt.strftime('%Y-%m-%d')
    #"Data standardized"
    return df
Clean_dataset=standardize(r"C:\Users\DELL\OneDrive\Desktop\DATA\Reviews.csv")
Clean_dataset.to_csv(r"C:\Users\DELL\OneDrive\Desktop\DATA\Clean_reviews.csv", index=False)

def clean_text(text):
    text=text.lower()
    text = text.replace('<br>', ' ').replace('<br/>', ' ').replace('<br />', ' ')
    translator=str.maketrans('', '', string.punctuation)
    text=text.translate(translator)
    text = ' '.join(text.split())
    return text

def preprocess_text(df):
    df['review_text'] = df['review_text'].apply(clean_text)
    df=df[df['review_text']!=""]
    return df
Final_Dataset=preprocess_text(Clean_dataset)
Final_Dataset.to_csv(r"C:\Users\DELL\OneDrive\Desktop\DATA\Clean_reviews.csv", index=False)
print("Cleaned and preprocessed data saved to Final_reviews.csv")

