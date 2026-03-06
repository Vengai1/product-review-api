import pandas as pd
from collections import Counter

def guess_product_name_v2(filepath, target_product_id):
    print(f"Loading AI Name Extractor V2 for {target_product_id}...")
    df = pd.read_csv(filepath)
    
    product_df = df[df['product_id'] == target_product_id]
    
    if product_df.empty:
        print("Error: Product not found.")
        return
        
    all_text = " ".join(product_df['review_text'].astype(str).tolist())
    words = all_text.split()
    
    #Expanded Stop Words: Now includes adjectives and extra filler words!
    boring_words = {
        'the', 'a', 'and', 'to', 'of', 'in', 'is', 'it', 'for', 'this', 'that',
        'i', 'my', 'you', 'with', 'on', 'was', 'as', 'are', 'at', 'but', 'be',
        'have', 'product', 'item', 'buy', 'bought', 'good', 'great', 'like',
        'love', 'very', 'just', 'so', 'not', 'they', 'them', 'these', 'flavor',
        'taste', 'one', 'would', 'get', 'can', 'we', 'were', 'out', 'from',
        'healthy', 'delicious', 'fresh', 'yummy', 'sweet', 'bad', 'also', 'their'
    }
    
    # 1. Filter out the boring words
    meaningful_words = [word for word in words if word not in boring_words and len(word) > 2]
    
    # 2. BIGRAM MAGIC: Create pairs of connected words using a list comprehension
    bigrams = [f"{meaningful_words[i]} {meaningful_words[i+1]}" for i in range(len(meaningful_words)-1)]
    
    # 3. Count both Unigrams (single words) and Bigrams (word pairs)
    unigram_counts = Counter(meaningful_words)
    bigram_counts = Counter(bigrams)
    
    # 4. Display the AI's thought process clearly
    print("\n" + "="*45)
    print(f" AI ANALYSIS FOR: {target_product_id}")
    print(f"Data available: {len(product_df)} reviews\n")
    
    print("Top Single Words (Unigrams):")
    for word, count in unigram_counts.most_common(3):
        print(f" - {word.title()} ({count} times)")
        
    print("\nTop Word Pairs (Bigrams):")
    for bigram, count in bigram_counts.most_common(3):
        print(f" - {bigram.title()} ({count} times)")
        
    # Pick the top Bigram as the final guess
    if bigram_counts:
        best_guess = bigram_counts.most_common(1)[0][0].title()
    else:
        best_guess = "Unknown Product"
        
    print(f"\n Final AI Guess:  {best_guess} ")
    print("="*45 + "\n")

# --- Run the Prototype ---
CLEAN_FILE = r"C:\Users\DELL\OneDrive\Desktop\DATA\Clean_reviews.csv"

# Testing it on the Dog Food ID from your screenshot!
product_id=input("Enter the product ID to analyze: ")
guess_product_name_v2(CLEAN_FILE, product_id)