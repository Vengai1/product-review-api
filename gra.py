import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\DATA\Clean_reviews.csv")
target_product = input("Enter the product ID to analyze: ")
df[df['product_id'] == target_product]['rating(out of 5)'].value_counts().sort_index().plot(kind='bar', color='royalblue')
plt.title(f'Ratings for {target_product}')
plt.show()