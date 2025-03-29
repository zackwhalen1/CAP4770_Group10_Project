import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import os


plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(font_scale=1.2)

file_path = "winemag-data-130k-v2.csv"

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\nColumns in the dataset:", df.columns.tolist())
print("\nFirst few rows of the dataset:")
print(df.head())
print("\nBasic statistics of numerical columns:")
print(df.describe())

if not os.path.exists("visualizations"): os.makedirs("visualizations")

plt.figure(figsize=(12, 6))
sns.histplot(df['points'], kde=True, bins=20)
plt.title('Distribution of Wine Scores')
plt.xlabel('Points')
plt.ylabel('Count')
plt.savefig('visualizations/wine_scores_distribution.png')
plt.close()

plt.figure(figsize=(14, 8))
country_counts = df['country'].value_counts().head(10)
sns.barplot(x=country_counts.values, y=country_counts.index)
plt.title('Top 10 Countries by Number of Wines')
plt.xlabel('Count')
plt.ylabel('Country')
plt.savefig('visualizations/top_10_countries.png')
plt.close()

plt.figure(figsize=(14, 8))
variety_counts = df['variety'].value_counts().head(10)
sns.barplot(x=variety_counts.values, y=variety_counts.index)
plt.title('Top 10 Wine Varieties')
plt.xlabel('Count')
plt.ylabel('Variety')
plt.savefig('visualizations/top_10_varieties.png')
plt.close()

plt.figure(figsize=(14, 8))
top_countries = df['country'].value_counts().head(5).index
country_scores = df[df['country'].isin(top_countries)]
sns.boxplot(x='country', y='points', data=country_scores)
plt.title('Wine Score Comparison by Top 5 Countries')
plt.xlabel('Country')
plt.ylabel('Points')
plt.savefig('visualizations/score_by_country.png')
plt.close()

plt.figure(figsize=(14, 8))
price_score_df = df.dropna(subset=['price'])
price_score_df = price_score_df[price_score_df['price'] < price_score_df['price'].quantile(0.95)]
sns.scatterplot(x='price', y='points', data=price_score_df, alpha=0.5)
sns.regplot(x='price', y='points', data=price_score_df, scatter=False, color='red')
plt.title('Price vs. Points Relationship')
plt.xlabel('Price (USD)')
plt.ylabel('Points')
plt.savefig('visualizations/price_vs_points.png')
plt.close()

plt.figure(figsize=(14, 8))
variety_price = df.groupby('variety')['price'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=variety_price.values, y=variety_price.index)
plt.title('Top 10 Most Expensive Wine Varieties (Avg. Price)')
plt.xlabel('Average Price (USD)')
plt.ylabel('Variety')
plt.savefig('visualizations/avg_price_by_variety.png')
plt.close()

plt.figure(figsize=(14, 10))
wine_descriptions = ' '.join(df['description'].dropna())
wordcloud = WordCloud(width=800, height=600, background_color='white', max_words=100, 
                        contour_width=3, contour_color='steelblue').generate(wine_descriptions)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Common Words in Wine Descriptions', fontsize=20)
plt.savefig('visualizations/wine_descriptions_wordcloud.png')
plt.close()