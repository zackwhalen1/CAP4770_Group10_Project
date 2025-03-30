#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import os

os.makedirs('visualizations', exist_ok=True)

file_path = "winemag-data-130k-v2.csv"
df = pd.read_csv(file_path)

#10 desciptions
df = df[['description']].dropna().head(10)

#features; exclude the unwanted words
custom_stopwords = list(ENGLISH_STOP_WORDS.union(['2012', '2016', 'alongside', 'aromas', 'certainly', 'include', 'notes', 'offering', 'offers', 'palate']))

vectorizer = TfidfVectorizer(stop_words=custom_stopwords)
X = vectorizer.fit_transform(df['description'])

#Excel File Please
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
tfidf_df.to_excel('visualizations/tfidf_matrix.xlsx', index=False)
print("Saved TF-IDF matrix to visualizations/tfidf_matrix.xlsx")