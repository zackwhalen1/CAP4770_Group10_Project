#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import os

os.makedirs('visualizations', exist_ok=True)

file_path = "winemag-data-130k-v2.csv"
df = pd.read_csv(file_path)

#desciptions
df = df[['description']].dropna()

#features; exclude the unwanted words
custom_stopwords = list(ENGLISH_STOP_WORDS.union(['alongside', 'aromas', 'certainly', 'include', 'notes', 'offering', 'offers', 'palate', 'does', 'doesn', 'feature', 'features', 'flavor',
                                                   'flavored', 'flavors', 'gets', 'given', 'gives', 'giving', 'goes', 'going', 'holds', 'delivers', 'drink', 'drinking', 'feel', 'element', 'elements',
                                                    'feels', 'followed', 'like', 'make', 'tastes', 'tones', 'add', 'adds', 'come', 'comes', 'need', 'needs', 'note', 'way', 'wine', 'winemaker',
                                                    'winery', 'wines', 'year', 'years', 'll']))

#no excludes, no numbers, nothing in more than 60% of stuff, nothing that shows up less than 1000 times
vectorizer = TfidfVectorizer(stop_words=custom_stopwords, token_pattern=r'\b[a-zA-Z]{2,}\b', max_df=0.60, min_df=1000)
X = vectorizer.fit_transform(df['description'])

#check the words as a sheet (used for checking custom stopwords)
#print("sheet?")
#words = vectorizer.get_feature_names_out()
#vocab_df = pd.DataFrame({'word': words})
#vocab_df.to_excel('visualizations/tfidf_vocabulary.xlsx', index=False)
#print("END")

#Excel File Please (used for screenshots in presentation)
#print("sheet?")
#tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
#tfidf_df.to_excel('visualizations/tfidf_matrix.xlsx', index=False)
#print("Saved TF-IDF matrix to visualizations/tfidf_matrix.xlsx")