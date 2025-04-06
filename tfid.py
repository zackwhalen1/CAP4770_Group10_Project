#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import LabelEncoder
import os

#runtime A
import time
start_time = time.time()

os.makedirs('visualizations', exist_ok=True)

file_path = "winemag-data-130k-v2.csv"
df = pd.read_csv(file_path)

#desciptions, labeled onlt
df = df[['description', 'variety']].dropna()

#varieties
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['variety'])

#features; exclude the unwanted words
custom_stopwords = list(ENGLISH_STOP_WORDS.union(['alongside', 'aromas', 'certainly', 'include', 'notes', 'offering', 'offers', 'palate', 'does', 'doesn', 'feature', 'features', 'flavor',
                                                   'flavored', 'flavors', 'gets', 'given', 'gives', 'giving', 'goes', 'going', 'holds', 'delivers', 'drink', 'drinking', 'feel', 'element', 'elements',
                                                    'feels', 'followed', 'like', 'make', 'tastes', 'tones', 'add', 'adds', 'come', 'comes', 'need', 'needs', 'note', 'way', 'wine', 'winemaker',
                                                    'winery', 'wines', 'year', 'years', 'll']))

#no excludes, no numbers, nothing in more than 60% of stuff, nothing that shows up less than 1000 times
vectorizer = TfidfVectorizer(stop_words=custom_stopwords, token_pattern=r'\b[a-zA-Z]{2,}\b', max_df=0.60, min_df=1000)
x = vectorizer.fit_transform(df['description'])

#check the words as a sheet (used for checking custom stopwords)
#print("sheet?")
#words = vectorizer.get_feature_names_out()
#vocab_df = pd.DataFrame({'word': words})
#vocab_df.to_excel('visualizations/tfidf_vocabulary.xlsx', index=False)
#print("END")

#Excel File Please (used for screenshots in presentation)
#print("sheet?")
#tfidf_df = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())
#tfidf_df.to_excel('visualizations/tfidf_matrix.xlsx', index=False)
#print("Saved TF-IDF matrix to visualizations/tfidf_matrix.xlsx")

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

# Keep only the top N most common wine varieties
print("Number of total wine varieties: ")
print(df['variety'].nunique())

top_n = 20
top_varieties = df['variety'].value_counts().nlargest(top_n).index
df = df[df['variety'].isin(top_varieties)]

# Re-encode labels and TF-IDF after filtering
y = label_encoder.fit_transform(df['variety'])
x = vectorizer.fit_transform(df['description'])

# Splitting the data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

#Training/initializing the models
model_MNB = MultinomialNB().fit(x_train, y_train)
model_RF = RandomForestClassifier(n_estimators=20, random_state=42).fit(x_train, y_train)
model_DTC = DecisionTreeClassifier().fit(x_train, y_train)

#Model dictionary for for loop
models = {
    "Multinomial Naive Bayes": model_MNB,
    "Random Forest": model_RF,
    "Decision Tree": model_DTC
}

for name, model in models.items():
    y_pred = model.predict(x_test)
    error = 1 - accuracy_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                annot=False, fmt='d')
    plt.title(
        f"{name} - Confusion Matrix\n"
        f"Accuracy: {accuracy:.3f} | Error Rate: {error:.3f} | F1 Score: {f1:.3f}"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"visualizations/{name.replace(' ', '_')}_confusion_matrix.png")
    plt.close()

#runtime B
end_time = time.time()
print(f"\nRuntime: {end_time - start_time:.2f} seconds")
   