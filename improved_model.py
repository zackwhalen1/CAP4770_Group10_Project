import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder
import os

def inference_model(description):
    """Predicts the wine variety based on a text description."""
    loaded_pipeline = joblib.load('wine_variety_predictor_xgb.joblib')
    loaded_encoder = joblib.load('label_encoder.joblib')
    predicted_encoded = loaded_pipeline.predict([description])
    predicted_variety = loaded_encoder.inverse_transform(predicted_encoded)
    return predicted_variety[0]

if not (os.path.exists('wine_variety_predictor_xgb.joblib') or os.path.exists('label_encoder.joblib')):
    df = pd.read_csv('winemag-data-130k-v2.csv', index_col=0)
    df.dropna(subset=['description', 'variety'], inplace=True)

    variety_counts = df['variety'].value_counts()
    top_n_varieties = variety_counts.nlargest(20).index
    df_filtered = df[df['variety'].isin(top_n_varieties)].copy()

    data = df_filtered

    X = data['description']
    y_raw = data['variety']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    joblib.dump(label_encoder, 'label_encoder.joblib')
    print("Label encoder saved as label_encoder.joblib")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.7)),
        # Uncomment to train on CPU
        # ('clf', XGBClassifier(eval_metric='mlogloss', random_state=42))
        ('clf', XGBClassifier(eval_metric='mlogloss', tree_method='approx', device="cuda", random_state=42))
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)
    print("Model training complete.")

    print("Evaluating model on test data...")
    y_pred_encoded = pipeline.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    y_test_labels = label_encoder.inverse_transform(y_test)

    print("Classification Report:")
    print(classification_report(y_test_labels, y_pred, labels=label_encoder.classes_, zero_division=0))
    accuracy = accuracy_score(y_test, y_pred_encoded)
    print(f"Overall Accuracy: {accuracy:.4f}")

    print("Saving model and vectorizer...")
    joblib.dump(pipeline, 'wine_variety_predictor_xgb.joblib')
    print("Model saved as wine_variety_predictor_xgb.joblib")

if __name__ == "__main__":
    print("--- Wine Variety Predictor (XGBoost) ---")
    while True:
        user_input = input("Enter wine description keywords (or type 'quit' to exit): ")
        if user_input.lower() == 'quit': break
        if not user_input.strip():
            print("Please enter some keywords.")
            continue

        predicted = inference_model(user_input)
        print(f"Predicted wine variety: {predicted}")
