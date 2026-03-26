import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle

# Ensure dependencies are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def download_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = "smsspamcollection.zip"
    extract_dir = "data"
    
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        os.remove(zip_path)
    return os.path.join(extract_dir, "SMSSpamCollection")

def clean_text(text):
    text = text.lower() # Lowercase
    text = re.sub(f"[{string.punctuation}]", "", text) # Remove punctuation
    tokens = nltk.word_tokenize(text) # Tokenize
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words] # Remove stopwords
    return " ".join(tokens)

def main():
    print("Loading data...")
    data_path = download_data()
    # Read the TSV file
    df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'message'])
    
    print("Formatting and cleaning text (this may take a few seconds)...")
    # Convert labels: spam=1, ham=0
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    df['cleaned_message'] = df['message'].apply(clean_text)
    
    print("Vectorizing data via TF-IDF...")
    X = df['cleaned_message']
    y = df['label']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
    X_test_tfidf = vectorizer.transform(X_test).toarray()
    
    print("\n--- Training Naive Bayes (MultinomialNB) ---")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_predictions = nb_model.predict(X_test_tfidf)
    
    print(f"Accuracy:  {accuracy_score(y_test, nb_predictions):.4f}")
    print(f"Precision: {precision_score(y_test, nb_predictions):.4f}")
    print(f"Recall:    {recall_score(y_test, nb_predictions):.4f}")
    print(f"F1-Score:  {f1_score(y_test, nb_predictions):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, nb_predictions))
    
    print("\n--- Training Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    lr_predictions = lr_model.predict(X_test_tfidf)
    
    print(f"Accuracy:  {accuracy_score(y_test, lr_predictions):.4f}")
    print(f"Precision: {precision_score(y_test, lr_predictions):.4f}")
    print(f"Recall:    {recall_score(y_test, lr_predictions):.4f}")
    print(f"F1-Score:  {f1_score(y_test, lr_predictions):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, lr_predictions))
    
    print("\nSaving models and vectorizer to 'models/'...")
    if not os.path.exists('models'):
        os.makedirs('models')
    
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('models/nb_model.pkl', 'wb') as f:
        pickle.dump(nb_model, f)
    with open('models/lr_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
        
    # Save dataframe to csv for word cloud reading later without redownloading
    df_clean = df[['label', 'cleaned_message']]
    df_clean.to_csv('models/cleaned_data.csv', index=False)
        
    print("Training complete! Run 'streamlit run app.py' to launch the UI.")

if __name__ == "__main__":
    main()
