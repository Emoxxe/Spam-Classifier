import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

st.set_page_config(page_title="Spam Classifier", layout="wide", page_icon="🛡️")

# Ensure NLTK components exist before processing runtime user input
try:
    set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

@st.cache_resource
def load_models():
    try:
        with open('models/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('models/nb_model.pkl', 'rb') as f:
            nb_model = pickle.load(f)
        with open('models/lr_model.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        return vectorizer, nb_model, lr_model
    except FileNotFoundError:
        return None, None, None

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

vectorizer, nb_model, lr_model = load_models()

st.title("🛡️ AI Spam Email Classifier")
st.markdown("Predict whether a message is **Spam** or **Ham (Not Spam)** using Natural Language Processing.")

if vectorizer is None:
    st.error("Models not found! Please open your terminal and run `python train_model.py` first to generate the pickled models and vectorizer.")
    st.stop()

# Main Prediction UI
st.sidebar.header("Model Settings")
model_choice = st.sidebar.radio("Choose the model to use:", ["Naive Bayes", "Logistic Regression"])
active_model = nb_model if model_choice == "Naive Bayes" else lr_model

st.subheader("Test a Custom Message 📝")
user_input = st.text_area(
    "Type or paste a long email/SMS message here:", 
    height=250, 
    placeholder="e.g. WINNER! Claim your free $1000 prize now by clicking this link!!\n\n(You can paste very long emails here and the box will fit them...)"
)

if st.button("Predict Message", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter a message to predict.")
    else:
        # Preprocess using the exact same logic
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned]).toarray()
        
        # Predict
        prediction = active_model.predict(vectorized)[0]
        prob = active_model.predict_proba(vectorized)[0]
        confidence = prob[prediction] * 100
        
        if prediction == 1:
            st.error(f"🚨 **SPAM DETECTED!** (Confidence: {confidence:.1f}%)")
        else:
            st.success(f"✅ **NOT SPAM (Ham).** (Confidence: {confidence:.1f}%)")

st.markdown("---")
st.subheader("Dataset Visualization: Word Clouds ☁️")
st.markdown("See the most common words found in Spam messages vs Normal (Ham) messages based on the training data.")

if st.checkbox("Generate Word Clouds"):
    if os.path.exists("models/cleaned_data.csv"):
        df = pd.read_csv("models/cleaned_data.csv")
        # Handle nan values that might occur if a text was completely cleaned out
        df['cleaned_message'] = df['cleaned_message'].fillna("")
        
        spam_words = " ".join(df[df['label'] == 1]['cleaned_message'])
        ham_words = " ".join(df[df['label'] == 0]['cleaned_message'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔥 Top Spam Words")
            wc_spam = WordCloud(width=600, height=400, background_color="black", colormap="Reds").generate(spam_words)
            fig_spam, ax_spam = plt.subplots(figsize=(6,4))
            ax_spam.imshow(wc_spam, interpolation="bilinear")
            ax_spam.axis("off")
            st.pyplot(fig_spam)
            
        with col2:
            st.markdown("### 🟢 Top Ham Words")
            wc_ham = WordCloud(width=600, height=400, background_color="white", colormap="Greens").generate(ham_words)
            fig_ham, ax_ham = plt.subplots(figsize=(6,4))
            ax_ham.imshow(wc_ham, interpolation="bilinear")
            ax_ham.axis("off")
            st.pyplot(fig_ham)
    else:
        st.warning("Cleaned dataset not found. Please train the model to generate visualization data.")
