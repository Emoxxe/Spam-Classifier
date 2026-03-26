# 🛡️ AI Spam Email Classifier

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)

A powerful Machine Learning web application utilizing Natural Language Processing (NLP) to classify text messages and emails as either **Spam** or **Ham (Not Spam)**. The application handles text cleaning, tokenization, and vectorization natively before passing it through pre-trained algorithmic models.

---

## ✨ Features
* **Live Message Prediction:** Instantly detects if a custom message is Spam with real-time confidence scores.
* **Dual AI Models:** Toggle instantly between a **Naive Bayes (`MultinomialNB`)** model and a **Logistic Regression** model for comparison.
* **Data Visualizations:** Automatically visually renders "Word Clouds" to display the most frequent terms used by real-world scammers based on the SMS Spam Collection dataset.
* **NLTK Text Pre-Processing:** Programmatically removes punctuation, converts text to lowercase, and strips standard English stopwords to increase AI accuracy.

---

## 💻 Tech Stack
* **Frontend/UI:** [Streamlit](https://streamlit.io/)
* **NLP & Text Cleaning:** [NLTK](https://www.nltk.org/)
* **Machine Learning Engine:** Scikit-Learn
* **Math & Vectors:** Python, NumPy & Pandas

---

## 🚀 How to Run Locally

If you want to run this application on your own computer:

1. Clone this repository to your machine.
2. Ensure you have Python 3.12 installed.
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```
