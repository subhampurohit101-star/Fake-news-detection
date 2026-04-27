import streamlit as st
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------
# Load and Balance Dataset
# ---------------------------
@st.cache_data
def load_data():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")

    fake["label"] = 0
    true["label"] = 1

    # Balance dataset
    min_len = min(len(fake), len(true))
    fake = fake.sample(min_len, random_state=42)
    true = true.sample(min_len, random_state=42)

    data = pd.concat([fake, true])
    data = data.sample(frac=1, random_state=42)

    return data[["text", "label"]]

data = load_data()

# ---------------------------
# Clean Text
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data["text"] = data["text"].apply(clean_text)

# ---------------------------
# Train Model
# ---------------------------
@st.cache_resource
def train_model(data):
    X = data["text"]
    y = data["label"]

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, vectorizer, acc

model, vectorizer, accuracy = train_model(data)

# ---------------------------
# UI
# ---------------------------
st.title("📰 Fake News Detection System")
st.write("Enter a news article to check if it's Real or Fake")

st.subheader("📊 Model Accuracy")
st.write(f"Accuracy: **{accuracy:.4f}**")

# ---------------------------
# Prediction# ---------------------------
user_input = st.text_area("Enter News Text")

if st.button("Analyze"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        result = model.predict(vector)[0]

        if result == 1:
            st.success("✅ Real News")
        else:
            st.error("❌ Fake News")

        # Debug (optional)
        st.write("Prediction Code:", result)

    else:
        st.warning("Please enter some text")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Developed using Machine Learning & Streamlit")