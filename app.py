import streamlit as st
import joblib
import re
from tensorflow import keras

st.set_page_config(page_title="Fake News Detection App", page_icon="📰")

st.title("Fake News Detection App")
st.write("Paste a news headline or short article and check whether it is predicted as Real or Fake.")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Load model and vectorizer
model = keras.models.load_model("mlp_model.h5")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

user_input = st.text_area("Enter news text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        input_features = vectorizer.transform([cleaned]).toarray()
        prediction_prob = model.predict(input_features)[0][0]

        if prediction_prob > 0.5:
            st.success("Prediction: REAL")
        else:
            st.error("Prediction: FAKE")

        st.write(f"Confidence: {prediction_prob:.4f}")