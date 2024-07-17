import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# Load the trained model
def load_trained_model():
    model = load_model("sentiment_analysis_model.h5")
    return model

# Preprocess the input review
def preprocess_review(review, tokenizer, maxlen=200):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)
    return padded_sequence

# Predict the sentiment of the review
def predict_sentiment(review, model, tokenizer):
    padded_sequence = preprocess_review(review, tokenizer)
    prediction = model.predict(padded_sequence)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment

# Load the tokenizer
def load_tokenizer():
    with open("tokenizer.json") as f:
        data = json.load(f)
        tokenizer = Tokenizer()
        tokenizer.word_index = data["word_index"]
    return tokenizer

# Streamlit app
def main():
    st.title("IMDB Movie Review Sentiment Analysis")

    # Load model and tokenizer
    model = load_trained_model()
    tokenizer = load_tokenizer()

    # Input review from the user
    review = st.text_area("Enter a movie review:", height=200)

    if st.button("Predict Sentiment"):
        if review:
            sentiment = predict_sentiment(review, model, tokenizer)
            st.write(f"The sentiment of the review is: {sentiment}")
        else:
            st.write("Please enter a review to analyze.")

if __name__ == "__main__":
    main()
