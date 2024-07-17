
# 🎬 IMDB Movie Review Sentiment Analysis 🎥

Welcome to the IMDB Movie Review Sentiment Analysis project! This project leverages a Long Short-Term Memory (LSTM) neural network model to analyze and predict the sentiment of movie reviews. The sentiment can be either positive or negative. We’ve built an interactive web interface using Streamlit to make it easy for users to input reviews and get instant sentiment predictions.

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

Make sure you have the following installed:

- Python 3.7+
- Streamlit
- TensorFlow
- Keras

### Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/yourusername/IMDB-Sentiment-Analysis.git
    cd IMDB-Sentiment-Analysis
    ```

2. **Install dependencies**

    Use pip to install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Files Included

- `streamlit_app.py` - The main Streamlit application.
- `sentiment_analysis_model.h5` - The trained LSTM model.
- `tokenizer.json` - The tokenizer used for preprocessing the text data.

### Usage

1. **Run the Streamlit app**

    ```bash
    streamlit run streamlit_app.py
    ```

2. **Open your browser**

    After running the command, Streamlit will automatically open a new tab in your default web browser. If it doesn't, you can manually open your browser and navigate to `http://localhost:8501`.

3. **Input a movie review**

    Enter a movie review in the text area provided and click on the "Predict Sentiment" button. The app will analyze the review and display whether the sentiment is positive or negative.

## 📁 Project Structure

```
IMDB-Sentiment-Analysis/
│
├── sentiment_analysis_model.h5     # Trained LSTM model
├── tokenizer.json                  # Tokenizer for text preprocessing
├── streamlit_app.py                # Streamlit application
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
└── .gitignore                      # Git ignore file
```

## 🛠️ Key Components

### Model Loading

The pre-trained LSTM model is loaded using Keras:

```python
from tensorflow.keras.models import load_model

model = load_model("sentiment_analysis_model.h5")
```

### Tokenizer Loading

The tokenizer is loaded from a JSON file to ensure the text data is preprocessed consistently:

```python
import json
from tensorflow.keras.preprocessing.text import Tokenizer

def load_tokenizer():
    with open("tokenizer.json") as f:
        data = json.load(f)
        tokenizer = Tokenizer()
        tokenizer.word_index = data["word_index"]
    return tokenizer

tokenizer = load_tokenizer()
```

### Sentiment Prediction

The sentiment prediction is done by preprocessing the input review and passing it to the LSTM model:

```python
def preprocess_review(review, tokenizer, maxlen=200):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)
    return padded_sequence

def predict_sentiment(review, model, tokenizer):
    padded_sequence = preprocess_review(review, tokenizer)
    prediction = model.predict(padded_sequence)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment
```

## 🌟 Features

- **Easy-to-use interface**: A simple and intuitive interface built with Streamlit.
- **Real-time predictions**: Get instant sentiment analysis of movie reviews.
- **Pre-trained model**: Utilizes a powerful LSTM model trained on a large dataset of IMDB reviews.
