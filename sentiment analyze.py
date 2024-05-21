import nltk
import numpy as np
import tensorflow as tf
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK data if not already downloaded
nltk.download('vader_lexicon')
nltk.download('punkt')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    """
    Analyzes the sentiment of the given text using a simple ANN.

    Args:
    text (str): The text to analyze.

    Returns:
    str: Sentiment label ('positive', 'neutral', or 'negative').
    """
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Initialize counters for positive, negative, and neutral sentiments
    pos_count = 0
    neg_count = 0
    neu_count = 0

    # Analyze each sentence and aggregate sentiment counts
    for sentence in sentences:
        # Get the polarity scores for the sentence
        scores = sid.polarity_scores(sentence)

        # Determine sentiment based on the compound score
        if scores['compound'] >= 0.05:
            pos_count += 1
        elif scores['compound'] <= -0.05:
            neg_count += 1
        else:
            neu_count += 1

    # Determine overall sentiment based on counts
    if pos_count > neg_count and pos_count > neu_count:
        return 'positive'
    elif neg_count > pos_count and neg_count > neu_count:
        return 'negative'
    else:
        return 'neutral'


# Example usage
text = "I hate this product! It's unworthy."
sentiment = analyze_sentiment(text)
print("Sentiment:", sentiment)

# Additional ANN related code
# Convert sentiment labels to numeric values
sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}

# Create training data (dummy data)
X_train = np.array([[1.0, 0.5, 0.8], [0.2, 0.7, 0.1], [0.9, 0.2, 0.3]])
y_train = np.array([sentiment_mapping['positive'], sentiment_mapping['neutral'], sentiment_mapping['negative']])

# Define a simple ANN model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=1)

# Example prediction using the trained model
prediction = model.predict(np.array([[0.8, 0.6, 0.9]]))
predicted_sentiment = 'positive' if prediction > 0.5 else 'negative'
print("Predicted Sentiment:", predicted_sentiment)
