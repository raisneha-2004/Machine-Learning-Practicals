import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Training data
reviews = [
    "This movie is amazing",
    "I love this movie",
    "The story is excellent",
    "The movie was fantastic",
    "I enjoyed this film",
    "This movie is terrible",
    "I hate this movie",
    "The story is boring",
    "The movie was awful",
    "I disliked this film"
]

labels = [
    "Positive", "Positive", "Positive", "Positive", "Positive",
    "Negative", "Negative", "Negative", "Negative", "Negative"
]

# Create and train model
model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(reviews, labels)

# Web app
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review to check its sentiment.")

user_review = st.text_area("Enter your review:")

if st.button("Analyze Sentiment"):
    if user_review.strip():
        prediction = model.predict([user_review])[0]

        if prediction == "Positive":
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😞")
    else:
        st.warning("Please enter a review.")

