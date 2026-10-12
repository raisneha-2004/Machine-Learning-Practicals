
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 1: Training data
sentences = [
    "I love this product",
    "This is amazing",
    "I am very happy",
    "This is a good movie",
    "I hate this product",
    "This is terrible",
    "I am very sad",
    "This is a bad movie"
]

# Step 2: Sentiment labels
labels = [
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Negative",
    "Negative",
    "Negative",
    "Negative"
]

# Step 3: Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

# Step 4: Train the model
model = MultinomialNB()
model.fit(X, labels)

# Step 5: Test new sentences
test_sentences = [
    "I love this product",
    "This is terrible",
    "I am very happy",
    "This is a bad movie",
    "This is a good movie"
]

# Step 6: Predict sentiment
test_data = vectorizer.transform(test_sentences)
predictions = model.predict(test_data)

# Step 7: Display results
for sentence, sentiment in zip(test_sentences, predictions):
    print("Sentence:", sentence)
    print("Sentiment:", sentiment)
    print()