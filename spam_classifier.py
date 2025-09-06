from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Simple dataset
messages = [
    "Win a free iPhone now",        # spam
    "Congratulations, you won cash",# spam
    "Lowest price on medicines",    # spam
    "Are you coming to class?",     # ham
    "Can you send me the notes?",   # ham
    "Let's meet tomorrow"           # ham
]
labels = ["spam", "spam", "spam", "ham", "ham", "ham"]

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X, labels)

# Ask user for input
msg = input("Enter a message: ")
msg_vec = vectorizer.transform([msg])
prediction = model.predict(msg_vec)[0]

print("Prediction:", prediction)
