Source code:

!pip install scikit-learn pandas numpy python-telegram-bot

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

# Load data
# Assuming 'spam.csv' has columns 'Message' and 'Category'
df = pd.read_csv('/content/spam.csv')

# Data cleaning and preprocessing if needed
# Example: removing punctuation, stop words, etc.

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Message'])
y = df['Category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))

# Model persistence
joblib.dump(clf, 'spam_classifier.pkl')

# Save the vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')

#Testing
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Load the saved model
clf = joblib.load('/content/spam_classifier.pkl')

# Load the saved vectorizer
vectorizer = joblib.load('/content/vectorizer.pkl')

# Example input for testing
new_messages = [
    "Free offer! Click here now!",
    "Hi John, how are you today?"
]

# Transform new input using the loaded vectorizer
X_new = vectorizer.transform(new_messages)

# Predict categories
predictions = clf.predict(X_new)

# Print predictions
for message, category in zip(new_messages, predictions):
    print(f"Input Message: {message}")
    print(f"Predicted Category: {category}")
    print("---")

#Integrating with a Telegram bot
!pip install telebot
import numpy as np
import pandas as pd
import joblib
import telebot
from sklearn.feature_extraction.text import CountVectorizer

# Load the trained spam email classifier model
clf = joblib.load('/content/spam_classifier.pkl')

# Load the saved vectorizer
vectorizer = joblib.load('/content/vectorizer.pkl')

TOKEN = "6505899626:AAHCDm5GF2cb9uj6bxHJ8A5wGwX3GKXMuwE"
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Welcome to the Spam Email Classifier Bot! Send me an email text and I'll classify it as spam or not spam.")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    text = message.text
    prediction = classify_email(text, vectorizer, clf)
    bot.reply_to(message, prediction)

def classify_email(text: str, vectorizer, clf) -> str:
    # Transform input using the loaded vectorizer
    msg_features = vectorizer.transform([text])

    # Predict category
    prediction = clf.predict(msg_features)[0]

    if prediction == 'spam':
        return "This email is classified as spam."
    else:
        return "This email is classified as not spam."

bot.polling()
