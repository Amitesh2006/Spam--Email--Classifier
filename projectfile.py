"""
Spam Email Classifier Project
Author: Your Name
Run Command: python spam_project.py

Description:
This project trains a simple machine learning model using Naive Bayes
to classify text messages or emails as SPAM or NOT SPAM.
It uses a built-in demo dataset (no external files needed).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ------------------------------
# Step 1: Create demo dataset
# ------------------------------
def load_data():
    data = [
        ("Win money now!!!", 1),
        ("Exclusive offer for you, limited time only!", 1),
        ("Congratulations! You have won a free iPhone!", 1),
        ("Claim your $500 cash prize here!", 1),
        ("Your account is locked. Click to verify now.", 1),
        ("Urgent! You won a lottery. Respond immediately.", 1),
        ("You have been selected for a cash reward!", 1),
        ("Let's meet tomorrow for project discussion.", 0),
        ("Your appointment is confirmed at 10 AM.", 0),
        ("Can you send the assignment file?", 0),
        ("Don't forget about our meeting tomorrow.", 0),
        ("Please review the attached report.", 0),
        ("Lunch at 1 PM today?", 0),
        ("Call me when you reach home.", 0),
        ("Team meeting rescheduled to 3 PM.", 0)
    ]
    texts, labels = zip(*data)
    return list(texts), list(labels)

# ------------------------------
# Step 2: Build pipeline model
# ------------------------------
def build_pipeline():
    return Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ])

# ------------------------------
# Step 3: Train and evaluate model
# ------------------------------
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = build_pipeline()
    print("\nTraining the model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n✅ Model Training Complete!")
    print("\n📊 Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("\n📈 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n🧾 Classification Report:\n", classification_report(y_test, y_pred))
    return model

# ------------------------------
# Step 4: Save model
# ------------------------------
def save_model(model):
    joblib.dump(model, "spam_model.joblib")
    print("\n💾 Model saved as spam_model.joblib")

# ------------------------------
# Step 5: Interactive testing
# ------------------------------
def test_model(model):
    print("\n📨 Type any message to check if it's SPAM or NOT SPAM.")
    print("Type 'exit' to quit.\n")
    while True:
        msg = input(">>> ").strip()
        if msg.lower() == "exit":
            print("\n👋 Exiting Spam Classifier. Goodbye!")
            break
        pred = model.predict([msg])[0]
        print("Prediction:", "🚨 SPAM" if pred == 1 else "✅ NOT SPAM")

# ------------------------------
# Main execution
# ------------------------------
if __name__ == "__main__":
    X, y = load_data()
    model = train_and_evaluate(X, y)
    save_model(model)
    test_model(model)
