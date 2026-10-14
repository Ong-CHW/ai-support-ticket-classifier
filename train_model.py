
import pandas as pd

train_df = pd.read_json("data/train.jsonl", lines=True)
val_df = pd.read_json("data/val.jsonl", lines=True)

# Remove missing or empty messages
train_df = train_df.dropna(subset=["text", "label"]).copy()
val_df = val_df.dropna(subset=["text", "label"]).copy()

train_df["text"] = train_df["text"].str.strip()
val_df["text"] = val_df["text"].str.strip()

train_df = train_df[train_df["text"] != ""]
val_df = val_df[val_df["text"] != ""]

# Separate inputs (X) from labels (y)
X_train = train_df["text"]
y_train = train_df["label"]

X_val = val_df["text"]
y_val = val_df["label"]

print("Training examples:", len(X_train))
print("Validation examples:", len(X_val))


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

print("Model training completed!")


from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)

print(f"Validation Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_val, y_pred, zero_division=0))

# Create a DataFrame containing validation results
results = val_df[["text", "label"]].copy()

# Add the model's predicted categories
results["predicted"] = y_pred

# Find rows where the prediction doesn't match the actual label
mistakes = results[results["label"] != results["predicted"]]

print(f"\nIncorrect predictions: {len(mistakes)}")

print(mistakes.to_string(index=False))


custom_tickets = [
    "I was charged twice for my subscription.",
    "Every time I open the app, it crashes.",
    "I want to cancel my account permanently.",
    "How do I authenticate requests to your API?",
    "I would like to upgrade to a premium plan.",
    "Your customer service is absolutely terrible."
]

predictions = model.predict(custom_tickets)

for ticket, prediction in zip(custom_tickets, predictions):
    print(f"\nTicket: {ticket}")
    print(f"Predicted category: {prediction}")


# Load the test dataset
test_df = pd.read_json("data/test.jsonl", lines=True)

# Separate input messages and correct labels
X_test = test_df["text"]
y_test = test_df["label"]

# Generate predictions
y_test_pred = model.predict(X_test)

# Calculate test accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\nFinal Test Accuracy: {test_accuracy:.2%}")

print("\nFinal Test Classification Report:")
print(classification_report(y_test, y_test_pred, zero_division=0))


from pathlib import Path
import joblib

# Define where our trained model will be saved
model_path = Path("models/support_ticket_classifier.joblib")

# Create the models folder if it doesn't exist
model_path.parent.mkdir(parents=True, exist_ok=True)

# Save our trained model
joblib.dump(model, model_path)

print(f"\nModel saved successfully to: {model_path}")
