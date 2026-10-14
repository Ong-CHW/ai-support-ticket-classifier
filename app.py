
from pathlib import Path

import joblib
import streamlit as st


# Path to our saved machine learning model
MODEL_PATH = Path(__file__).resolve().parent / "models" / "support_ticket_classifier.joblib"


# Load the model and cache it
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# Configure the web application
st.set_page_config(
    page_title="AI Support Ticket Classifier",
    page_icon="🎫🤓🤓🤓🤓"
)

st.title("AI Support Ticket Classifier")

st.write(
    "Enter a customer support message to automatically "
    "classify it using machine learning."
)


# Text input for the customer support ticket
ticket_text = st.text_area(
    "Customer Support Message",
    placeholder="e.g. I was charged twice for my subscription.",
    height=150
)


# Classification button
if st.button("Classify Ticket"):

    if not ticket_text.strip():
        st.warning("Please enter a support message.")

    elif not MODEL_PATH.is_file():
        st.error("Trained model not found. Run train_model.py first.")

    else:
        model = load_model()

        prediction = model.predict([ticket_text])[0]

        st.success(
            f"Predicted Category: {prediction.capitalize()}"
        )
