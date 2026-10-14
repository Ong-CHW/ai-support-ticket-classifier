
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# App configuration
st.set_page_config(
    page_title="TicketFlow | AI Ticket Classifier",
    page_icon="🎫",
    layout="wide"
)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "support_ticket_classifier.joblib"
)


# Example support tickets
EXAMPLES = {
    "Choose an example": None,
    "Billing - Duplicate charge":
        "I was charged twice for my monthly subscription.",
    "Technical - Application crashing":
        "The application crashes every time I upload a file.",
    "Cancellation - End subscription":
        "I want to cancel my subscription immediately.",
    "API - Authentication":
        "How do I generate an API key and authenticate requests?",
    "Upgrade - Premium plan":
        "I would like to upgrade my current plan to premium.",
    "Complaint - Poor service":
        "Your customer support team has been extremely unhelpful."
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# Update the text box when an example is selected
def use_example():
    selected = st.session_state.example

    if EXAMPLES[selected] is not None:
        st.session_state.ticket_text = EXAMPLES[selected]
        st.session_state.pop("result", None)


# Initialize session state
if "ticket_text" not in st.session_state:
    st.session_state.ticket_text = ""


# Minor visual styling
st.markdown("""
<style>
    .block-container {
        max-width: 1150px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }

    h1 {
        letter-spacing: -0.035em;
    }
</style>
""", unsafe_allow_html=True)


# Header
st.caption("TICKETFLOW  /  MACHINE LEARNING")

st.title("AI Support Ticket Classifier")

st.write(
    "Automatically categorize customer support messages "
    "using natural language processing and machine learning."
)


# Project statistics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Ticket categories", "6", border=True)

with col2:
    st.metric("Synthetic test accuracy", "99.15%", border=True)

with col3:
    st.metric("ML model", "Logistic Regression", border=True)


st.write("")

# Main interface
left, right = st.columns([1.1, 1], gap="large")


# Left panel: User input
with left:
    with st.container(border=True):
        st.subheader("Classify a ticket")

        st.caption(
            "Enter a customer message or select a sample ticket."
        )

        st.selectbox(
            "Try an example",
            options=list(EXAMPLES.keys()),
            key="example",
            on_change=use_example
        )

        st.text_area(
            "Customer message",
            key="ticket_text",
            placeholder="Describe the customer's issue...",
            height=170
        )

        classify = st.button(
            "Classify ticket",
            type="primary",
            use_container_width=True
        )

        if classify:
            ticket_text = st.session_state.ticket_text.strip()

            # Clear previous prediction
            st.session_state.pop("result", None)

            if len(ticket_text.split()) < 3:
                st.warning(
                    "Please enter a meaningful support message "
                    "with at least three words."
                )

            elif not MODEL_PATH.is_file():
                st.error(
                    "Trained model not found. Run train_model.py first."
                )

            else:
                try:
                    model = load_model()

                    prediction = model.predict([ticket_text])[0]
                    probabilities = model.predict_proba(
                        [ticket_text]
                    )[0]

                    scores = pd.DataFrame({
                        "Category": model.classes_,
                        "Probability": probabilities
                    })

                    scores = scores.sort_values(
                        "Probability",
                        ascending=False
                    )

                    # Store results between Streamlit reruns
                    st.session_state.result = {
                        "text": ticket_text,
                        "prediction": prediction,
                        "scores": scores
                    }

                except Exception as error:
                    st.error(f"Classification failed: {error}")


# Right panel: Results
with right:
    with st.container(border=True):
        st.subheader("Classification result")

        result = st.session_state.get("result")

        if (
            result is not None
            and result["text"] == st.session_state.ticket_text.strip()
        ):
            scores = result["scores"]
            prediction = result["prediction"]
            top_probability = float(scores.iloc[0]["Probability"])

            st.caption("PREDICTED CATEGORY")
            st.markdown(f"### {prediction.capitalize()}")

            st.metric(
                "Highest estimated probability",
                f"{top_probability:.1%}"
            )

            if top_probability < 0.5:
                st.warning(
                    "The model does not strongly favor one category. "
                    "Consider reviewing this ticket manually."
                )

            st.divider()
            st.markdown("**Category probabilities**")

            # Horizontal probability bars
            for row in scores.itertuples(index=False):
                name_col, bar_col, pct_col = st.columns(
                    [1.6, 3, 0.9],
                    vertical_alignment="center"
                )

                with name_col:
                    st.caption(row.Category.capitalize())

                with bar_col:
                    st.progress(float(row.Probability))

                with pct_col:
                    st.caption(f"{row.Probability:.1%}")

            st.caption(
                "Probabilities are model estimates and "
                "have not been calibrated."
            )

        else:
            st.info(
                "Your classification results will appear here."
            )

            st.caption(
                "Enter a message and click Classify ticket to begin."
            )


# Footer
st.divider()

st.caption(
    "Built with Python, scikit-learn, TF-IDF, "
    "Logistic Regression and Streamlit. "
    "Trained on synthetic support ticket data."
)
