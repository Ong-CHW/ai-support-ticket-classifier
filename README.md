
# AI Support Ticket Classifier

A machine learning application that automatically classifies customer support messages into six categories using Natural Language Processing (NLP).

Built with Python, scikit-learn, and Streamlit.

## Demo

![Application Screenshot](assets/demo.png)

## Features

- Automatic classification of customer support tickets
- Six categories: API, Billing, Cancellation, Complaint, Technical, and Upgrade
- TF-IDF text vectorization
- Logistic Regression classifier
- Model-estimated category probabilities
- Interactive Streamlit dashboard
- Sample tickets for testing predictions
- Input validation and low-confidence warnings

## Tech Stack

- **Python** - Main programming language
- **pandas** - Data loading and preparation
- **scikit-learn** - Machine learning pipeline and evaluation
- **TF-IDF** - Text feature extraction
- **Logistic Regression** - Multi-class classification
- **Streamlit** - Interactive web application
- **joblib** - Model serialization

## Model Performance

| Metric | Result |
|---|---|
| Training samples | 9,354 |
| Validation samples | 1,169 |
| Test samples | 1,170 |
| Validation accuracy | 98.89% |
| Test accuracy | 99.15% |

The model was evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

**Note:** The dataset contains synthetic support tickets. Performance on real-world customer messages may differ, and the predicted probabilities have not been calibrated.

## How It Works

1. A customer enters a support message.
2. TF-IDF converts the text into numerical features.
3. Logistic Regression predicts the most likely category.
4. The application displays the predicted category and estimated probabilities.

The TF-IDF vectorizer and classifier are combined in a scikit-learn Pipeline.

## Dataset

Dataset: [Support Ticket Router 12K Cleaned](https://huggingface.co/datasets/cngchis/Support-Ticket-Router-12K-Cleaned)

The dataset contains labeled synthetic customer support messages across six categories.

The training, validation, and test datasets are kept separate to evaluate model performance on unseen samples.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ong-CHW/ai-support-ticket-classifier.git
cd ai-support-ticket-classifier
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Install dependencies

On Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Download the dataset

Download `train.jsonl`, `val.jsonl`, and `test.jsonl` from the dataset's `processed` directory on Hugging Face.

Place all three files inside the `data/` directory.

### 5. Train and save the model

```powershell
.\.venv\Scripts\python.exe train_model.py
```

This trains the model, evaluates performance, and saves the trained Pipeline to the `models/` directory.

### 6. Launch the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL displayed in the terminal, typically:

http://localhost:8501

## Project Structure

```text
ai-support-ticket-classifier/
├── .streamlit/
│   └── config.toml
├── assets/
│   └── demo.png
├── data/                   # Not tracked by Git
├── models/                 # Generated locally
├── app.py
├── train_model.py
├── explore.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitations

- Training data is synthetic and may not represent real customer messages.
- The classifier assigns one category per message, even when multiple categories may apply.
- Estimated probabilities are not calibrated confidence scores.
- The model may perform poorly on unfamiliar or ambiguous messages.

## Future Improvements

- Evaluate the classifier on real-world support tickets
- Add multi-label ticket classification
- Improve probability calibration
- Add automatic ticket prioritization
