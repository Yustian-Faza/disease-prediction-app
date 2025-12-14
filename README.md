# Symptom-Based Disease Prediction App

This repository contains a Streamlit web application for interactive disease consultation based on patient‑reported symptoms. The app guides users through adaptive questions and predicts the most likely diseases using a CatBoost multiclass classifier and a symptom–disease knowledge base. This project was developed as part of the final project for Statistical Machine Learning class.

## Features

- Interactive, step‑by‑step symptom questioning flow.
- Adaptive question selection using information gain and symptom frequencies.
- Disease prediction powered by a CatBoost multiclass model with cross‑validation.
- Automatic handling of “yes/no” symptom answers and narrowing of possible diseases.
- Disease descriptions and precaution recommendations from external CSV/Excel files.
- Evaluation metrics (accuracy, precision, recall, F1) on a held‑out test set.
- Session management with `st.session_state` to support multi‑step consultations.

## Tech Stack

- Python
- Streamlit (web UI)
- CatBoost (multiclass classification)
- scikit‑learn (cross‑validation, metrics)
- pandas, NumPy (data handling)
- openpyxl (Excel I/O)

## Data Requirements

Place the following files in the same directory as `disease_prediction_app.py`:

- `Training_reordered.xlsx` – training dataset (symptom indicators + `Disease` label).
- `Testing_reordered.xlsx` – test dataset with the same structure.
- `Symptom-severity.csv` – symptom metadata (e.g., severity information).
- `disease_description.csv` – mapping from disease names to descriptions.
- `disease_precaution.csv` – mapping from disease names to precaution advice.

Ensure that disease names are consistently formatted across all files (no leading/trailing spaces).

## Installation

1. Create and activate a virtual environment.
2. Install required packages:
   - `streamlit`
   - `catboost`
   - `scikit-learn`
   - `pandas`
   - `numpy`
   - `openpyxl`

3. Clone this repository and move into the project directory.
4. Copy all required data files into the same directory as `disease_prediction_app.py`.

## How to Run

Run the Streamlit app from your terminal:

- `streamlit run disease_prediction_app.py`

Then open the URL shown in the terminal (usually `http://localhost:8501`) in your browser.

## How It Works

### 1. Data Loading and Preprocessing

The app:

- Loads training and testing data from the Excel files.
- Cleans string values, replaces `"none"` with `0`, and fills missing values with `0`.
- Builds a binary symptom matrix with one column per symptom.
- Uses a `MultiLabelBinarizer` to maintain a consistent symptom space.
- Constructs a `disease_symptom_map` listing symptoms associated with each disease.

### 2. Model Training

The training routine:

- Uses stratified k‑fold cross‑validation with a `CatBoostClassifier`.
- Computes accuracy, precision, recall, and F1‑score for each fold and on the test set.
- Fits a final CatBoost model on the full training data.
- Saves the final model and the fitted `MultiLabelBinarizer` as `.pkl` files.
- Stores evaluation metrics for later inspection.

### 3. Symptom Questioning Logic

The app computes:

- Symptom frequencies in the training data.
- Disease prior probabilities.
- Conditional probabilities of symptoms given diseases and not given diseases.

It selects the next symptom to ask based on:

- Information gain with respect to current disease probabilities.
- Priority for symptoms that discriminate among top candidate diseases.
- Exploration heuristics when many answers are “no”.
- Special triggers for particular disease–symptom pairs when needed.

Session state keeps track of:

- Current positive symptoms.
- All answered symptoms (yes and no).
- Current list of possible diseases.
- Number of questions asked and special flags.

### 4. Disease Prediction

When enough symptoms have been answered, the app:

- Builds an input feature vector from the selected symptoms.
- Uses the CatBoost model to get disease probabilities.
- Filters out diseases inconsistent with the observed “yes” and “no” symptoms.
- Renormalizes probabilities and presents the most likely diseases.
- Displays associated descriptions and precaution advice from the CSV files.

## Usage Flow

1. Open the app and click “Start Consultation”.
2. Answer the symptom questions (Yes / No) one by one.
3. When sufficient information is collected, switch to the prediction stage.
4. Review the suggested diseases, descriptions, and precautions.
5. Use “New Consultation” to reset the session and start a new case.

## Project Structure

- `disease_prediction_app.py` – main Streamlit application.
- `Training_reordered.xlsx` – training data.
- `Testing_reordered.xlsx` – test data.
- `disease_description.csv` – disease descriptions.
- `disease_precaution.csv` – precaution recommendations.
- `README.md` – project documentation.

## Notes and Limitations

- This app is intended for educational and prototyping purposes only and is **not** a replacement for professional medical diagnosis.
- Model performance depends heavily on the quality and coverage of the training data.
- The list of diseases and symptoms is limited to what is present in the provided datasets.
