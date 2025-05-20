import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import streamlit as st

# --- Load and Preprocess Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('dataset_common_diseases.csv')
    df_symptom = pd.read_csv('symptom_severity_common.csv')
    df_desc = pd.read_csv('disease_description_common.csv')
    df_precaution = pd.read_csv('disease_precaution_common.csv')

    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x).fillna("none")
    df_desc['Disease'] = df_desc['Disease'].str.strip()
    df_precaution['Disease'] = df_precaution['Disease'].str.strip()

    # Create set of all unique symptoms from Symptom_1 - Symptom_17 columns
    symptom_columns = sorted(set(
        val for col in df.columns[1:] 
        for val in df[col].unique() if val != 'none'
    ))

    # Manual one-hot encoding
    one_hot = np.zeros((df.shape[0], len(symptom_columns)), dtype=int)
    for i, row in df.iterrows():
        for col in df.columns[1:]:
            symptom = row[col]
            if symptom != 'none':
                idx = symptom_columns.index(symptom)
                one_hot[i, idx] = 1

    X = one_hot
    y = df['Disease'].values

    return df, df_symptom, X, y, symptom_columns, df_desc, df_precaution

# Load data
df, df_symptom, X, y, symptom_columns, df_desc, df_precaution = load_data()

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y_encoded, train_size=0.85, shuffle=True, random_state=42)

# Train models
model_xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
model_xgb.fit(x_train, y_train)

model_svm = SVC(probability=True, random_state=42)
model_svm.fit(x_train, y_train)

model_log = LogisticRegression(max_iter=1000, random_state=42)
model_log.fit(x_train, y_train)

def create_input_vector(selected_symptoms, all_symptoms):
    input_vec = np.zeros(len(all_symptoms), dtype=int)
    for symptom in selected_symptoms:
        if symptom != "None" and symptom in all_symptoms:
            idx = all_symptoms.index(symptom)
            input_vec[idx] = 1
    return input_vec.reshape(1, -1)

# Streamlit UI
st.title("🩺 Disease Predictor")
st.header("Input up to 17 Symptoms")

symptom_options = sorted(df_symptom['Symptom'].dropna().unique())
symptom_inputs = []

# Create columns for better layout
cols = st.columns(3)
col_index = 0

for i in range(17):
    with cols[col_index]:
        symptom = st.selectbox(f"Symptom {i+1}", ["None"] + symptom_options, key=f"symptom_{i+1}")
        symptom_inputs.append(symptom)
    col_index = (col_index + 1) % 3

if st.button("Predict Disease", type="primary"):
    if all(s == "None" for s in symptom_inputs):
        st.warning("⚠️ Please select at least one symptom.")
    elif len(set([s for s in symptom_inputs if s != "None"])) != len([s for s in symptom_inputs if s != "None"]):
        st.warning("⚠️ Duplicate symptoms selected. Please choose unique symptoms.")
    else:
        input_data = create_input_vector(symptom_inputs, symptom_columns)

        # XGB top 3
        probs_xgb = model_xgb.predict_proba(input_data)[0]
        top3_idx_xgb = probs_xgb.argsort()[-3:][::-1]
        top3_preds_xgb = le.inverse_transform(top3_idx_xgb)
        top3_probs_xgb = probs_xgb[top3_idx_xgb]

        # SVM top 3
        probs_svm = model_svm.predict_proba(input_data)[0]
        top3_idx_svm = probs_svm.argsort()[-3:][::-1]
        top3_preds_svm = le.inverse_transform(top3_idx_svm)
        top3_probs_svm = probs_svm[top3_idx_svm]

        # Logistic top 3
        probs_log = model_log.predict_proba(input_data)[0]
        top3_idx_log = probs_log.argsort()[-3:][::-1]
        top3_preds_log = le.inverse_transform(top3_idx_log)
        top3_probs_log = probs_log[top3_idx_log]

        st.subheader("Top 3 Predictions per Model")

        st.markdown("**XGBoost:**")
        for disease, prob in zip(top3_preds_xgb, top3_probs_xgb):
            st.write(f"- {disease}: {prob*100:.2f}%")

        st.markdown("**SVM:**")
        for disease, prob in zip(top3_preds_svm, top3_probs_svm):
            st.write(f"- {disease}: {prob*100:.2f}%")

        st.markdown("**Logistic Regression:**")
        for disease, prob in zip(top3_preds_log, top3_probs_log):
            st.write(f"- {disease}: {prob*100:.2f}%")

        # Combine all unique predictions for advice
        all_preds = set(list(top3_preds_xgb) + list(top3_preds_svm) + list(top3_preds_log))

        st.subheader("Disease Info & Precautions")

        unique_preds = list(dict.fromkeys(all_preds))  # Tetap mempertahankan urutan dan menghilangkan duplikat

        for disease in unique_preds:
            cleaned_disease = disease.strip()

            st.markdown(f"### 🦠 {disease}")

            # Cari deskripsi
            desc_row = df_desc[df_desc['Disease'] == cleaned_disease]
            if not desc_row.empty:
                st.markdown(f"**Description:** {desc_row.iloc[0]['Description']}")
            else:
                st.markdown("**Description:** Not available.")

            # Cari precaution
            precaution_row = df_precaution[df_precaution['Disease'] == cleaned_disease]
            if not precaution_row.empty:
                precautions = [
                    precaution_row.iloc[0][f'Precaution_{i}']
                    for i in range(1, 5)
                    if pd.notna(precaution_row.iloc[0][f'Precaution_{i}'])
                ]
                if precautions:
                    st.markdown("**Precautions:**")
                    for p in precautions:
                        st.write(f"- {p}")
                else:
                    st.markdown("**Precautions:** Not available.")
            else:
                st.markdown("**Precautions:** Not available.")
