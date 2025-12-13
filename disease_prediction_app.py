import numpy as np
import pandas as pd
import streamlit as st
import math
import pickle
from catboost import CatBoostClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import openpyxl

st.set_page_config(page_title="Interactive Disease Consultation", layout="centered")

# --- Konfigurasi Konstanta ---
MIN_QUESTIONS_TO_ASK = 3
TOP_N_DISEASES_FOR_SYMPTOM_PRIORITY = 5
PROB_SIMILARITY_THRESHOLD = 0.15
BOOST_FACTOR_TOP_N_SYMPTOMS = 1.8
BOOST_FACTOR_SIMILAR_DISEASES = 2.5
NO_SYMPTOM_THRESHOLD = 0.3
EXPLORATION_QUESTION_INTERVAL = 5
EXPLORATION_BOOST_FACTOR = 1.2
UTI_SPECIFIC_QUESTION_TRIGGER_COUNT = 10
UTI_SPECIFIC_SYMPTOM = 'hip_joint_pain'
UTI_DISEASE_NAME = 'Osteoarthristis'
N_SPLITS_CROSS_VALIDATION = 5
RANDOM_STATE_SPLIT = 42

# --- Load and Preprocess Data ---
@st.cache_data
def load_data():
    try:
        df_train = pd.read_excel('Training_reordered.xlsx')
        df_test = pd.read_excel('Testing_reordered.xlsx')

        df_symptom = pd.read_csv('Symptom-severity.csv')
        diseases = []
        descriptions = []

        with open('disease_description.csv', 'r', encoding='latin1') as f:
            lines = f.readlines()[1:]

        for line in lines:
            line = line.strip().strip('"')
            disease, description = line.split(',"', 1)
            description = description.strip('"')
            diseases.append(disease)
            descriptions.append(description)

        df_desc = pd.DataFrame({
            'Disease': diseases,
            'Description': descriptions
        })
        df_precaution = pd.read_csv('disease_precaution.csv')
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Make sure all required files (Training_reordered.xlsx, Testing_reordered.xlsx, Symptom-severity.csv, disease_description.csv, disease_precaution.csv) are in the same directory as your application.")
        return None, None, None, None, None, None, None, None, None, None, None, None

    df_train = df_train.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df_train = df_train.replace('none', 0).fillna(0)
    df_test = df_test.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df_test = df_test.replace('none', 0).fillna(0)


    df_desc['Disease'] = df_desc['Disease'].str.strip()
    df_precaution['Disease'] = df_precaution['Disease'].str.strip()

    symptom_columns_train = [col for col in df_train.columns if col != 'Disease']
    symptom_columns_test = [col for col in df_test.columns if col != 'Disease']

    all_symptom_columns_list = sorted(list(set(symptom_columns_train) | set(symptom_columns_test)))

    X_train_raw = df_train[symptom_columns_train].astype(int)
    y_train_data = df_train['Disease'].values

    X_test_raw = df_test[symptom_columns_test].astype(int)
    y_test_data = df_test['Disease'].values

    mlb = MultiLabelBinarizer()
    mlb.fit([[s] for s in all_symptom_columns_list])

    X_train_binarized = X_train_raw.values
    X_test_binarized = X_test_raw.values

    all_unique_symptoms_global = mlb.classes_

    disease_symptom_map = {}
    for _, row in df_train.iterrows():
        disease_name = str(row['Disease']).strip()
        related_symptoms = set()
        for col in symptom_columns_train:
            if row[col] == 1:
                related_symptoms.add(col.strip())
        disease_symptom_map[disease_name] = sorted(list(related_symptoms))

    return df_train, df_test, df_symptom, X_train_binarized, y_train_data, X_test_binarized, y_test_data, all_unique_symptoms_global, df_desc, df_precaution, disease_symptom_map, mlb

def prepare_symptom_data(df, symptom_columns):
    disease_symptom_map = {}
    symptom_frequency = {s: 0 for s in symptom_columns}

    disease_counts = df['Disease'].value_counts().to_dict()
    total_samples = len(df)

    disease_prior_probabilities = {disease: count / total_samples for disease, count in disease_counts.items()}

    symptom_given_disease_counts = {disease: {symptom: 0 for symptom in symptom_columns} for disease in disease_counts.keys()}
    symptom_not_given_disease_counts = {disease: {symptom: 0 for symptom in symptom_columns} for disease in disease_counts.keys()}

    for _, row in df.iterrows():
        disease = str(row['Disease']).strip()
        disease_symptom_map.setdefault(disease, set())

        row_symptoms = set()
        for col in symptom_columns:
            if row[col] == 1:
                disease_symptom_map[disease].add(col.strip())
                symptom_frequency[col.strip()] += 1
                row_symptoms.add(col.strip())

        for s in symptom_columns:
            if s in row_symptoms:
                symptom_given_disease_counts[disease][s] += 1
            else:
                symptom_not_given_disease_counts[disease][s] += 1

    for disease in disease_symptom_map:
        disease_symptom_map[disease] = sorted(list(disease_symptom_map[disease]))

    symptom_given_disease_probs = {}
    symptom_not_given_disease_probs = {}

    for disease in disease_counts.keys():
        symptom_given_disease_probs[disease] = {}
        symptom_not_given_disease_probs[disease] = {}

        total_disease_occurrences = disease_counts[disease]

        for symptom in symptom_columns:
            count_s_given_d = symptom_given_disease_counts[disease][symptom]
            count_not_s_given_d = symptom_not_given_disease_counts[disease][symptom]

            symptom_given_disease_probs[disease][symptom] = (count_s_given_d + 1) / (total_disease_occurrences + 2)
            symptom_not_given_disease_probs[disease][symptom] = (count_not_s_given_d + 1) / (total_disease_occurrences + 2)

    most_common_symptoms = sorted(symptom_frequency.items(), key=lambda item: item[1], reverse=True)
    initial_symptoms_to_ask = [s[0] for s in most_common_symptoms[:15]]

    return disease_symptom_map, initial_symptoms_to_ask, symptom_frequency, disease_prior_probabilities, symptom_given_disease_probs, symptom_not_given_disease_probs


def update_possible_diseases_logic(selected_symptoms_set, disease_symptom_map_arg):
    if not selected_symptoms_set:
        return sorted(list(disease_symptom_map_arg.keys()))

    current_possible_diseases = []
    for disease, symptoms_of_disease in disease_symptom_map_arg.items():
        if selected_symptoms_set.issubset(set(symptoms_of_disease)):
            current_possible_diseases.append(disease)

    return sorted(current_possible_diseases)

def calculate_entropy(probabilities):
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def get_next_symptom(current_symptoms_set, answered_symptoms_set, possible_diseases_list, disease_symptom_map_arg, symptom_frequency_map, disease_prior_probabilities, symptom_given_disease_probs, symptom_not_given_disease_probs, question_count_arg, all_unique_symptoms_arg):
    if not possible_diseases_list:
        return None

    if len(current_symptoms_set) == 0 and \
       len(answered_symptoms_set) == UTI_SPECIFIC_QUESTION_TRIGGER_COUNT and \
       UTI_SPECIFIC_SYMPTOM not in answered_symptoms_set: # Fix: Use answered_symptoms_set

        if UTI_DISEASE_NAME in disease_symptom_map_arg and \
           UTI_SPECIFIC_SYMPTOM in disease_symptom_map_arg[UTI_DISEASE_NAME]:
            st.session_state['asking_uti_specific_symptom'] = True
            return UTI_SPECIFIC_SYMPTOM

    current_predicted_probs = {}
    if st.session_state['model_bayes'] and st.session_state['mlb']:
        input_vector_for_ig = np.zeros((1, len(st.session_state['mlb'].classes_)))
        for s in current_symptoms_set:
            try:
                idx = st.session_state['mlb'].classes_.tolist().index(s)
                input_vector_for_ig[0, idx] = 1
            except ValueError:
                pass

        probs_from_ml_model = st.session_state['model_bayes'].predict_proba(input_vector_for_ig)[0]
        disease_names_from_model = st.session_state['model_bayes'].classes_

        for disease, prob in zip(disease_names_from_model, probs_from_ml_model):
            if disease in possible_diseases_list:
                current_predicted_probs[disease] = prob

        total_filtered_prob = sum(current_predicted_probs.values())
        if total_filtered_prob > 0:
            current_predicted_probs = {d: p / total_filtered_prob for d, p in current_predicted_probs.items()}
        else:
            current_predicted_probs = {d: disease_prior_probabilities.get(d, 0) for d in possible_diseases_list}
            total_fallback_prob = sum(current_predicted_probs.values())
            if total_fallback_prob > 0:
                current_predicted_probs = {d: p / total_fallback_prob for d, p in current_predicted_probs.items()}

    else:
        current_predicted_probs = {d: disease_prior_probabilities.get(d, 0) for d in possible_diseases_list}
        total_fallback_prob = sum(current_predicted_probs.values())
        if total_fallback_prob > 0:
            current_predicted_probs = {d: p / total_fallback_prob for d, p in current_predicted_probs.items()}

    sorted_current_predictions = sorted(current_predicted_probs.items(), key=lambda item: item[1], reverse=True)

    top_n_relevant_diseases = []
    for disease, prob in sorted_current_predictions:
        if disease in possible_diseases_list:
            top_n_relevant_diseases.append((disease, prob))
        if len(top_n_relevant_diseases) >= TOP_N_DISEASES_FOR_SYMPTOM_PRIORITY:
            break

    all_relevant_symptoms = set()
    for disease in possible_diseases_list:
        for symptom in disease_symptom_map_arg.get(disease, []):
            if symptom not in answered_symptoms_set:
                all_relevant_symptoms.add(symptom)

    if not all_relevant_symptoms:
        return None

    best_ig_symptom = None
    max_information_gain = -1

    relevant_predicted_probs_for_ig = {d: prob for d, prob in current_predicted_probs.items() if d in possible_diseases_list}
    if not relevant_predicted_probs_for_ig:
        relevant_predicted_probs_for_ig = {d: disease_prior_probabilities.get(d, 0) for d in possible_diseases_list}

    total_initial_prob_for_ig = sum(relevant_predicted_probs_for_ig.values())
    if total_initial_prob_for_ig == 0:
        filtered_candidates = {s: score for s, score in symptom_frequency_map.items() if s in all_relevant_symptoms and s not in answered_symptoms_set}
        if filtered_candidates:
            return max(filtered_candidates, key=lambda s: symptom_frequency_map.get(s, 0))
        return None

    normalized_initial_probs_for_ig = [p / total_initial_prob_for_ig for p in relevant_predicted_probs_for_ig.values()]
    initial_entropy = calculate_entropy(normalized_initial_probs_for_ig)

    diseases_with_similar_probs = set()
    if len(top_n_relevant_diseases) > 1:
        for i in range(min(2, len(top_n_relevant_diseases))):
            for j in range(i + 1, min(2, len(top_n_relevant_diseases))):
                d1, p1 = top_n_relevant_diseases[i]
                d2, p2 = top_n_relevant_diseases[j]
                if max(p1, p2) > 1e-10 and abs(p1 - p2) / max(p1, p2) < PROB_SIMILARITY_THRESHOLD:
                    diseases_with_similar_probs.add(d1)
                    diseases_with_similar_probs.add(d2)

    is_mostly_no_scenario = (len(current_symptoms_set) / max(1, len(answered_symptoms_set))) < NO_SYMPTOM_THRESHOLD

    for symptom in all_relevant_symptoms:
        probs_after_yes = {}
        probs_after_no = {}

        for disease in possible_diseases_list:
            prior_for_ig = current_predicted_probs.get(disease, 0)
            prob_s_given_d = symptom_given_disease_probs.get(disease, {}).get(symptom, 1e-10)
            probs_after_yes[disease] = prior_for_ig * prob_s_given_d

            prob_not_s_given_d = symptom_not_given_disease_probs.get(disease, {}).get(symptom, 1e-10)
            probs_after_no[disease] = prior_for_ig * prob_not_s_given_d

        total_prob_yes_overall = sum(probs_after_yes.values())
        total_prob_no_overall = sum(probs_after_no.values())

        entropy_yes = 0
        if total_prob_yes_overall > 0:
            normalized_probs_yes = [p / total_prob_yes_overall for p in probs_after_yes.values() if p > 0]
            entropy_yes = calculate_entropy(normalized_probs_yes)

        entropy_no = 0
        if total_prob_no_overall > 0:
            normalized_probs_no = [p / total_prob_no_overall for p in probs_after_no.values() if p > 0]
            entropy_no = calculate_entropy(normalized_probs_no)

        current_information_gain = 0
        if total_initial_prob_for_ig > 0:
            current_information_gain = initial_entropy - \
                                       ((total_prob_yes_overall / total_initial_prob_for_ig) * entropy_yes + \
                                        (total_prob_no_overall / total_initial_prob_for_ig) * entropy_no)

        if symptom in [s for d, _ in top_n_relevant_diseases for s in disease_symptom_map_arg.get(d, [])]:
            current_information_gain *= BOOST_FACTOR_TOP_N_SYMPTOMS

        is_discriminative_among_similar = False
        if diseases_with_similar_probs:
            appears_in_similar_diseases = [d for d in diseases_with_similar_probs if symptom in disease_symptom_map_arg.get(d, [])]
            not_appears_in_similar_diseases = [d for d in diseases_with_similar_probs if symptom not in disease_symptom_map_arg.get(d, [])]

            if appears_in_similar_diseases and not_appears_in_similar_diseases:
                is_discriminative_among_similar = True
            elif len(appears_in_similar_diseases) == 1 and len(diseases_with_similar_probs) > 1:
                is_discriminative_among_similar = True

        if is_discriminative_among_similar:
            current_information_gain *= BOOST_FACTOR_SIMILAR_DISEASES

        if is_mostly_no_scenario:
            symptom_global_frequency = symptom_frequency_map.get(symptom, 0)

            if symptom_global_frequency < 50 and any(symptom in disease_symptom_map_arg.get(d, []) for d in possible_diseases_list):
                has_strong_positive_link = False
                for disease in possible_diseases_list:
                    if symptom_given_disease_probs.get(disease, {}).get(symptom, 0) > 0.5:
                        has_strong_positive_link = True
                        break

                has_strong_negative_link_for_some = False
                for disease in possible_diseases_list:
                    if symptom_not_given_disease_probs.get(disease, {}).get(symptom, 1) < 0.2:
                        has_strong_negative_link_for_some = True
                        break

                if has_strong_positive_link or has_strong_negative_link_for_some:
                    current_information_gain *= 1.5

        if (question_count_arg % EXPLORATION_QUESTION_INTERVAL == 0 and question_count_arg > 0) or \
           (max_information_gain < 0.1 and question_count_arg > MIN_QUESTIONS_TO_ASK):

            symptom_global_frequency = symptom_frequency_map.get(symptom, 0)

            if symptom_global_frequency > 1 and symptom_global_frequency < len(all_unique_symptoms_arg) / 5 :
                if symptom not in answered_symptoms_set:
                    current_information_gain *= EXPLORATION_BOOST_FACTOR


        if current_information_gain > max_information_gain:
            max_information_gain = current_information_gain
            best_ig_symptom = symptom

    next_symptom = best_ig_symptom

    if next_symptom is None:
        if is_mostly_no_scenario:
            all_possible_diseases_symptoms = set()
            for disease in possible_diseases_list:
                for s in disease_symptom_map_arg.get(disease, []):
                    if s not in answered_symptoms_set:
                        all_possible_diseases_symptoms.add(s)

            if all_possible_diseases_symptoms:
                fallback_symptom_by_frequency = max(all_possible_diseases_symptoms, key=lambda s: symptom_frequency_map.get(s, 0))
                return fallback_symptom_by_frequency

        for disease, _ in top_n_relevant_diseases:
            for symptom in disease_symptom_map_arg.get(disease, []):
                if symptom not in answered_symptoms_set:
                    return symptom

        filtered_candidates_from_all_symptoms = {s: symptom_frequency_map.get(s, 0) for s in all_unique_symptoms_arg if s not in answered_symptoms_set}
        if filtered_candidates_from_all_symptoms:
            return max(filtered_candidates_from_all_symptoms, key=filtered_candidates_from_all_symptoms.get)

    return next_symptom


def predict_disease_bayes(selected_symptoms, answered_symptoms, model_bayes, mlb, disease_symptom_map_arg):
    if not model_bayes or not mlb:
        return {}

    input_symptoms_vector = np.zeros((1, len(mlb.classes_)))

    for symptom in selected_symptoms:
        try:
            symptom_idx = mlb.classes_.tolist().index(symptom)
            input_symptoms_vector[0, symptom_idx] = 1
        except ValueError:
            pass

    probabilities = model_bayes.predict_proba(input_symptoms_vector)[0]
    disease_names = model_bayes.classes_

    raw_predictions = {disease: prob for disease, prob in zip(disease_names, probabilities)}

    final_filtered_probs = {}
    symptoms_no = answered_symptoms - selected_symptoms

    for disease, prob in raw_predictions.items():
        disease_symptom_set = set(disease_symptom_map_arg.get(disease, []))

        if not selected_symptoms.issubset(disease_symptom_set):
            continue

        is_consistent_with_no_symptoms = True
        for symptom_false in symptoms_no:
            if symptom_false in disease_symptom_set:
                is_consistent_with_no_symptoms = False
                break

        if is_consistent_with_no_symptoms:
            final_filtered_probs[disease] = prob

    total_prob = sum(final_filtered_probs.values())
    if total_prob == 0:
        return {}

    normalized_filtered_probs = {
        disease: prob / total_prob
        for disease, prob in final_filtered_probs.items()
    }

    return normalized_filtered_probs

# --- Data Loading and Model Training ---
df_train_base, df_test_base, df_symptom_unused, X_train_data, y_train_data, X_test_data, y_test_data, all_unique_symptoms_global, df_desc, df_precaution, disease_symptom_map_global, mlb_global = load_data()


# Inisialisasi model Scikit-learn
@st.cache_resource
def train_scikit_learn_model(X_train, y_train, X_test, y_test, n_splits=N_SPLITS_CROSS_VALIDATION):
    model = CatBoostClassifier(
        iterations=50,
        learning_rate=0.1,
        depth=2,
        loss_function='MultiClass',
        verbose=False,
        random_seed=RANDOM_STATE_SPLIT
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE_SPLIT)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        fold_model = CatBoostClassifier(
            iterations=50,
            learning_rate=0.1,
            depth=2,
            loss_function='MultiClass',
            verbose=False,
            random_seed=RANDOM_STATE_SPLIT + fold
        )
        fold_model.fit(X_train_fold, y_train_fold)
        y_pred_fold = fold_model.predict(X_val_fold)
        y_pred_fold = y_pred_fold.flatten()

        accuracies.append(accuracy_score(y_val_fold, y_pred_fold))
        precisions.append(precision_score(y_val_fold, y_pred_fold, average='weighted', zero_division=0))
        recalls.append(recall_score(y_val_fold, y_pred_fold, average='weighted', zero_division=0))
        f1_scores.append(f1_score(y_val_fold, y_pred_fold, average='weighted', zero_division=0))

    final_model = CatBoostClassifier(
        iterations=50,
        learning_rate=0.1,
        depth=2,
        loss_function='MultiClass',
        verbose=False,
        random_seed=RANDOM_STATE_SPLIT
    )
    final_model.fit(X_train, y_train)

    # --- ADDED CODE TO SAVE THE MODEL AND MLB ---
    try:
        with open('catboost_model.pkl', 'wb') as f:
            pickle.dump(final_model, f)
        st.sidebar.success("CatBoost model saved as catboost_model.pkl")

        # Access mlb_global from the global scope if it's available
        # It's better practice to pass mlb_global as an argument if possible
        # or load it within this function if it's not a dependency of load_data's return.
        # For this setup, we assume mlb_global is accessible here.
        if 'mlb_global' in globals() and globals()['mlb_global'] is not None:
            with open('mlb_binarizer.pkl', 'wb') as f:
                pickle.dump(globals()['mlb_global'], f)
            st.sidebar.success("MultiLabelBinarizer saved as mlb_binarizer.pkl")
        else:
            st.sidebar.warning("MultiLabelBinarizer not available for saving.")

    except Exception as e:
        st.sidebar.error(f"Error saving model or binarizer: {e}")
    # --- END OF ADDED CODE ---

    y_pred_test = final_model.predict(X_test)
    y_pred_test = y_pred_test.flatten()
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    test_class_report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)


    metrics = {
        "cross_validation_accuracies": accuracies,
        "cross_validation_precision_scores": precisions,
        "cross_validation_recall_scores": recalls,
        "cross_validation_f1_scores": f1_scores,
        "average_accuracy_cv": np.mean(accuracies),
        "average_precision_cv": np.mean(precisions),
        "average_recall_cv": np.mean(recalls),
        "average_f1_cv": np.mean(f1_scores),
        # Metrik test set tetap disimpan di objek metrics, hanya tidak ditampilkan di UI
        "test_set_accuracy": test_accuracy,
        "test_set_precision": test_precision,
        "test_set_recall": test_recall,
        "test_set_f1_score": test_f1,
        "test_set_classification_report": test_class_report
    }

    return final_model, metrics

if df_train_base is None or all_unique_symptoms_global is None:
    data_loaded_successfully = False
    disease_symptom_map = {}
    initial_symptoms_to_ask = []
    symptom_frequency_map = {}
    disease_prior_probabilities = {}
    symptom_given_disease_probs = {}
    symptom_not_given_disease_probs = {}
    model_bayes = None
    mlb = None
    model_metrics = None
else:
    data_loaded_successfully = True
    model_bayes, model_metrics = train_scikit_learn_model(X_train_data, y_train_data, X_test_data, y_test_data)
    mlb = mlb_global

    symptom_cols_for_prepare = [col for col in df_train_base.columns if col != 'Disease']
    _ , _, symptom_frequency_map, disease_prior_probabilities, symptom_given_disease_probs, symptom_not_given_disease_probs = prepare_symptom_data(df_train_base, symptom_cols_for_prepare)


# --- Application State (Using st.session_state for Streamlit) ---
if 'stage' not in st.session_state:
    st.session_state['stage'] = 'start'
if 'current_symptoms' not in st.session_state:
    st.session_state['current_symptoms'] = set()
if 'answered_symptoms' not in st.session_state:
    st.session_state['answered_symptoms'] = set()
if 'possible_diseases' not in st.session_state:
    st.session_state['possible_diseases'] = []
if 'asked_symptoms_step_by_step' not in st.session_state:
    st.session_state['asked_symptoms_step_by_step'] = set()
if 'next_symptom_to_ask_step_by_step' not in st.session_state:
    st.session_state['next_symptom_to_ask_step_by_step'] = None
if 'question_count' not in st.session_state:
    st.session_state['question_count'] = 0
if 'asking_uti_specific_symptom' not in st.session_state:
    st.session_state['asking_uti_specific_symptom'] = False
if 'answered_uti_symptom_no' not in st.session_state:
    st.session_state['answered_uti_symptom_no'] = False

st.session_state['model_bayes'] = model_bayes
st.session_state['mlb'] = mlb
st.session_state['disease_symptom_map'] = disease_symptom_map_global
st.session_state['model_metrics'] = model_metrics

# --- UI Functions (Modified for Streamlit) ---
def reset_consultation_state():
    st.session_state['stage'] = 'start'
    st.session_state['current_symptoms'] = set()
    st.session_state['answered_symptoms'] = set()
    st.session_state['possible_diseases'] = []
    st.session_state['asked_symptoms_step_by_step'] = set()
    st.session_state['next_symptom_to_ask_step_by_step'] = None
    st.session_state['question_count'] = 0
    st.session_state['asking_uti_specific_symptom'] = False
    st.session_state['answered_uti_symptom_no'] = False

def handle_yes_click():
    if st.session_state['next_symptom_to_ask_step_by_step']:
        st.session_state['current_symptoms'].add(st.session_state['next_symptom_to_ask_step_by_step'])
        st.session_state['answered_symptoms'].add(st.session_state['next_symptom_to_ask_step_by_step'])
        st.session_state['question_count'] += 1
        if st.session_state['next_symptom_to_ask_step_by_step'] == UTI_SPECIFIC_SYMPTOM:
            st.session_state['asking_uti_specific_symptom'] = False
            st.session_state['answered_uti_symptom_no'] = False
    st.rerun()

def handle_no_click():
    if st.session_state['next_symptom_to_ask_step_by_step']:
        st.session_state['answered_symptoms'].add(st.session_state['next_symptom_to_ask_step_by_step'])
        st.session_state['question_count'] += 1
        if st.session_state['next_symptom_to_ask_step_by_step'] == UTI_SPECIFIC_SYMPTOM:
            st.session_state['asking_uti_specific_symptom'] = False
            st.session_state['answered_uti_symptom_no'] = True
    st.rerun()

def handle_predict_click():
    if not st.session_state['current_symptoms'] and not st.session_state['answered_symptoms']:
        st.warning("Please answer at least one symptom to get a more accurate prediction.")
        return
    st.session_state['stage'] = 'predict'
    st.rerun()

# --- Main Streamlit Application Flow ---

st.markdown("""
<style>
.main-header {
    font-size: 3em;
    color: #1E90FF; /* DodgerBlue */
    text-align: center;
    margin-bottom: 30px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}
.subheader {
    font-size: 1.8em;
    color: #363636;
    margin-top: 25px;
    margin-bottom: 15px;
    border-bottom: 2px solid #f0f2f6;
    padding-bottom: 5px;
}
.stButton>button {
    width: 100%;
    border-radius: 8px;
    font-size: 1.1em;
    padding: 10px 20px;
}
.stButton>button.primary {
    background-color: #4CAF50; /* Green */
    color: white;
    border: none;
}
.stButton>button.secondary {
    background-color: #008CBA; /* Blue */
    color: white;
    border: none;
}
.stButton>button:hover {
    opacity: 0.8;
}
.stWarning, .stInfo, .stSuccess {
    border-radius: 8px;
    padding: 15px;
    margin-top: 15px;
    margin-bottom: 15px;
}
.symptom-question {
    background-color: #e6f7ff; /* Light blue */
    padding: 20px;
    border-radius: 10px;
    border-left: 8px solid #1890ff; /* Darker blue */
    margin-bottom: 20px;
    font-size: 1.5em;
    font-weight: bold;
    color: #333;
}
.symptom-list-yes {
    background-color: #e8f5e9; /* Light green */
    padding: 10px;
    border-radius: 5px;
    border-left: 5px solid #4CAF50;
    margin-bottom: 10px;
}
.symptom-list-no {
    background-color: #ffebee; /* Light red */
    padding: 10px;
    border-radius: 5px;
    border-left: 5px solid #F44336;
    margin-bottom: 10px;
}
.possible-diseases-info {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
    margin-top: 15px;
    font-style: italic;
    color: #555;
}
.prediction-card {
    background-color: #f9f9f9;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
/* IMPORTANT: Added !important to ensure override */
.disease-name-prediction {
    font-size: 2em !important;
    color: #D32F2F; /* Dark Red */
    font-weight: bold;
    margin-bottom: 10px;
    text-align: center;
}
.info-section-header {
    font-size: 1.5em;
    color: #4A4A4A;
    margin-top: 30px;
    margin-bottom: 15px;
    border-bottom: 1px solid #ddd;
    padding-bottom: 5px;
}
.detail-disease-card {
    background-color: #e0f2f7;
    padding: 15px;
    border-radius: 8px;
    border-left: 5px solid #0288d1;
    margin-bottom: 20px;
    box-shadow: 1px 1px 5px rgba(0,0,0,0.08);
}
.detail-disease-card h4 {
    color: #0288d1;
    margin-top: 0;
    margin-bottom: 10px;
    font-size: 1.4em;
}
.detail-disease-card ul {
    list-style-type: disc;
    margin-left: 20px;
    padding-left: 0;
}
.detail-disease-card li {
    margin-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)


st.markdown("<h1 class='main-header'>🩺 Interactive Disease Consultation</h1>", unsafe_allow_html=True)

if not data_loaded_successfully:
    st.error("Error! Application cannot run because required data files were not found. Please ensure all CSV and Excel files are in the same directory.")
    st.stop()

# --- Tampilan Hasil Validasi Model ---
st.sidebar.markdown("## Model Validation Metrics")
if st.session_state['model_metrics']:
    metrics = st.session_state['model_metrics']
    st.sidebar.markdown("### Cross-Validation Results (on Training Data):")
    st.sidebar.write(f"**Average Accuracy (CV):** {metrics['average_accuracy_cv']:.4f}")
    st.sidebar.write(f"**Average Precision (CV):** {metrics['average_precision_cv']:.4f}")
    st.sidebar.write(f"**Average Recall (CV):** {metrics['average_recall_cv']:.4f}")
    st.sidebar.write(f"**Average F1-Score (CV):** {metrics['average_f1_cv']:.4f}")

    with st.sidebar.expander("Per-Fold CV Scores"):
        for i in range(len(metrics['cross_validation_accuracies'])):
            st.write(f"**Fold {i+1}:**")
            st.write(f"  Accuracy: {metrics['cross_validation_accuracies'][i]:.4f}")
            st.write(f"  Precision: {metrics['cross_validation_precision_scores'][i]:.4f}")
            st.write(f"  Recall: {metrics['cross_validation_recall_scores'][i]:.4f}")
            st.write(f"  F1-Score: {metrics['cross_validation_f1_scores'][i]:.4f}")

    # --- Bagian yang dihapus ---
    # st.sidebar.markdown("### Test Set Results (on Unseen Data):")
    # st.sidebar.write(f"**Accuracy (Test Set):** {metrics['test_set_accuracy']:.4f}")
    # st.sidebar.write(f"**Precision (Test Set):** {metrics['test_set_precision']:.4f}")
    # st.sidebar.write(f"**Recall (Test Set):** {metrics['test_set_recall']:.4f}")
    # st.sidebar.write(f"**F1-Score (Test Set):** {metrics['test_set_f1_score']:.4f}")

    # with st.sidebar.expander("Detailed Classification Report (Test Set)"):
    #     report_df = pd.DataFrame(metrics['test_set_classification_report']).transpose()
    #     st.dataframe(report_df)
    # --- Akhir Bagian yang dihapus ---
else:
    st.sidebar.info("Model metrics not available.")

if st.session_state['stage'] == 'start':
    st.markdown("<p>Welcome to the Interactive Disease Consultation. We will help you identify possible diseases based on the symptoms you are experiencing.</p>", unsafe_allow_html=True)
    st.write("Click the button below to start your consultation journey!")
    if st.button("Start Consultation", type="primary"):
        reset_consultation_state()
        st.session_state['stage'] = 'step_by_step_questions'
        st.session_state['next_symptom_to_ask_step_by_step'] = get_next_symptom(
            st.session_state['current_symptoms'],
            st.session_state['answered_symptoms'],
            sorted(list(st.session_state['disease_symptom_map'].keys())),
            st.session_state['disease_symptom_map'],
            symptom_frequency_map,
            disease_prior_probabilities,
            symptom_given_disease_probs,
            symptom_not_given_disease_probs,
            st.session_state['question_count'],
            all_unique_symptoms_global
        )
        st.rerun()

elif st.session_state['stage'] == 'step_by_step_questions':
    st.markdown("<h2 class='subheader'>1. Step-by-Step Symptom Questions</h2>", unsafe_allow_html=True)
    st.markdown("<p>In this mode, we will ask symptoms one by one to help narrow down possible diseases. Select 'Yes' if you are experiencing the symptom, or 'No' if not.</p>", unsafe_allow_html=True)

    current_possible_diseases = sorted(list(st.session_state['disease_symptom_map'].keys()))

    temp_possible = []
    for disease in current_possible_diseases:
        if st.session_state['current_symptoms'].issubset(set(st.session_state['disease_symptom_map'].get(disease, []))):
            temp_possible.append(disease)
    current_possible_diseases = temp_possible

    for symptom_no in (st.session_state['answered_symptoms'] - st.session_state['current_symptoms']):
        temp_possible_after_no = []
        for disease in current_possible_diseases:
            if symptom_no not in set(st.session_state['disease_symptom_map'].get(disease, [])):
                temp_possible_after_no.append(disease)
        current_possible_diseases = temp_possible_after_no

    st.session_state['possible_diseases'] = current_possible_diseases

    if st.session_state['answered_uti_symptom_no'] and not st.session_state['current_symptoms']:
        st.error("Based on your responses, especially indicating no related symptoms, we are currently unable to provide a specific diagnosis. Please consult a healthcare professional for a comprehensive assessment.")
        if st.button("Start New Consultation", type="secondary"):
            reset_consultation_state()
            st.rerun()
        st.stop()

    if not st.session_state['possible_diseases'] and len(st.session_state['answered_symptoms']) > 0:
        st.warning("Sorry, based on the symptoms you provided, we could not find a matching disease in our database. Please check your symptoms again or try another consultation.")
        if st.button("New Consultation", type="secondary"):
            reset_consultation_state()
            st.rerun()
        st.stop()

    st.session_state['next_symptom_to_ask_step_by_step'] = get_next_symptom(
        st.session_state['current_symptoms'],
        st.session_state['answered_symptoms'],
        st.session_state['possible_diseases'],
        st.session_state['disease_symptom_map'],
        symptom_frequency_map,
        disease_prior_probabilities,
        symptom_given_disease_probs,
        symptom_not_given_disease_probs,
        st.session_state['question_count'],
        all_unique_symptoms_global
    )

    prediction_trigger = False

    if st.session_state['asking_uti_specific_symptom']:
        pass
    elif len(st.session_state['possible_diseases']) == 1:
        prediction_trigger = True
    elif st.session_state['question_count'] >= MIN_QUESTIONS_TO_ASK:
        current_predicted_probs_for_check = predict_disease_bayes(
            st.session_state['current_symptoms'],
            st.session_state['answered_symptoms'],
            st.session_state['model_bayes'],
            st.session_state['mlb'],
            st.session_state['disease_symptom_map']
        )
        sorted_current_predictions_for_check = sorted(current_predicted_probs_for_check.items(), key=lambda item: item[1], reverse=True)
        top_n_diseases_for_check = [d[0] for d in sorted_current_predictions_for_check[:TOP_N_DISEASES_FOR_SYMPTOM_PRIORITY]]

        has_unanswered_symptoms_in_top_n = False
        for disease in top_n_diseases_for_check:
            for symptom in st.session_state['disease_symptom_map'].get(disease, []):
                if symptom not in st.session_state['answered_symptoms']:
                    has_unanswered_symptoms_in_top_n = True
                    break
            if has_unanswered_symptoms_in_top_n:
                break

        if not has_unanswered_symptoms_in_top_n and st.session_state['next_symptom_to_ask_step_by_step'] is None:
            prediction_trigger = True
    elif len(st.session_state['possible_diseases']) <= 2 and len(st.session_state['current_symptoms']) > 0 and st.session_state['next_symptom_to_ask_step_by_step'] is None:
        prediction_trigger = True
    elif not st.session_state['current_symptoms'] and len(st.session_state['answered_symptoms']) > 0 and st.session_state['next_symptom_to_ask_step_by_step'] is None:
        prediction_trigger = True
    elif st.session_state['next_symptom_to_ask_step_by_step'] is None and st.session_state['question_count'] == 0:
        st.warning("No symptoms to ask. The database might be empty or all symptoms have been answered. Or your database contains no common symptoms.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Get Prediction", type="secondary", on_click=handle_predict_click):
                pass
        with col2:
            if st.button("New Consultation", type="secondary", on_click=reset_consultation_state):
                st.rerun()
        st.stop()

    if prediction_trigger:
        st.session_state['stage'] = 'predict'
        st.rerun()
    else:
        if st.session_state['next_symptom_to_ask_step_by_step']:
            st.markdown(f"<div class='symptom-question'>Are you experiencing '{st.session_state['next_symptom_to_ask_step_by_step']}'?</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes", type="primary", on_click=handle_yes_click):
                    st.rerun()
            with col2:
                if st.button("No", type="primary", on_click=handle_no_click):
                    st.rerun()
        else:
            st.info("No more differentiating symptoms to ask based on your current answers. Please click 'Get Prediction'.")

        st.markdown("---")

        if st.session_state['current_symptoms']:
            st.markdown(f"<div class='symptom-list-yes'><b>Symptoms you have confirmed 'Yes':</b> {', '.join(sorted(list(st.session_state['current_symptoms'])))} </div>", unsafe_allow_html=True)
        else:
            st.info("No symptoms confirmed 'Yes' yet.")

        symptoms_no = sorted(list(st.session_state['answered_symptoms'] - st.session_state['current_symptoms']))
        if symptoms_no:
            st.markdown(f"<div class='symptom-list-no'><b>Symptoms you have confirmed 'No':</b> {', '.join(symptoms_no)}</div>", unsafe_allow_html=True)
        else:
            st.info("No symptoms confirmed 'No' yet.")

        st.markdown(f"<div class='possible-diseases-info'><i>Still possible diseases ({len(st.session_state['possible_diseases'])}): {', '.join(st.session_state['possible_diseases'])}</i></div>", unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Get Prediction", type="secondary", on_click=handle_predict_click):
                pass
        with col2:
            if st.button("New Consultation", type="secondary", on_click=reset_consultation_state):
                st.rerun()

elif st.session_state['stage'] == 'predict':
    st.markdown("<h2 class='subheader'>2. Prediction Results</h2>", unsafe_allow_html=True)

    if not st.session_state['current_symptoms'] and not st.session_state['answered_symptoms']:
        st.warning("You have not selected any symptoms. Please start a new consultation.")
        if st.button("New Consultation", type="secondary", on_click=reset_consultation_state):
            st.rerun()
        st.stop()

    if st.session_state['answered_uti_symptom_no'] and not st.session_state['current_symptoms']:
        st.error("Based on your responses, especially indicating no related symptoms, we are currently unable to provide a specific diagnosis. Please consult a healthcare professional for a comprehensive assessment.")
        if st.button("Start New Consultation", type="secondary"):
            reset_consultation_state()
            st.rerun()
        st.stop()

    predicted_probs = predict_disease_bayes(
        st.session_state['current_symptoms'],
        st.session_state['answered_symptoms'],
        st.session_state['model_bayes'],
        st.session_state['mlb'],
        st.session_state['disease_symptom_map']
    )

    if not predicted_probs:
        st.warning("No matching diseases found with all the symptoms you provided. There might be a symptom combination not covered in the database or too specific.")
    else:
        sorted_predictions = sorted(predicted_probs.items(), key=lambda item: item[1], reverse=True)

        st.markdown("<h3 class='info-section-header'>Disease Prediction</h3>", unsafe_allow_html=True)
        for disease, prob in sorted_predictions[:min(3, len(sorted_predictions))]:
            with st.container():
                st.markdown(f"""
                <div class='prediction-card'>
                    <span class='disease-name-prediction'>{disease}</span>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("<h3 class='info-section-header'>Symptom Summary</h3>", unsafe_allow_html=True)
        if st.session_state['current_symptoms']:
            st.markdown(f"<div class='symptom-list-yes'><b>Symptoms you are experiencing:</b> {', '.join(sorted(list(st.session_state['current_symptoms'])))}</div>", unsafe_allow_html=True)
        else:
            st.info("You have not confirmed any 'Yes' symptoms.")

        symptoms_no = sorted(list(st.session_state['answered_symptoms'] - st.session_state['current_symptoms']))
        if symptoms_no:
            st.markdown(f"<div class='symptom-list-no'><b>Symptoms you are NOT experiencing:</b> {', '.join(symptoms_no)}</div>", unsafe_allow_html=True)
        else:
            st.info("No symptoms confirmed 'No' yet.")
        st.markdown("---")

        st.markdown("<h3 class='info-section-header'>Predicted Disease Symptom Details</h3>", unsafe_allow_html=True)
        diseases_to_show_symptoms = [pred[0] for pred in sorted_predictions[:min(3, len(sorted_predictions))]]

        for disease in diseases_to_show_symptoms:
            cleaned_disease = disease.strip()
            disease_symptoms = st.session_state['disease_symptom_map'].get(cleaned_disease, [])

            symptoms_display = []
            for s in disease_symptoms:
                if s in st.session_state['current_symptoms']:
                    symptoms_display.append(f"<span style='color:green;'><b>{s} (YES)</b></span>")
                elif s in symptoms_no:
                    symptoms_display.append(f"<span style='color:red;'><b>{s} (NO)</b></span>")
                else:
                    symptoms_display.append(s)

            st.markdown(f"""
            <div class='detail-disease-card'>
                <h4>🧬 Symptoms related to: {cleaned_disease}</h4>
                <ul>{"".join([f"<li>{s_display}</li>" for s_display in symptoms_display])}</ul>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")

        st.markdown("<h3 class='info-section-header'>Disease Information & Precautions</h3>", unsafe_allow_html=True)

        diseases_for_info = [pred[0] for pred in sorted_predictions[:min(3, len(sorted_predictions))]]

        for disease in diseases_for_info:
            cleaned_disease = disease.strip()
            desc_row = df_desc[df_desc['Disease'] == cleaned_disease]
            precaution_row = df_precaution[df_precaution['Disease'] == cleaned_disease]

            desc_html = f"<b>Description:</b> {desc_row.iloc[0]['Description']}" if not desc_row.empty else "<b>Description:</b> Not available."

            precautions_html = ""
            if not precaution_row.empty:
                precautions_list = [
                    precaution_row.iloc[0][f'Precaution_{i}']
                    for i in range(1, 5)
                    if pd.notna(precaution_row.iloc[0][f'Precaution_{i}']) and str(precaution_row.iloc[0][f'Precaution_{i}']).strip() != 'none'
                ]
                if precautions_list:
                    precautions_html = "<b>Precautions:</b><ul>" + "".join([f"<li>{p}</li>" for p in precautions_list]) + "</ul>"
                else:
                    precautions_html = "<b>Precautions:</b> Not available."
            else:
                precautions_html = "<b>Precautions:</b> Not available."

            st.markdown(f"""
            <div class='detail-disease-card'>
                <h4>🦠 {disease}</h4>
                {desc_html}
                {precautions_html}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")

    st.markdown("---")
    st.button("New Consultation", type="secondary", on_click=reset_consultation_state)
