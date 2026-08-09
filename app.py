from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA_PATH = Path(__file__).resolve().parent / 'Adult Census Income.csv'
MODEL_DIR = Path(__file__).resolve().parent / 'model'
REFERENCE_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status',
    'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss',
    'hours.per.week', 'native.country'
]

MODEL_FILES = {
    'Logistic Regression': MODEL_DIR / 'Logistic_Regression.joblib',
    'Decision Tree': MODEL_DIR / 'Decision_Tree.joblib',
    'kNN': MODEL_DIR / 'kNN.joblib',
    'Naive Bayes': MODEL_DIR / 'Naive_Bayes.joblib',
    'Random Forest': MODEL_DIR / 'Random_Forest.joblib',
}


def load_reference_columns():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        return list(df.drop(columns=['income']).columns)
    return REFERENCE_COLUMNS


def normalize_target(series):
    s = series.astype(str).str.strip()
    mapping = {
        '0': 0,
        '1': 1,
        '<=50k': 0,
        '<=50K': 0,
        '>50k': 1,
        '>50K': 1,
        '0.0': 0,
        '1.0': 1,
    }
    converted = s.map(mapping)
    if converted.isna().any():
        unique = sorted(s.unique().tolist())
        raise ValueError(f"Unsupported target values in uploaded data. Found: {unique}")
    return converted.astype(int)


def prepare_uploaded_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    required_cols = load_reference_columns()
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    df = df[required_cols + ['income']]
    df = df.replace('?', pd.NA)
    target = normalize_target(df['income'])
    features = df.drop(columns=['income'])
    return features, target


def compute_metrics(y_true, y_pred, y_prob):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_prob),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(['<=50K', '>50K'])
    ax.set_yticklabels(['<=50K', '>50K'])
    plt.tight_layout()
    return fig


st.set_page_config(page_title='Adult Income Model Explorer', layout='wide')
st.title('Adult Income Classification Model PerformanceDashboard')

with st.sidebar:
    st.header('Controls')
    uploaded_file = st.file_uploader('Upload test CSV', type=['csv'])
    model_name = st.selectbox('Select model', list(MODEL_FILES.keys()))

if uploaded_file is None:
    st.info('Upload a CSV file containing the same columns as the Adult Census dataset. A sample file is available in the project folder.')
    st.stop()

try:
    X_test, y_test = prepare_uploaded_data(uploaded_file)
    model = joblib.load(MODEL_FILES[model_name])
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)
except Exception as exc:
    st.error(f'Error while processing uploaded data: {exc}')
    st.stop()

st.subheader(f'Model: {model_name}')

col1, col2, col3, col4, col5, col6 = st.columns(6)
metric_cols = [
    ('Accuracy', metrics['Accuracy']),
    ('AUC', metrics['AUC']),
    ('Precision', metrics['Precision']),
    ('Recall', metrics['Recall']),
    ('F1', metrics['F1']),
    ('MCC', metrics['MCC']),
]
for col, (label, value) in zip([col1, col2, col3, col4, col5, col6], metric_cols):
    col.metric(label, f'{value:.4f}')

st.subheader('Confusion Matrix')
fig = plot_confusion_matrix(y_test, y_pred)
st.pyplot(fig)

st.subheader('Prediction Summary')
summary = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
})
summary['Actual'] = summary['Actual'].map({0: '<=50K', 1: '>50K'})
summary['Predicted'] = summary['Predicted'].map({0: '<=50K', 1: '>50K'})
st.dataframe(summary.head(20), use_container_width=True)

st.subheader('Uploaded Data Preview')
uploaded_file.seek(0)
preview = pd.read_csv(uploaded_file)
st.dataframe(preview.head(10), use_container_width=True)
