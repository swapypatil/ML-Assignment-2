from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

REFERENCE_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status',
    'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss',
    'hours.per.week', 'native.country'
]

MODEL_SPECS = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'kNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
}


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
    missing = [col for col in REFERENCE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    df = df[REFERENCE_COLUMNS + ['income']]
    df = df.replace('?', pd.NA)
    target = normalize_target(df['income'])
    features = df.drop(columns=['income'])
    return features, target


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    transformers = []
    if numeric_cols:
        transformers.append(
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_cols)
        )
    if categorical_cols:
        transformers.append(
            (
                'cat',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ]),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder='drop')


def fit_and_evaluate_model(model_name, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    estimator = MODEL_SPECS[model_name]
    pipeline = Pipeline([
        ('preprocess', build_preprocessor(X_train)),
        ('scaler', StandardScaler(with_mean=False)),
        ('model', estimator),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_test, y_pred),
    }
    return metrics, y_test, y_pred


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
st.title('Adult Income Classification Model Performance Dashboard')

with st.sidebar:
    st.header('Controls')
    uploaded_file = st.file_uploader('Upload test CSV', type=['csv'])
    model_name = st.selectbox('Select model', list(MODEL_SPECS.keys()))

if uploaded_file is None:
    st.info('Upload a CSV file containing the same columns as the Adult Census dataset.')
    st.stop()

try:
    X, y = prepare_uploaded_data(uploaded_file)
    metrics, y_true, y_pred = fit_and_evaluate_model(model_name, X, y)
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
fig = plot_confusion_matrix(y_true, y_pred)
st.pyplot(fig)

st.subheader('Prediction Summary')
summary = pd.DataFrame({
    'Actual': y_true,
    'Predicted': y_pred,
})
summary['Actual'] = summary['Actual'].map({0: '<=50K', 1: '>50K'})
summary['Predicted'] = summary['Predicted'].map({0: '<=50K', 1: '>50K'})
st.dataframe(summary.head(20), use_container_width=True)

st.subheader('Uploaded Data Preview')
uploaded_file.seek(0)
preview = pd.read_csv(uploaded_file)
st.dataframe(preview.head(10), use_container_width=True)
