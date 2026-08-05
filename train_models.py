from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
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

DATA_PATH = Path(__file__).resolve().parent / 'test_data.csv'
MODEL_DIR = Path(__file__).resolve().parent / 'model'
RESULTS_PATH = Path(__file__).resolve().parent / 'model_metrics.csv'


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.replace('?', pd.NA)
    X = df.drop(columns=['income'])
    y = (df['income'] == '>50K').astype(int)
    return X, y


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                'num',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                ]),
                numeric_cols,
            )
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


def evaluate_model(model_name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_test, y_pred),
    }
    return metrics


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model_specs = {
        'Logistic_Regression': LogisticRegression(max_iter=500, random_state=42),
        'Decision_Tree': DecisionTreeClassifier(random_state=42),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'Naive_Bayes': GaussianNB(),
        'Random_Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = []

    for name, estimator in model_specs.items():
        preprocessor = build_preprocessor(X_train)
        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('scaler', StandardScaler(with_mean=False)),
            ('model', estimator),
        ])

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        results.append(metrics)

        model_path = MODEL_DIR / f'{name}.joblib'
        joblib.dump(pipeline, model_path)
        print(f'\n{name}:')
        for key, val in metrics.items():
            if key != 'Model':
                print(f'  {key}: {val:.4f}')

    results_df = pd.DataFrame(results)
    results_df = results_df[['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]
    results_df.to_csv(RESULTS_PATH, index=False)

    best_model = results_df.sort_values(by='F1', ascending=False).iloc[0]
    print('\nBest model by F1:', best_model['Model'])
    print('Saved metrics to:', RESULTS_PATH)
    print('Saved models to:', MODEL_DIR)


if __name__ == '__main__':
    main()
