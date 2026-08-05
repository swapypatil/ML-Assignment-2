from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).with_name('Decision_Tree.joblib')


def load_model():
    return joblib.load(MODEL_PATH)


def predict(X):
    model = load_model()
    return model.predict(X)


def predict_proba(X):
    model = load_model()
    return model.predict_proba(X)


if __name__ == '__main__':
    print(f'Loaded model from: {MODEL_PATH}')
