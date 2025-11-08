import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

warnings.filterwarnings('ignore')


def train_and_save_model(preprocessed_csv_path: str, model_path: str):
    """Train RandomForest with the notebook hyperparameters and save it.

    The script mirrors the notebook: split (25% test, random_state=23, stratify),
    train best_rf on the training split, then save the trained model.
    """
    print(f"Loading preprocessed data from: {preprocessed_csv_path}")
    df = pd.read_csv(preprocessed_csv_path)

    # Ensure required columns exist
    for c in ['berlangganan_deposito', 'customer_number']:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in {preprocessed_csv_path}")

    X = df.drop(columns=['berlangganan_deposito', 'customer_number'])
    y = df['berlangganan_deposito']

    print("Splitting data (train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=23, stratify=y
    )

    print("Training RandomForest (best_rf) with notebook hyperparameters...")
    best_rf = RandomForestClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=1,
        n_estimators=100,
        class_weight={0: 1, 1: 4},
        random_state=23,
    )

    best_rf.fit(X_train, y_train)

    print(f"Saving model to: {model_path}")
    joblib.dump(best_rf, model_path)
    print("Model saved successfully.")


if __name__ == '__main__':
    train_and_save_model('data/preprocessed.csv', 'models/best_rf.joblib')