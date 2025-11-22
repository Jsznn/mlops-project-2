import os
import pandas as pd
import joblib
import warnings
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config_loader import load_config

warnings.filterwarnings('ignore')

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.exceptions import MlflowException
except Exception:
    mlflow = None

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


def train_and_log_lr(preprocessed_csv: str, model_path: str, experiment_name: str = "mlops_project_2"):
    if mlflow is None:
        raise RuntimeError("mlflow is not installed in this environment. Please install it and re-run this script.")

    df = pd.read_csv(preprocessed_csv)
    if 'berlangganan_deposito' not in df.columns or 'customer_number' not in df.columns:
        raise ValueError("preprocessed CSV must contain 'berlangganan_deposito' and 'customer_number' columns")

    X = df.drop(columns=['berlangganan_deposito', 'customer_number'])
    y = df['berlangganan_deposito']
    
    config = load_config()
    test_size = config['training']['test_size']
    random_state = config['training']['random_state']
    max_iter = config['training']['max_iter']
    class_weight = config['training']['class_weight']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Train Logistic Regression with class balancing
    lr = LogisticRegression(class_weight=class_weight, max_iter=max_iter, random_state=random_state)
    lr.fit(X_train, y_train)

    # Evaluate
    y_pred = lr.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("LogisticRegression evaluation:")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    # Save locally
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(lr, model_path)
    print(f"Local model saved to {model_path}")

    # Log to MLflow
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_param('model_type', 'LogisticRegression')
        try:
            params = lr.get_params()
            for k in ['C', 'penalty', 'class_weight', 'max_iter', 'random_state']:
                if k in params:
                    mlflow.log_param(k, str(params[k]))
        except Exception:
            pass

        mlflow.log_metric('accuracy', float(acc))
        mlflow.log_metric('f1_score', float(f1))

        mlflow.sklearn.log_model(lr, artifact_path='model')
        run_id = run.info.run_id
        model_artifact_uri = f"runs:/{run_id}/model"
        print(f"Model logged to MLflow run {run_id}")

        # Try register
        registered_name = 'best_lr'
        try:
            mv = mlflow.register_model(model_artifact_uri, registered_name)
            print(f"Model registered as {registered_name}, version {mv.version}")
        except MlflowException as me:
            print("Model registry not available or registration failed:", str(me))
        except Exception as e:
            print("Model registration failed:", str(e))

    print("Logging completed.")


if __name__ == '__main__':
    config = load_config()
    PREPROCESSED_CSV = config['paths']['preprocessed_data']
    MODEL_PATH = config['paths']['model_lr']
    EXPERIMENT_NAME = config['mlflow']['experiment_name']
    
    # Handle paths relative to root if running from src/models
    if not os.path.exists(PREPROCESSED_CSV) and os.path.exists(os.path.join('..', '..', PREPROCESSED_CSV)):
         PREPROCESSED_CSV = os.path.join('..', '..', PREPROCESSED_CSV)
         MODEL_PATH = os.path.join('..', '..', MODEL_PATH)

    if not os.path.exists(PREPROCESSED_CSV):
        raise FileNotFoundError(f"Preprocessed CSV not found at {PREPROCESSED_CSV}. Run src/data/preprocessing.py first.")

    train_and_log_lr(PREPROCESSED_CSV, MODEL_PATH, EXPERIMENT_NAME)
