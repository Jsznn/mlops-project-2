import pandas as pd
import joblib
import warnings
import os
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return acc, f1, classification_report(y_test, y_pred)


def log_model_with_mlflow(model_path: str, preprocessed_csv: str, experiment_name: str = "mlops_project_2"):
    if mlflow is None:
        raise RuntimeError("mlflow is not installed in this environment. Please install it (python -m pip install mlflow) and re-run this script.")

    # Load data
    df = pd.read_csv(preprocessed_csv)
    if 'berlangganan_deposito' not in df.columns or 'customer_number' not in df.columns:
        raise ValueError("preprocessed CSV must contain 'berlangganan_deposito' and 'customer_number' columns")

    X = df.drop(columns=['berlangganan_deposito', 'customer_number'])
    y = df['berlangganan_deposito']
    
    config = load_config()
    test_size = config['training']['test_size']
    random_state = config['training']['random_state']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Load model
    model = joblib.load(model_path)

    try:
        estimators = getattr(model, 'estimators_', None)
        if estimators is not None:
            for est in estimators:
                if not hasattr(est, 'monotonic_cst'):
                    setattr(est, 'monotonic_cst', None)
    except Exception:
    
        pass

    # Evaluate
    acc, f1, cls_report = evaluate_model(model, X_test, y_test)
    print("Evaluation results:")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(cls_report)

    # MLflow logging
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        # Log basic metadata
        mlflow.log_param("model_path", model_path)

        # If the model object has the sklearn params we used, log them, otherwise fallback
        try:
            params = model.get_params()
            for k, v in params.items():
                # only log a few params to avoid spam
                if k in [
                    'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'class_weight', 'random_state'
                ]:
                    mlflow.log_param(k, str(v))
        except Exception:
            pass

        # Log metrics
        mlflow.log_metric('accuracy', float(acc))
        mlflow.log_metric('f1_score', float(f1))

        # Log the sklearn model
        # This will save the model under the run's artifacts and make it easy to serve/restore
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = run.info.run_id
        model_artifact_uri = f"runs:/{run_id}/model"
        print(f"Model logged to MLflow run {run_id}")

        # Try registering model to model registry under name 'best_rf' if registry is available
        registered_name = "best_rf"
        try:
            mv = mlflow.register_model(model_artifact_uri, registered_name)
            print(f"Model registered as {registered_name}, version {mv.version}")
        except MlflowException as me:
            # Likely no model registry is configured (local filesystem tracking server doesn't support registry)
            print("Model registry not available or registration failed:", str(me))
            print("Model artifact is still available under the run and can be promoted manually.")
        except Exception as e:
            print("Model registration failed:", str(e))

    print("MLflow logging completed.")


if __name__ == '__main__':
    config = load_config()
    MODEL_PATH = config['paths']['model_rf']
    PREPROCESSED_CSV = config['paths']['preprocessed_data']
    EXPERIMENT_NAME = config['mlflow']['experiment_name']
    
    # Handle paths relative to root if running from src/models
    if not os.path.exists(MODEL_PATH) and os.path.exists(os.path.join('..', '..', MODEL_PATH)):
         MODEL_PATH = os.path.join('..', '..', MODEL_PATH)
         PREPROCESSED_CSV = os.path.join('..', '..', PREPROCESSED_CSV)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run src/models/save_model.py first.")
    if not os.path.exists(PREPROCESSED_CSV):
        raise FileNotFoundError(f"Preprocessed CSV not found at {PREPROCESSED_CSV}. Run src/data/preprocessing.py first.")

    log_model_with_mlflow(MODEL_PATH, PREPROCESSED_CSV, EXPERIMENT_NAME)
