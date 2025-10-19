import os

BASE_DIR = os.path.abspath(os.getenv("DAGSTER_HOME", "."))
DB_FILE = "mlflow_local_tracking.db"
DB_PATH = os.path.join(BASE_DIR, DB_FILE)
TRACKING_URI = f"sqlite:///{DB_PATH}"
EXPERIMENT_NAME = "ml_fraud_detection"
REGISTERED_MODEL_NAME = "tuned-fraud-detector"
