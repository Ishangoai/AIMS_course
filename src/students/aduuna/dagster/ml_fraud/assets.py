import mlflow
import pandas as pd
from dagster import AssetExecutionContext, asset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

from ..ml.resources import mlflow_resource


@asset(
    description="Load raw fraud detection data from the credit card fraud dataset.",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="fraud_ingest"
)
def fraud_raw_data(context: AssetExecutionContext) -> pd.DataFrame:
    """Load raw fraud detection data from the credit card fraud dataset."""
    mlflow_client = context.resources.mlflow_tracking
    
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

    try:
        # Download the dataset
        context.log.info("Downloading credit card fraud dataset...")
        df = pd.read_csv(url)
        context.log.info(f"Loaded {len(df)} credit card transactions")
        context.log.info(f"Dataset shape: {df.shape}")
        context.log.info(f"Fraud cases: {df['Class'].sum()}")
        context.log.info(f"Normal cases: {len(df) - df['Class'].sum()}")

        # Log dataset info to MLflow
        mlflow_client.log_param("dataset_url", url)
        mlflow_client.log_metric("total_transactions", len(df))
        mlflow_client.log_metric("fraud_cases", int(df['Class'].sum()))
        mlflow_client.log_metric("normal_cases", int(len(df) - df['Class'].sum()))
        mlflow_client.log_metric("fraud_rate", float(df['Class'].mean()))

        return df
    except Exception as e:
        context.log.error(f"Error downloading dataset: {e}")
        # Fallback to a small synthetic dataset
        context.log.info("Using synthetic fallback dataset...")
        data = {
            'V1': [1.0, -1.5, 2.1, 0.8, -0.3] * 20,
            'V2': [0.5, 1.2, -0.8, 1.5, 0.9] * 20,
            'Amount': [100.0, 50.0, 1000.0, 25.0, 200.0] * 20,
            'Class': [0, 0, 1, 0, 0] * 20
        }
        df = pd.DataFrame(data)
        mlflow_client.log_param("dataset_url", "synthetic_fallback")
        mlflow_client.log_metric("total_transactions", len(df))
        return df


@asset(
    description="Process the raw fraud data for model training.",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="fraud_transform"
)
def fraud_processed_data(context: AssetExecutionContext, fraud_raw_data: pd.DataFrame) -> pd.DataFrame:
    """Process the raw fraud data for model training."""
    mlflow_client = context.resources.mlflow_tracking
    df = fraud_raw_data.copy()

    # Basic preprocessing
    context.log.info("Processing fraud detection data...")

    # Handle any missing values
    df = df.dropna()

    # Log basic statistics
    context.log.info(f"Processed dataset shape: {df.shape}")
    if 'Class' in df.columns:
        context.log.info(f"Fraud rate: {df['Class'].mean():.4f}")
        mlflow_client.log_metric("processed_fraud_rate", float(df['Class'].mean()))

    mlflow_client.log_metric("processed_dataset_size", len(df))
    mlflow_client.log_metric("processed_num_features", df.shape[1] - 1)  # -1 for target column

    return df


@asset(
    description="Split data into training (80%) and test (20%) sets.",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="fraud_transform"
)
def fraud_train_test_split(context: AssetExecutionContext, fraud_processed_data: pd.DataFrame) -> dict:
    """Split the fraud data into training and test sets."""
    mlflow_client = context.resources.mlflow_tracking
    
    # Separate features and target
    X = fraud_processed_data.drop('Class', axis=1)
    y = fraud_processed_data['Class']
    
    # Split into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Maintain class distribution
    )
    
    context.log.info(f"Training set: {X_train.shape[0]} samples")
    context.log.info(f"Test set: {X_test.shape[0]} samples")
    context.log.info(f"Training fraud rate: {y_train.mean():.4f}")
    context.log.info(f"Test fraud rate: {y_test.mean():.4f}")
    
    # Log split statistics to MLflow
    mlflow_client.log_param("test_size", 0.2)
    mlflow_client.log_param("random_state", 42)
    mlflow_client.log_metric("train_size", len(X_train))
    mlflow_client.log_metric("test_size", len(X_test))
    mlflow_client.log_metric("train_fraud_rate", float(y_train.mean()))
    mlflow_client.log_metric("test_fraud_rate", float(y_test.mean()))
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


@asset(
    description="Tune RandomForest hyperparameters using 3-fold cross-validation with GridSearch.",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="fraud_model"
)
def fraud_hyperparameter_tuning(context: AssetExecutionContext, fraud_train_test_split: dict) -> dict:
    """Perform hyperparameter tuning using GridSearchCV with MLflow nested runs."""
    mlflow_client = context.resources.mlflow_tracking
    
    X_train = fraud_train_test_split['X_train']
    y_train = fraud_train_test_split['y_train']
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    context.log.info("Starting hyperparameter tuning with 3-fold cross-validation...")
    context.log.info(f"Parameter grid: {param_grid}")

    # Ensure experiment exists
    try:
        experiment = mlflow_client.get_experiment_by_name("fraud_detection_analysis")
        if experiment is None:
            experiment_id = mlflow_client.create_experiment("fraud_detection_analysis")
        else:
            experiment_id = experiment.experiment_id
    except Exception:
        experiment_id = mlflow_client.create_experiment("fraud_detection_analysis")

    # Track individual trials as nested runs
    trial_results = []
    best_score = 0
    best_params = {}
    
    # Manual grid search to log each trial as nested run
    trial_num = 0
    for n_estimators in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            for min_samples_split in param_grid['min_samples_split']:
                for min_samples_leaf in param_grid['min_samples_leaf']:
                    trial_num += 1
                    params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf
                    }
                    
                    run_name = f"trial_{trial_num}_n{n_estimators}_d{max_depth}_split{min_samples_split}_leaf{min_samples_leaf}"
                    
                    with mlflow.start_run(
                        experiment_id=experiment_id,
                        run_name=run_name,
                        nested=True
                    ):
                        try:
                            # Create and evaluate model
                            rf_trial = RandomForestClassifier(random_state=42, **params)
                            cv_scores = cross_val_score(rf_trial, X_train, y_train, cv=3, scoring='roc_auc')
                            mean_score = cv_scores.mean()
                            std_score = cv_scores.std()
                            
                            # Log parameters and metrics
                            mlflow_client.log_params(params)
                            mlflow_client.log_metric("cv_roc_auc_mean", mean_score)
                            mlflow_client.log_metric("cv_roc_auc_std", std_score)
                            
                            context.log.info(f"Trial {trial_num}: {params} -> AUC: {mean_score:.4f} (+/- {std_score:.4f})")
                            
                            trial_results.append({
                                'trial': trial_num,
                                'params': params,
                                'cv_score': mean_score,
                                'cv_std': std_score
                            })
                            
                            if mean_score > best_score:
                                best_score = mean_score
                                best_params = params
                                
                        except Exception as e:
                            context.log.error(f"Trial {trial_num} failed: {e}")
                            mlflow_client.log_param("error", str(e))
    
    context.log.info(f"Hyperparameter tuning completed. Best score: {best_score:.4f}")
    context.log.info(f"Best parameters: {best_params}")
    
    # Log best results to parent run
    mlflow_client.log_params(best_params)
    mlflow_client.log_metric("best_cv_roc_auc", best_score)
    mlflow_client.log_param("total_trials", trial_num)
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'trial_results': trial_results
    }