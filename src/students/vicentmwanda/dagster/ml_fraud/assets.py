from datetime import datetime
from typing import Dict

import dagster as dg
import dagster_slack
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn as ms
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split

from .resources import mlflow_client, mlflow_resource

# Accesing the raw data from the source uri


@dg.asset(
    description="Download the raw data",
    compute_kind="python",
    group_name="ml_fraud_data_ingest"
)
def fraud_ml_raw_data(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    import pandas as pd
    data_url = "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"
    df = pd.read_csv(data_url)
    return dg.MaterializeResult(value={
    "raw_data": df
    },
    metadata={
    "Headings": df.columns.to_list(),
    "Data Preview": str(df.head())
    })


# Splitting raw data into features and targets, and convert into numpy arrays
@dg.asset(
    description="Splitting the raw into features and targets and converting into numpy arrays.",
    compute_kind="python",
    group_name="ml_fraud_data_transform"
)
def fraud_ml_raw_data_split(context: dg.AssetExecutionContext,
fraud_ml_raw_data: Dict
) -> dg.MaterializeResult:

    raw_data: pd.DataFrame = fraud_ml_raw_data['raw_data']

    raw_target_data = raw_data['Class']
    raw_feature_data = raw_data.drop('Class', axis=1)

    target_data = raw_target_data.to_numpy()  # data labels
    feature_data = raw_feature_data.to_numpy()  # features

    return dg.MaterializeResult(value={
    "features": feature_data,
    "targets": target_data,
    "feature_labels": raw_feature_data.columns,
    "target_labels": 'Class'
    },
    metadata={
    'Feature data samples': len(feature_data),
    'Target data samples': len(target_data),
    "Total number of samples": len(raw_data)
    }
    )


# Splitting data in to the training and test sets.
@dg.asset(
    description="Splitting data in to the training and test sets.",
    compute_kind="python",
    group_name="ml_fraud_data_transform"
)
def fraud_ml_train_test_split(context: dg.AssetExecutionContext,
fraud_ml_raw_data_split: Dict
) -> dg.MaterializeResult:

    data = fraud_ml_raw_data_split

    features: pd.DataFrame = data["features"]
    targets: pd.DataFrame = data["targets"]

    x_train, x_test, y_train, y_test = train_test_split(
        features, targets,
        test_size=0.2,       # 20% test data
        random_state=42,     # to ensure results can be repeateed
        stratify=targets          # to a keep targets in the train/test splits
        )

    return dg.MaterializeResult(value={
    "x_train": x_train,
    "y_train": y_train,
    "x_test": x_test,
    "y_test": y_test,
    "feature_labels": data["feature_labels"],
    "target_labels": data["target_labels"],
    },
    )


# function get the experiment for the mlflow
def get_experiment(mlflow_client, name='fraud model experiment'):
    try:
        experiment = mlflow_client.get_experiment_by_name(name)
        if experiment is None:
            experiment = mlflow_client.create_experiment(name)
        experiment_id = experiment.experiment_id

        return experiment, experiment_id

    except Exception:
        experiment = mlflow_client.create_experiment(name)
        experiment_id = experiment.experiment_id

        return experiment, experiment_id


# Training the classifier on the training data with 3 k-fold
@dg.asset(
    description="This asset trains the random forest classifier using 3 k-fold on the training data",
    compute_kind="python",
    group_name="ml_fraud_model",
    resource_defs={"fraud_mlflow_tracking": mlflow_resource}
)
def fraud_ml_model_training(context: dg.AssetExecutionContext,
fraud_ml_train_test_split: Dict
) -> dg.MaterializeResult:

    mlflow_client = context.resources.fraud_mlflow_tracking

    data = fraud_ml_train_test_split

    x_train = data["x_train"]
    y_train = data["y_train"]

    classifier = RandomForestClassifier(n_estimators=200, min_samples_leaf=1, min_samples_split=5,
    max_features='sqrt', random_state=42)

    # parameters for k-fold using on one parameter
    params_grid = {
        'max_depth': [10, 20, 30, 40]  # the hyper parameters to select from during during
    }

    _, experiment_id = get_experiment(mlflow_client)

    with mlflow_client.start_run(experiment_id=experiment_id,
    run_name="3 Fold Cross Validation Training (By Aaron & Vicent)",
    nested=True):
        grid_search = GridSearchCV(
                           estimator=classifier,      # model to tune
                           param_grid=params_grid,    # hyperparameter grid
                           cv=3,                      # 3-fold cross-validation
                           scoring='accuracy',        # metric to evaluate models
                           n_jobs=-1,                 # use all CPU cores
                           verbose=2,                 # print progress details
                           refit=True,                 # retrain best model on full training data
                           return_train_score=True
                           )

        grid_search.fit(x_train, y_train)
        for i, params in enumerate(grid_search.cv_results_['params']):
            with mlflow.start_run(run_name=f"Trial_{i + 1}_with_max_depth_{params['max_depth']}", nested=True):
                mlflow.log_params(params)

                mean_train_score = grid_search.cv_results_['mean_train_score'][i]
                mean_test_score = grid_search.cv_results_['mean_test_score'][i]
                mlflow.log_metric("mean_train_score", mean_train_score)
                mlflow.log_metric("mean_test_score", mean_test_score)

                context.log.info(f'trail {i + 1} {mean_train_score}')

    our_best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    importance = our_best_model.feature_importances_   # type: ignore

    feature_importance = pd.DataFrame({'feature': data['feature_labels'],
        'importance': importance
    }).sort_values(by='importance', ascending=False)

    train_score = grid_search.best_score_

    mlflow_client.log_params(best_params)
    mlflow.log_metric("Best validation score", train_score)

    context.log.info(
    f'''
    Cross validation score: {train_score}
    Best parameters:
    {best_params}
    '''
    )

    return dg.MaterializeResult(value={
    "model": our_best_model,
    "train_score": train_score,
    "best_params": best_params
    },
    metadata={
        "feature_importance": str(feature_importance.head(12))  # type: ignore
    }
    )


def create_confusion_matrix(model, y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    blues = plt.cm.Blues  # type: ignore
    disp.plot(ax=ax, cmap=blues)
    plt.title("Confusion Matrix (Model By Vicent & Aaron)")

    image_path = "/tmp/fraud_model_confusion_matrix.png"
    plt.savefig(image_path)
    plt.close(fig)

    return image_path


# Testing the trained model
@dg.asset(
    description="Testing the trained model on the remaining 20% data",
    compute_kind="python",
    group_name="ml_fraud_model",
    resource_defs={"fraud_mlflow_tracking": mlflow_resource, "fraud_mlflow_client": mlflow_client}
)
def fraud_ml_model_testing(context: dg.AssetExecutionContext,
fraud_ml_model_training: Dict,
fraud_ml_train_test_split: Dict
) -> dg.MaterializeResult:

    mlflow_client = context.resources.fraud_mlflow_tracking

    main_mlflow_client = context.resources.fraud_mlflow_client

    model = fraud_ml_model_training["model"]
    data = fraud_ml_train_test_split

    x_test = data["x_test"]
    y_test = data["y_test"]

    y_pred = model.predict(x_test)

    train_score = fraud_ml_model_training['train_score']

    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    context.log.info(f"test accuracy {test_accuracy}")

    confusion_matrix_path = create_confusion_matrix(model, y_test, y_pred)

    _, experiment_id = get_experiment(mlflow_client)

    with mlflow_client.start_run(experiment_id=experiment_id,
    run_name="Model Testing (By Aaron & Vicent)",
    nested=True):
        mlflow.log_artifact(confusion_matrix_path, artifact_path="confusion_matrix")
        mlflow.log_metric("training accuracy", train_score)
        mlflow_client.log_metric("testing accuracy", test_accuracy)

    feature_names = fraud_ml_train_test_split['feature_labels']
    registered_model_name = "fraud_prediction_model_aaron_vicent"
    with mlflow.start_run(nested=True) as current_run:
        context.log.info(f"Starting nested MLflow run for model logging: {current_run.info.run_id}")

        log_model_info = ms.log_model(
            sk_model=model,
            artifact_path="Fraud Model",
            input_example=pd.DataFrame(x_test[:min(5, len(x_test))], columns=feature_names),
            registered_model_name=registered_model_name
        )
        context.log.info(f"Model logged to MLflow Run ID: {current_run.info.run_id}")
        context.log.info(f"Logged model artifact URI: {log_model_info.model_uri}")

        # use search_model_versions with a proper filter string
        model_versions = mlflow_client.search_model_versions(
            filter_string=f"name='{registered_model_name}'"
        )

        # find the model version registered in this run
        matching_versions = [
            mv for mv in model_versions if mv.run_id == current_run.info.run_id
        ]

        if matching_versions:
            context.log.info("Successfully retrieved registered model version info from registry.")
        else:
            context.log.error(
                f"Could not find registered model version for run ID {current_run.info.run_id} "
                f"and name '{registered_model_name}'."
            )
            raise Exception("Failed to retrieve registered model version details after logging.")
        model_version = matching_versions[0].version
        context.log.info(f"Registered model version: {model_version}")

        # Transition model to Production stage

        main_mlflow_client.transition_model_version_stage(
            name=registered_model_name,
            version=model_version,
            stage="Production",
            archive_existing_versions=True
        )
        return dg.MaterializeResult(value={
        "model": model,
        "accuracy": test_accuracy,
        "recall": test_recall,
        "precision": test_precision,

     },
     )


# Sending slack message
@dg.asset(
    description="Sending slack message",
    compute_kind="python",
    group_name="ml_fraud_slack_msg",
    resource_defs={"slack_messenger": dagster_slack.SlackResource(token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN"))},
)
def fraud_ml_slack_message(context: dg.AssetExecutionContext,
fraud_ml_model_testing: Dict,
) -> dg.MaterializeResult:

    data = fraud_ml_model_testing
    slack: dagster_slack.SlackResource = context.resources.slack_messenger

    test_accuracy = data['accuracy']
    test_precision = data['precision']
    test_recall = data['recall']

    slack.get_client().chat_postMessage(
    channel='aims_course_october2025',
    text=f"""
    *🚀 Model Performance Report*
    _By 👽 Aaron & 🥸 Vicent_

    • 🎯 *Accuracy:* `{test_accuracy:.4f}
    • 🔍 *Recall:* `{test_recall:.4f}
    • 🧠 *Precision:* `{test_precision:.4f}

    🕒 *Timestamp:* {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    """)
    return dg.MaterializeResult(value={
    },
    )
