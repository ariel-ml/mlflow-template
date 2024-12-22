import mlflow
from dotenv import load_dotenv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from libs.mlflow import mlflow_helper
from libs import logger

LOG = logger.getLogger(__name__)

load_dotenv()

mlflow_helper.setup("Nested Experiment", system_metrics=True)


@mlflow.trace(name="trace/classification")
def train():
    # Set the name of the MLflow Experiment
    """
    MLFlow Nested Runs Example

    Train a logistic regression model and a random forest model on the Iris dataset.

    This script trains two models using two nested MLflow runs. The outer run is
    used to log experiment metadata and the inner runs are used to log each model's
    hyperparameters and metrics.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    with mlflow.start_run(
        run_name="model/classifier_model", log_system_metrics=True
    ) as run:
        LOG.info("Run ID: %s", run.info.run_id)

        extra_tags = {"developer": "Ariel"}
        # Load the Iris dataset
        data = datasets.load_iris(as_frame=True)
        X = data["data"]
        y = data["target"].astype(
            "float32"
        )  # Handles Integer columns in Python cannot represent missing values.

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        with mlflow.start_run(
            run_name="logistic-regression",
            tags=extra_tags,
            nested=True,
        ):
            # Define the model hyperparameters
            params = {
                "penalty": "l2",
                "C": 1,
                "random_state": 42,
                "n_jobs": -1,
            }

            # Train the model
            clf = LogisticRegression(**params)

            with mlflow_helper.start_log(
                clf, params, X_train, y_train, X_test, y_test, trace=True
            ):
                clf.fit(X_train, y_train)

        with mlflow.start_run(
            run_name="random-forest",
            tags=extra_tags,
            nested=True,
        ):
            # Define the model hyperparameters
            params = {
                "n_estimators": 5,
                "max_depth": 5,
                "random_state": 42,
                "n_jobs": -1,
            }

            clf = RandomForestClassifier(**params)

            with mlflow_helper.start_log(
                clf, params, X_train, y_train, X_test, y_test, trace=True
            ):
                clf.fit(X_train, y_train)


if __name__ == "__main__":
    train()
