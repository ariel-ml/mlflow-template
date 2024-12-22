import mlflow
import click
from dotenv import load_dotenv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from libs.mlflow import mlflow_helper
from libs import logger

LOG = logger.getLogger(__name__)

load_dotenv()

mlflow_helper.setup("Autolog Experiment", system_metrics=True)


@click.command(no_args_is_help=True)
@click.option("--estimators", default=100, type=int)
@click.option("--max-depth", default=None, type=int)
# @mlflow.trace(name="trace/classification") # Auto metrics logging
def train(estimators=None, max_depth=None):
    """
    MLflow Autolog Example

    Train a random forest classifier on the Iris dataset.

    Parameters
    ----------
    estimators : int, optional
        The number of trees in the forest. Defaults to 100.
    max_depth : int, optional
        The maximum depth of the tree. Defaults to None, which means the tree will
        grow until all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    Returns
    -------
    None
    """
    with mlflow.start_run(
        run_name="model/classifier_model", log_system_metrics=True
    ) as run:
        LOG.info("Run ID: %s", run.info.run_id)
        mlflow.autolog(
            log_input_examples=False,
            extra_tags={"model": "random_forest", "developer": "Ariel"},
        )
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

        # Define the model hyperparameters
        params = {
            "n_estimators": estimators,
            "max_depth": max_depth,
            # "multi_class": "auto",
            "random_state": 42,
            "n_jobs": -1,
        }

        # Train the model
        model = RandomForestClassifier(**params)
        # Manual metrics logging
        with mlflow.start_span("fit") as span:
            span.set_inputs(params)
            result = model.fit(X_train, y_train)
            span.set_outputs({"output": result})
        # or
        # mlflow.trace(model.fit)(X_train, y_train)

        mlflow_helper.log_auto(
            estimator=model,
            hyper_params=params,
            X_test=X_test,
            y_test=y_test,
            test_metrics=True,
            register_model=False,
        )


if __name__ == "__main__":
    train()
