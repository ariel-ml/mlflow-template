import mlflow
import mlflow.system_metrics
from dotenv import load_dotenv
import click
from sklearn.ensemble import RandomForestClassifier
from libs import data_utils, mlflow_helper, paths
from libs import logger

LOG = logger.getLogger(__name__)

load_dotenv()

mlflow_helper.setup("MLflow Manual Experiment", system_metrics=True)


@click.command(no_args_is_help=True)
@click.option("--estimators", default=100, required=False, type=int)
@click.option("--max-depth", default=None, required=False, type=int)
@mlflow.trace
def train(estimators=100, max_depth=None):
    """
    MLFlow Manual Example

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
    # Start an MLflow run
    with mlflow.start_run(
        run_name="model/classifier_model",
        log_system_metrics=False,  # Enable system metrics for the current run
        tags={"model": "random_forest", "developer": "Ariel"},
    ) as run:
        LOG.info("Run ID: %s", run.info.run_id)

        # Load the processed Iris dataset
        X_train, y_train, _ = data_utils.load_data(paths.TRAIN_DATA_FILE)
        X_test, y_test, _ = data_utils.load_data(paths.TEST_DATA_FILE)

        # Define the model hyperparameters
        params = {
            "n_estimators": estimators,
            "max_depth": max_depth,
            "random_state": 42,
            "n_jobs": -1,
        }

        # Train the model
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)

        mlflow_helper.log(
            estimator=clf,
            hyper_params=params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            train_metrics=True,
            test_metrics=True,
            register_model=True,
        )


if __name__ == "__main__":
    train()
