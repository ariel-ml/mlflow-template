import mlflow
import click
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from libs import mlflow_helper
from libs import logger

LOG = logger.getLogger(__name__)

mlflow_helper.setup("MLflow Hyperopt Experiment", system_metrics=True)


@click.command(no_args_is_help=True)
@click.option("--estimators", default=100, type=int)
@click.option("--max-depth", default=None, type=int)
@mlflow.trace(name="train/random_forest")
def train(estimators=None, max_depth=None):

    with mlflow.start_run(run_name="model/classifier_model") as run:
        mlflow.autolog(
            log_input_examples=True,
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
        with mlflow.start_span("fit") as span:
            span.set_inputs({"x": X_train[:3], "y": y_train[:3]})
            result = model.fit(X_train, y_train)
            span.set_outputs({"output": result})
        # or
        # mlflow.trace(model.fit)(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        run_id = run.info.run_id
        print(run_id)

        for key, value in params.items():
            mlflow.log_param("--" + key, value)

        mlflow.log_params(params)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train[:3],
            registered_model_name="model_autolog",
        )


if __name__ == "__main__":
    train()
