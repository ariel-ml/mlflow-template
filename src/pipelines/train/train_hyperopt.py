import mlflow
import click
import optuna
import pandas as pd
from dotenv import load_dotenv
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from libs.mlflow import mlflow_helper
from libs.data_etl import paths, data_utils
from libs import logger


LOG = logger.getLogger(__name__)

load_dotenv()

mlflow_helper.setup("Hyperopt Experiment", system_metrics=True)


@click.command(no_args_is_help=True)
@click.option(
    "--n_trials",
    default=10,
    help="The number of parameter evaluations for the optimizer to explore",
)
@mlflow.trace(name="optimize/random_forest")
def optimize(n_trials=None):
    X, y, _ = data_utils.load_data(paths.RAW_DATA_FILE)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Objective function to optimize
    def objective(trial: optuna.Trial):
        # Define hyperparameter search space
        n_estimators = trial.suggest_int("n_estimators", 5, 150)
        max_depth = trial.suggest_int("max_depth", 2, 32)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        ccp_alpha = trial.suggest_float("ccp_alpha", 0.01, 0.5, log=True)

        # Train a RandomForest model
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            random_state=42,
            n_jobs=-1,
        )
        # Use cross-validation to evaluate model performance
        return cross_val_score(
            clf, X_scaled, y, cv=3, scoring="accuracy", n_jobs=-1
        ).mean()

    with mlflow.start_run(
        run_name="optimize/random_forest",
        tags={"model": "random_forest", "developer": "Ariel"},
    ) as run:
        LOG.info("Run ID: %s", run.info.run_id)
        # Optimize the model
        with mlflow.start_span("hyperparam_search") as span:
            span.set_inputs({"n_trials": n_trials})
            # Run optimization
            study = optuna.create_study(direction="maximize")  # Maximizing the accuracy
            study.optimize(objective, n_trials=n_trials, n_jobs=-1)

            # Output the best hyperparameters
            best_params = study.best_params
            best_value = study.best_value

            LOG.info("Best hyperparameters: %s", best_params)
            LOG.info("Best metric value: %s", best_value)
            span.set_outputs(
                {"best_params": best_params, "best_metric_value": best_value}
            )

            best_model = RandomForestClassifier(**best_params)
            best_model.fit(X_scaled, y)

            mlflow_helper.log(
                best_model,
                best_params,
                X_scaled,
                y,
                register_model=True,
                test_metrics=False,
                train_metrics=True,
            )


if __name__ == "__main__":
    optimize()
