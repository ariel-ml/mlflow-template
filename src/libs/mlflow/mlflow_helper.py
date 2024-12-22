import os
import logging
from contextlib import contextmanager
from typing import Any, Dict, Literal
from mlflow.entities import ViewType, Span
from mlflow.tracking import MlflowClient
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.models import infer_signature
from mlflow.data import from_pandas
import mlflow
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from libs.data_etl import data_utils

LOG = logging.getLogger(__name__)


def __fully_qualified_class_name(obj):
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def __concat_columns(X, y):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    data = pd.concat([X, y], axis=1)
    data.rename(
        columns={
            data.columns[-1]: data_utils.TARGET_COLUMN,
        },
        inplace=True,
    )
    return data


def __log_metrics(metrics: Dict[str, Any]):
    for key, value in metrics.items():
        LOG.info(
            "%s: %s",
            key,
            value if isinstance(value, (float, int)) else value.item(),
        )


def get_actual_experiment_name() -> str:
    return mlflow.get_experiment(mlflow.active_run().info.experiment_id).name


def setup(
    experiment_name: str,
    system_metrics: bool = False,
    system_metrics_sampling_interval: int = 2,
):
    # Set the name of the MLflow Experiment
    """
    Set up MLflow experiment and system metrics.

    Parameters
    ----------
    experiment_name : str
        The name of the MLflow Experiment.
    system_metrics : bool, default=True
        Whether to enable system metrics logging globally.
    """
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")

    if experiment_id is None:
        mlflow.set_experiment(experiment_name)

    if system_metrics:
        # Enable system metrics globally
        mlflow.system_metrics.enable_system_metrics_logging()

        # Adjust sampling interval and logging frequency (optional)
        mlflow.system_metrics.set_system_metrics_sampling_interval(
            system_metrics_sampling_interval
        )


def log_auto(
    estimator: BaseEstimator,
    hyper_params: Dict[str, Any],
    X_test,
    y_test,
    test_metrics: bool = True,
    register_model: bool = False,
) -> Dict[str, Any]:
    """
    Log modified hyperparameters and extra metrics for mlflow.autolog() with MLflow.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to be logged.
    hyper_params : Dict[str, Any]
        The hyperparameters used in the estimator.
    X_test : pandas.DataFrame
        The test data features.
    y_test : pandas.Series
        The test data target labels.
    test_metrics : bool, default=True
        If True, log the metrics of the estimator on the test data.
    register_model : bool, default=False
        If True, register the model in the MLflow Model Registry.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the test metrics if test_metrics is True, otherwise an empty dictionary.
    """
    LOG.info("Logging training parameters...")

    for key, value in hyper_params.items():
        mlflow.log_param("_modified/" + key, value)

    mlflow.log_params(hyper_params)

    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    if register_model:
        mlflow.register_model(
            model_uri=model_uri,
            name=get_actual_experiment_name() + " / " + type(estimator).__name__,
        )

    if X_test is not None and y_test is not None:
        test_data = __concat_columns(X_test, y_test)
    else:
        test_data = None

    if test_metrics and test_data is not None:
        test_result = mlflow.evaluate(
            model=estimator.predict,  # model_info.model_uri,
            data=test_data,
            targets=data_utils.TARGET_COLUMN,
            model_type=getattr(estimator, "_estimator_type", None),
            evaluator_config={"metric_prefix": "test_"},
        )

        __log_metrics(test_result.metrics)

        return test_result.metrics
    return {}


def log(
    estimator: BaseEstimator,
    hyper_params: dict,
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None,
    train_metrics: bool = False,
    test_metrics: bool = True,
    register_model: bool = False,
) -> Dict[str, Any]:
    """
    Log params to MLflow.

    Parameters
    ----------
    estimator : BaseEstimator
        The model to be logged.
    hyper_params : dict
        The hyperparameters used to train the model.
    X_train : pandas.DataFrame
        The training data features.
    y_train : pandas.Series
        The training data target labels.
    X_test : pandas.DataFrame
        The test data features.
    y_test : pandas.Series
        The test data target labels.
    train_metrics : bool, default=False
        If True, log the training metrics.
    test_metrics : bool, default=True
        If True, log the test metrics.
    register_model : bool, default=False
        If True, register the model in the MLflow Model Registry.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the test metrics if test_metrics is True, otherwise an empty dictionary.
    """
    LOG.info("Logging training parameters...")
    # Log the hyperparameters
    mlflow.log_params(hyper_params)

    # Calculate metrics
    # Log metrics
    # mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
    # mlflow.log_metric(
    #     "test_precision",
    #     precision_score(y_test, y_pred, average="weighted"),
    # )
    # mlflow.log_metric(
    #     "test_recall", recall_score(y_test, y_pred, average="weighted")
    # )
    # mlflow.log_metric("test_f1", f1_score(y_test, y_pred, average="weighted"))
    # mlflow.log_metric("test_log_loss", log_loss(y_test, y_pred_proba))
    # mlflow.log_metric(
    #     "test_roc_auc_score", roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    # )
    if X_train is not None and y_train is not None:
        train_data = __concat_columns(X_train, y_train)
        train_dataset = from_pandas(train_data, targets=data_utils.TARGET_COLUMN)
        mlflow.log_input(train_dataset, context="Train")
    else:
        train_dataset = None

    if X_test is not None and y_test is not None:
        test_data = __concat_columns(X_test, y_test)
        test_dataset = from_pandas(test_data, targets=data_utils.TARGET_COLUMN)
        mlflow.log_input(test_dataset, context="Test")
    else:
        test_dataset = None

    estimator_name = type(estimator).__name__

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("estimator", __fully_qualified_class_name(estimator))
    mlflow.set_tag("estimator_name", estimator_name)

    # cm = confusion_matrix(y_test, y_pred, normalize="true")
    # plt.figure(figsize=(8, 6))
    # heatmap = sns.heatmap(cm, annot=True, cmap="Blues")
    # plt.title("Normalized Confusion Matrix")
    # plt.xlabel("Predicted label")
    # plt.ylabel("True label")

    # heatmap_figure = heatmap.get_figure()
    # heatmap_file_path = "test_confusion_matrix.png"
    # heatmap_figure.savefig(heatmap_file_path)
    # plt.close()

    # # Add the heatmap to artifacts
    # mlflow.log_artifact(heatmap_file_path)
    # os.remove(heatmap_file_path)

    # Infer the model signature
    signature = infer_signature(X_train, estimator.predict(X_train))

    # Log the model
    mlflow.sklearn.log_model(
        sk_model=estimator,
        artifact_path="model",
        signature=signature,
        input_example=(
            X_test[:3]
            if X_test is not None
            else X_train[:3] if X_train is not None else None
        ),
        registered_model_name=(
            get_actual_experiment_name() + " / " + estimator_name
            if register_model
            else None
        ),  # if set model will be registered
    )

    estimator_type = getattr(estimator, "_estimator_type", None)

    if train_metrics and train_dataset is not None:
        mlflow.evaluate(
            model=estimator.predict,
            data=train_dataset,
            model_type=estimator_type,
            evaluator_config={"metric_prefix": "train_"},
        )

    if test_metrics and test_dataset is not None:
        test_result = mlflow.evaluate(
            model=estimator.predict,
            data=test_dataset,
            model_type=estimator_type,
            evaluator_config={"metric_prefix": "test_"},
        )

        __log_metrics(test_result.metrics)
        return test_result.metrics
    return {}


@contextmanager
def start_log(
    estimator: BaseEstimator,
    hyper_params: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    trace: bool = False,
    log_train_metrics: bool = False,
    register_model: bool = False,
):
    """
    Context manager to log the training and evaluation of a model.

    Parameters
    ----------
    estimator : BaseEstimator
        The model to be trained and evaluated.
    hyper_params : dict
        The hyperparameters used for training.
    X_train, y_train : array-like
        The training data.
    X_test, y_test : array-like
        The test data.
    trace : bool, optional
        Whether to trace the training and evaluation using MLflow. Defaults to False.
    log_train_metrics : bool, optional
        Whether to log the training metrics. Defaults to False.
    register_model : bool, optional
        Whether to register the model. Defaults to False.

    Yields
    -------
    None or MLflow Span if `trace` is True

    Notes
    -----
    This context manager should be used with the `with` statement. It will log the
    training and evaluation of the model using MLflow. If `trace` is True, it will
    also trace the training and evaluation using MLflow. If `log_train_metrics` is
    True, it will also log the training metrics. If `register_model` is True, it will
    also register the model.

    Example usage:
    -------------
    .. code-block:: python
        with start_log(estimator,
                       hyper_params,
                       X_train,
                       y_train,
                       X_test,
                       y_test,
                       trace=True,
                       log_train_metrics=True,
                       register_model=True) as span:
            estimator.fit(X_train, y_train)
    """
    span: Span = None
    span_generator = None
    try:
        if trace:
            clf_name = type(estimator).__name__
            span_generator = mlflow.start_span(f"fit/{clf_name}")
            span = span_generator.__enter__()
            span.set_inputs(hyper_params)
        yield span if trace else None
    finally:
        test_metrics = log(
            estimator=estimator,
            hyper_params=hyper_params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            train_metrics=log_train_metrics,
            test_metrics=True,
            register_model=register_model,
        )
        if span:
            span.set_outputs(test_metrics)
            span_generator.__exit__(None, None, None)


def register_best_model(
    experiment_name: str, client: MlflowClient, metric: str
) -> None:
    """
    Register the best model of the experiment in the MLflow Model Registry.

    The best model is selected based on the given metric. The best model is the one with the lowest value of the metric.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment that contains the models to be compared.
    client : MlflowClient
        The MLflow client to use for interacting with the MLflow server.
    metric : str
        The metric to use for selecting the best model.
    """
    LOG.info("Registering best model...")
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=[f"metrics.{metric} ASC"],
    )
    best_run = runs[0]
    best_model_uri = f"runs:/{best_run.info.run_id}/model"
    mlflow.register_model(model_uri=best_model_uri, name="Best Regression Model")
    LOG.info("Registered model %s as 'Best Regression Model'", {best_model_uri})


def load_best_model(
    experiment_name: str,
    metric_name: str,
    metric_order: Literal["ASC", "DESC"] = "DESC",
    download=False,
    path="models",
) -> BaseEstimator:
    """
    Load the best model of an experiment based on a given metric.

    Parameters
    ----------
    experiment_name : str
        The name of the experiment to search for the best model.
    metric_name : str
        The name of the metric to use for evaluating the best model.
    metric_order : Literal["ASC", "DESC"], default="DESC"
        The order in which to sort the runs based on the given metric.
    download : bool, default=False
        Whether to download the model from the artifact store to the local machine.
    path : str, default="models"
        The path to download the model to.

    Returns
    -------
    BaseEstimator
        The best model associated with the experiment and metric name.

    Notes
    -----
    The best model is the one with the lowest (`metric_order`='ASC') or
    the highest value (`metric_order`='DESC') of the given metric.
    If no runs associated with the experiment have the given metric, the function returns None.
    If no model versions are associated with the best run, the function returns None.
    """
    client = MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        LOG.warning("Experiment %s does not exist", experiment_name)
        return None

    # Search for all runs associated with this model
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        # filter_string=f"tags.estimator_name = '{estimator_name}'",
        order_by=[f"metrics.{metric_name} {metric_order}"],
        max_results=1,
    )

    if not runs or len(runs) == 0:
        LOG.warning("No runs found")
        return None

    # Get the run ID of the best run
    best_run_id = runs[0].info.run_id
    LOG.info("Best run ID: %s", best_run_id)

    mvs = client.search_model_versions(f"run_id='{best_run_id}'", max_results=1)

    if not mvs or len(mvs) == 0:
        LOG.warning("No model versions found")
        return None

    mv = mvs[0]
    download_uri = client.get_model_version_download_uri(mv.name, mv.version)
    LOG.info("Model download URI: %s", download_uri)

    # Get Sklearn model option 1
    model_uri = f"models:/{mv.name}/{mv.version}"
    if download:
        repo = ModelsArtifactRepository(model_uri)
        repo.download_artifacts(artifact_path="model.pkl", dst_path=path)
        LOG.info("Model downloaded to %s", path)

    # Get Sklearn model option 2
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    original_model = pyfunc_model.get_raw_model()

    return original_model
