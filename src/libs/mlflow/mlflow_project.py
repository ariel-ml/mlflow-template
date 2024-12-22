import click
import mlflow
import mlflow.projects
from dotenv import load_dotenv

load_dotenv()


@click.command(no_args_is_help=True)
@click.option(
    "--entry-points",
    default=None,
    required=False,
    type=str,
    help="'None': runs all defined entry points or comma separated list of choosen entry points",
)
@click.option("--experiment-name", required=True, type=str)
@click.option(
    "--env-manager", default="local", required=False, type=str, help="Default: local"
)
def run_pipeline(
    entry_points: str = None, experiment_name: str = None, env_manager: str = None
):
    """
    Run all or a subset of the entry points in the MLproject file.

    Parameters:
        entry_points (str): 'all' or comma separated list of entry points.
        experiment_name (str): Name of the MLflow experiment.
        env_manager (str): Environment manager to use for the run.

    Returns:
        None
    """
    if entry_points is None:
        # Load the MLproject file
        mlflow_project = mlflow.projects.load_project(".")
        # Get all entry points
        mlflow_project_entry_points = getattr(mlflow_project, "_entry_points")
        entry_points_list = list(mlflow_project_entry_points.keys())
    else:
        entry_points_list = entry_points.split(",")

    for entry_point in entry_points_list:
        mlflow.projects.run(
            uri=".",
            entry_point=entry_point,
            experiment_name=experiment_name,
            env_manager=env_manager,
        )


if __name__ == "__main__":
    run_pipeline()
