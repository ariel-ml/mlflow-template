import os
import mlflow


print(list(filter(lambda x: x.startswith("MLFLOW"), os.environ.keys())))
print(os.environ["MLFLOW_RUN_ID"])
print(os.environ["MLFLOW_EXPERIMENT_ID"])


# Method 1: Retrieve experiment name from environment variable
experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")

# Method 2: Retrieve experiment name using MLflow API
if experiment_id:
    experiment_api = mlflow.get_experiment(experiment_id)
    experiment_name = experiment_api.name
else:
    experiment_name = None

print(f"Experiment Name (from env): {experiment_id}")
print(f"Experiment Name (from API): {experiment_name}")
