# MLFlow Template Project

## Overview

This project is a machine learning pipeline template designed to preprocess data, train models, and manage experiments using MLflow. It includes various components for data handling, feature engineering, and model training, both manually and with autologging.

## Useful tutorial
https://medium.com/@johnthuo/experiment-tracking-and-model-registry-with-mlflow-a74c30217e8c

## Project Structure

- `data`: Contains raw / processed / etc. data.
- `notebooks/*.ipynb`: Jupyter Notebooks for sketching and exploratory analysis.
- `models`: Houses final models.
- `src/libs`: Contains utility libraries for data ETL, MLflow integration, and logging.
- `src/pipelines`: Houses different pipeline stages such as feature engineering and model training.

## Key Components

- **Data ETL**: Uses `pandas` and `Pathlib` for data manipulation and file path handling.
- **MLflow Integration**: Facilitates experiment tracking, model registration, and system metrics logging.
- **Model Training**: Supports manual, autolog, nested and Optuna based training pipelines with `RandomForestClassifier`.

## Getting Started

1. **Setup Environment**: Ensure you have a Python environment with the necessary dependencies. 
    * Use `poetry install` to install the required packages. 
    * Use `poetry shell` to activate the environment.
2. **MLflow Setup**:
    
    Manual:
    * `mlflow server --app-name basic-auth --backend-store-uri sqlite:////mlflow/mlruns/mlruns.db --artifacts-destination /mlflow/artifacts --host 0.0.0.0 --port 5000`
    
    Docker:
    * `docker compose up -d`
3. **Run training**: Execute the training scripts under `src/pipelines` to train models.
    * `python src/pipelines/train/train_manual.py --estimators 100 --max-depth 5`
    
        or
    * `python -m pipelines.train.train_manual --estimators 100 --max-depth 5`
4. **Run MLFlow Project**: Execute MLFlow project entry points.
    * Native way using local Poetry .venv:
        ```bash
        # Execute given entry points
        mlflow run -e load --env-manager=local .
        mlflow run -e split --env-manager=local .
        mlflow run -e scale --env-manager=local .
        mlflow run -e train --env-manager=local -P estimators=100 -P max_depth=7 .
        ```
    * Using convenience wrapper: 

        `python -m pipelines.project_runner --entry-points load,split,scale,train --experiment-name ProjectExp1`

## Usage

- **Training**: Use the `train_manual.py`, `train_autolog.py`, `train_nested.py` scripts to train models.
- **Parameter Optimization**: Use the `train_hyperopt.py` script to perform hyperparameter optimization using Optuna.
- **Experiment Tracking**: Utilize MLflow to log parameters, metrics, and artifacts.

## Dependencies

- Python 3.11.7
- MLflow
- Scikit-learn
- Click
- Dotenv
- Optuna

## License

This project is licensed under the MIT License.


