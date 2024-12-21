## Useful tutorial
https://medium.com/@johnthuo/experiment-tracking-and-model-registry-with-mlflow-a74c30217e8c

## MLFlow
### Pipeline
```bash
# Execute given entry points
mlflow run -e download .
mlflow run -e process .
mlflow run -e train .

# Execute in the local Python/virtual environment
mlflow run --env-manager=local .

# Execute with parameters
mlflow run --experiment-name "Experiment1" -P n_jobs_param=3 -P regularization=0.5 -P max_iter=200 .

```