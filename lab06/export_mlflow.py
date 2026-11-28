from pathlib import Path

import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient
from train import load_config


def main():

    config = load_config()

    tracking_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    experiment = client.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:
        raise RuntimeError("MLflow experiment not found.")

    output_dir = Path("results/mlflow_export")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    runs.to_csv(
        output_dir / "mlflow_runs.csv",
        index=False
    )

    parameters = []
    metrics = []
    metric_history = []
    artifacts = []

    def collect_artifacts(run_id, run_name, path=None):

        for artifact in client.list_artifacts(run_id, path):

            if artifact.is_dir:

                collect_artifacts(
                    run_id,
                    run_name,
                    artifact.path
                )

            else:

                artifacts.append({
                    "run_id": run_id,
                    "run_name": run_name,
                    "artifact_path": artifact.path,
                    "file_size_bytes": artifact.file_size
                })

    for run_id in runs["run_id"]:

        run = client.get_run(run_id)

        run_name = run.data.tags.get(
            "mlflow.runName",
            run_id
        )

        print(f"Exporting: {run_name}")

        for key, value in run.data.params.items():

            parameters.append({
                "run_id": run_id,
                "run_name": run_name,
                "parameter": key,
                "value": value
            })

        for key, value in run.data.metrics.items():

            metrics.append({
                "run_id": run_id,
                "run_name": run_name,
                "metric": key,
                "value": value
            })

            history = client.get_metric_history(
                run_id,
                key
            )

            for item in history:

                metric_history.append({
                    "run_id": run_id,
                    "run_name": run_name,
                    "metric": key,
                    "step": item.step,
                    "value": item.value,
                    "timestamp": item.timestamp
                })

        collect_artifacts(run_id, run_name)

    pd.DataFrame(parameters).to_csv(
        output_dir / "mlflow_parameters.csv",
        index=False
    )

    pd.DataFrame(metrics).to_csv(
        output_dir / "mlflow_metrics.csv",
        index=False
    )

    pd.DataFrame(metric_history).to_csv(
        output_dir / "mlflow_metric_history.csv",
        index=False
    )

    pd.DataFrame(artifacts).to_csv(
        output_dir / "mlflow_artifacts.csv",
        index=False
    )

    summary = (
        "MLflow Experiment Export\n"
        "========================\n"
        f"Experiment: {experiment_name}\n"
        f"Experiment ID: {experiment.experiment_id}\n"
        f"Total runs: {len(runs)}\n"
        f"Parameter records: {len(parameters)}\n"
        f"Metric records: {len(metrics)}\n"
        f"Metric history records: {len(metric_history)}\n"
        f"Artifact records: {len(artifacts)}\n"
    )

    (output_dir / "mlflow_summary.txt").write_text(
        summary,
        encoding="utf-8"
    )

    print("\n" + summary)

    print(f"Export directory: {output_dir}")

    print("MLflow export completed successfully.")


if __name__ == "__main__":
    main()
