import time
import resource
import sys
import subprocess
import json
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from train import (
    load_config,
    set_seed,
    get_dataloaders,
    train_one_epoch,
    evaluate
)
from model import EyupLeNetVariant


RESULTS_DIR = Path("results")
BATCH_SIZES = [32, 64, 128, 256, 512]


def get_peak_memory_mb():
    peak = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    if sys.platform == "darwin":
        return peak / (1024 * 1024)

    return peak / 1024


def run_experiment(batch_size):
    config = load_config()

    config["data"]["batch_size"] = batch_size
    config["data"]["num_workers"] = 0
    config["training"]["epochs"] = 5

    set_seed(config["project"]["seed"])

    device = torch.device("cpu")

    train_loader, test_loader = get_dataloaders(config)

    model = EyupLeNetVariant(
        num_classes=config["model"]["num_classes"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["scheduler"]["step_size"],
        gamma=config["scheduler"]["gamma"]
    )

    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"]
    )

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    total_training_time = 0.0
    total_samples = 0

    with mlflow.start_run(
        run_name=f"Task4_Batch_{batch_size}"
    ):

        mlflow.log_params({
            "batch_size": batch_size,
            "num_workers": 0,
            "epochs": 5,
            "optimizer": "Adam",
            "scheduler": "StepLR",
            "device": "cpu",
            "seed": config["project"]["seed"]
        })

        for epoch in range(1, 6):

            start = time.perf_counter()

            train_loss, train_accuracy = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )

            epoch_time = time.perf_counter() - start

            total_training_time += epoch_time
            total_samples += len(train_loader.dataset)

            test_loss, test_accuracy = evaluate(
                model,
                test_loader,
                criterion,
                device
            )

            scheduler.step()

            throughput = (
                len(train_loader.dataset) / epoch_time
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "epoch_time_seconds": epoch_time,
                "throughput": throughput,
                "peak_memory_mb": get_peak_memory_mb()
            }, step=epoch)

            print(
                f"Batch Size {batch_size} | "
                f"Epoch {epoch}/5 | "
                f"Train Accuracy: {train_accuracy:.4f} | "
                f"Test Accuracy: {test_accuracy:.4f} | "
                f"Time: {epoch_time:.2f}s",
                flush=True
            )

        average_throughput = (
            total_samples / total_training_time
        )

        peak_memory = get_peak_memory_mb()

        result = {
            "batch_size": batch_size,
            "epochs": 5,
            "training_time_seconds": total_training_time,
            "throughput_samples_per_second": average_throughput,
            "peak_memory_mb": peak_memory,
            "final_train_accuracy": train_accuracy,
            "final_test_accuracy": test_accuracy,
            "final_test_loss": test_loss
        }

        mlflow.log_metrics({
            "average_throughput": average_throughput,
            "peak_memory_mb": peak_memory,
            "final_test_accuracy": test_accuracy
        })

        RESULTS_DIR.mkdir(exist_ok=True)

        result_path = (
            RESULTS_DIR / f"task4_batch_{batch_size}.json"
        )

        result_path.write_text(
            json.dumps(result, indent=4),
            encoding="utf-8"
        )

        mlflow.log_artifact(str(result_path))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    results = []

    for batch_size in BATCH_SIZES:

        print(
            f"\nTesting batch size = {batch_size}",
            flush=True
        )

        subprocess.run(
            [
                sys.executable,
                __file__,
                "--batch-size",
                str(batch_size)
            ],
            check=True
        )

        result_path = (
            RESULTS_DIR / f"task4_batch_{batch_size}.json"
        )

        results.append(
            json.loads(
                result_path.read_text(encoding="utf-8")
            )
        )

    df = pd.DataFrame(results)

    csv_path = RESULTS_DIR / "task4_batch_study.csv"
    df.to_csv(csv_path, index=False)

    print("\nBatch Size Study Results:")
    print(df.to_string(index=False))

    plots = [
        (
            "throughput_samples_per_second",
            "Batch Size vs Training Throughput",
            "Throughput (samples/s)",
            "task4_throughput.png"
        ),
        (
            "peak_memory_mb",
            "Batch Size vs Peak Memory",
            "Peak Process Memory (MiB)",
            "task4_memory.png"
        ),
        (
            "final_test_accuracy",
            "Batch Size vs Final Test Accuracy",
            "Test Accuracy",
            "task4_accuracy.png"
        )
    ]

    plot_paths = []

    for column, title, ylabel, filename in plots:

        plt.figure(figsize=(9, 5))

        plt.plot(
            df["batch_size"],
            df[column],
            marker="o",
            linewidth=2
        )

        plt.title(title)
        plt.xlabel("Batch Size")
        plt.ylabel(ylabel)
        plt.xticks(BATCH_SIZES)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plot_path = RESULTS_DIR / filename

        plt.savefig(plot_path, dpi=300)
        plt.close()

        plot_paths.append(plot_path)

    config = load_config()

    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"]
    )

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    with mlflow.start_run(
        run_name="Task4_Batch_Summary"
    ):

        mlflow.log_artifact(str(csv_path))

        for path in plot_paths:
            mlflow.log_artifact(str(path))

    print("\nBatch size study completed successfully.")


if __name__ == "__main__":

    if len(sys.argv) == 3 and sys.argv[1] == "--batch-size":
        run_experiment(int(sys.argv[2]))
    else:
        main()
