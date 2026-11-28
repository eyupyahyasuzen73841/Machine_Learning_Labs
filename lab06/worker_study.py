import time
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from train import load_config, set_seed, get_dataloaders
from model import EyupLeNetVariant


def main():
    config = load_config()

    workers_list = [0, 1, 2, 4, 8]
    device = torch.device("cpu")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results = []

    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"]
    )

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    for workers in workers_list:

        print(f"\nTesting num_workers = {workers}", flush=True)

        config["data"]["num_workers"] = workers

        set_seed(config["project"]["seed"])

        train_loader, _ = get_dataloaders(config)

        model = EyupLeNetVariant(
            num_classes=config["model"]["num_classes"]
        ).to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"]
        )

        model.train()

        total_samples = 0
        total_loading_time = 0.0
        total_compute_time = 0.0

        start_time = time.perf_counter()

        loading_start = time.perf_counter()
        iterator = iter(train_loader)
        total_loading_time += (
            time.perf_counter() - loading_start
        )

        for step in range(len(train_loader)):

            loading_start = time.perf_counter()

            images, labels = next(iterator)

            images = images.to(device)
            labels = labels.to(device)

            total_loading_time += (
                time.perf_counter() - loading_start
            )

            compute_start = time.perf_counter()

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_compute_time += (
                time.perf_counter() - compute_start
            )

            total_samples += labels.size(0)

            if (step + 1) % 200 == 0:
                print(
                    f"Workers {workers} | "
                    f"Batch {step + 1}/{len(train_loader)}",
                    flush=True
                )

        total_time = time.perf_counter() - start_time

        throughput = total_samples / total_time

        loading_percentage = (
            total_loading_time / total_time * 100
        )

        result = {
            "num_workers": workers,
            "samples": total_samples,
            "epoch_time_seconds": total_time,
            "data_loading_seconds": total_loading_time,
            "compute_seconds": total_compute_time,
            "data_loading_percentage": loading_percentage,
            "throughput_samples_per_second": throughput
        }

        results.append(result)

        with mlflow.start_run(
            run_name=f"Task3_Workers_{workers}"
        ):

            mlflow.log_params({
                "num_workers": workers,
                "batch_size": config["data"]["batch_size"],
                "epochs": 1,
                "device": "cpu"
            })

            mlflow.log_metrics({
                "epoch_time_seconds": total_time,
                "data_loading_seconds": total_loading_time,
                "compute_seconds": total_compute_time,
                "data_loading_percentage": loading_percentage,
                "throughput": throughput
            })

        print(f"Epoch time: {total_time:.2f} seconds")
        print(f"Throughput: {throughput:.2f} samples/s")
        print(f"Data loading: {loading_percentage:.2f}%")

        del iterator
        del train_loader
        del model
        del optimizer

    df = pd.DataFrame(results)

    csv_path = results_dir / "task3_worker_study.csv"
    df.to_csv(csv_path, index=False)

    print("\nWorker Study Results:")
    print(df.to_string(index=False))

    best = df.loc[
        df["throughput_samples_per_second"].idxmax()
    ]

    print(f"\nBest num_workers: {int(best['num_workers'])}")
    print(
        f"Best throughput: "
        f"{best['throughput_samples_per_second']:.2f} samples/s"
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        df["num_workers"],
        df["throughput_samples_per_second"],
        marker="o",
        linewidth=2
    )

    plt.title("DataLoader Workers vs Training Throughput")
    plt.xlabel("Number of Workers")
    plt.ylabel("Throughput (samples/second)")
    plt.xticks(workers_list)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    throughput_path = results_dir / "task3_throughput.png"

    plt.savefig(throughput_path, dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))

    plt.bar(
        df["num_workers"].astype(str),
        df["data_loading_percentage"]
    )

    plt.title("Data Loading Time by Worker Count")
    plt.xlabel("Number of Workers")
    plt.ylabel("Data Loading Time (%)")
    plt.tight_layout()

    loading_path = results_dir / "task3_data_loading.png"

    plt.savefig(loading_path, dpi=300)
    plt.close()

    with mlflow.start_run(run_name="Task3_Worker_Summary"):

        mlflow.log_metric(
            "best_num_workers",
            int(best["num_workers"])
        )

        mlflow.log_metric(
            "best_throughput",
            float(best["throughput_samples_per_second"])
        )

        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(throughput_path))
        mlflow.log_artifact(str(loading_path))

    print("\nWorker study completed successfully.")


if __name__ == "__main__":
    main()
