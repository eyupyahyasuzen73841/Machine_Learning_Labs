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
    evaluate
)

from model import EyupLeNetVariant


RESULTS_DIR = Path("results")
EPOCHS = 5

EXPERIMENTS = {
    "normal": (256, 1),
    "accumulation": (64, 4)
}


def get_peak_memory_mb():

    peak = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    if sys.platform == "darwin":
        return peak / (1024 * 1024)

    return peak / 1024


def train_accumulation(
    model,
    loader,
    optimizer,
    device,
    accumulation_steps
):

    model.train()

    criterion = nn.CrossEntropyLoss(
        reduction="sum"
    )

    optimizer.zero_grad()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    accumulated_samples = 0
    optimizer_steps = 0

    for step, (images, labels) in enumerate(loader, 1):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        batch_samples = labels.size(0)

        accumulated_samples += batch_samples

        total_loss += loss.item()

        total_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        total_samples += batch_samples

        if (
            step % accumulation_steps == 0
            or step == len(loader)
        ):

            for parameter in model.parameters():

                if parameter.grad is not None:

                    parameter.grad.div_(
                        accumulated_samples
                    )

            optimizer.step()
            optimizer.zero_grad()

            optimizer_steps += 1
            accumulated_samples = 0

    return (
        total_loss / total_samples,
        total_correct / total_samples,
        optimizer_steps
    )


def run_experiment(name):

    batch_size, accumulation_steps = EXPERIMENTS[name]

    config = load_config()

    config["data"]["batch_size"] = batch_size
    config["data"]["num_workers"] = 0

    seed = config["project"]["seed"]

    set_seed(seed)

    device = torch.device("cpu")

    train_loader, test_loader = get_dataloaders(config)

    model = EyupLeNetVariant(
        num_classes=config["model"]["num_classes"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
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

    history = []

    total_time = 0.0
    total_optimizer_steps = 0

    with mlflow.start_run(
        run_name=f"Task6_{name}"
    ):

        mlflow.log_params({
            "method": name,
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "effective_batch_size": (
                batch_size * accumulation_steps
            ),
            "epochs": EPOCHS,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "scheduler": "StepLR",
            "seed": seed,
            "device": "cpu",
            "num_workers": 0
        })

        for epoch in range(1, EPOCHS + 1):

            start = time.perf_counter()

            train_loss, train_accuracy, steps = (
                train_accumulation(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    accumulation_steps
                )
            )

            epoch_time = time.perf_counter() - start

            total_time += epoch_time
            total_optimizer_steps += steps

            test_loss, test_accuracy = evaluate(
                model,
                test_loader,
                criterion,
                device
            )

            scheduler.step()

            metrics = {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "epoch_time_seconds": epoch_time,
                "optimizer_steps": steps,
                "peak_memory_mb": get_peak_memory_mb()
            }

            mlflow.log_metrics(
                metrics,
                step=epoch
            )

            history.append({
                "method": name,
                "epoch": epoch,
                **metrics
            })

            print(
                f"{name} | "
                f"Epoch {epoch}/{EPOCHS} | "
                f"Train Accuracy: {train_accuracy:.4f} | "
                f"Test Accuracy: {test_accuracy:.4f} | "
                f"Optimizer Steps: {steps} | "
                f"Time: {epoch_time:.2f}s",
                flush=True
            )

        throughput = (
            len(train_loader.dataset)
            * EPOCHS
            / total_time
        )

        result = {
            "method": name,
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "effective_batch_size": (
                batch_size * accumulation_steps
            ),
            "epochs": EPOCHS,
            "optimizer_steps": total_optimizer_steps,
            "training_time_seconds": total_time,
            "throughput_samples_per_second": throughput,
            "peak_memory_mb": get_peak_memory_mb(),
            "final_train_accuracy": train_accuracy,
            "final_test_accuracy": test_accuracy,
            "final_test_loss": test_loss
        }

        RESULTS_DIR.mkdir(exist_ok=True)

        result_path = (
            RESULTS_DIR / f"task6_{name}.json"
        )

        history_path = (
            RESULTS_DIR / f"task6_{name}_history.csv"
        )

        result_path.write_text(
            json.dumps(result, indent=4),
            encoding="utf-8"
        )

        pd.DataFrame(history).to_csv(
            history_path,
            index=False
        )

        mlflow.log_metrics({
            "final_test_accuracy": test_accuracy,
            "total_optimizer_steps": total_optimizer_steps,
            "average_throughput": throughput,
            "final_peak_memory_mb": get_peak_memory_mb()
        })

        mlflow.log_artifact(str(result_path))
        mlflow.log_artifact(str(history_path))


def main():

    RESULTS_DIR.mkdir(exist_ok=True)

    results = []
    histories = []

    for name in EXPERIMENTS:

        print(f"\nTesting {name}", flush=True)

        subprocess.run(
            [sys.executable, __file__, name],
            check=True
        )

        result_path = (
            RESULTS_DIR / f"task6_{name}.json"
        )

        history_path = (
            RESULTS_DIR / f"task6_{name}_history.csv"
        )

        results.append(
            json.loads(
                result_path.read_text(encoding="utf-8")
            )
        )

        histories.append(
            pd.read_csv(history_path)
        )

    df = pd.DataFrame(results)

    history_df = pd.concat(
        histories,
        ignore_index=True
    )

    csv_path = (
        RESULTS_DIR / "task6_accumulation_study.csv"
    )

    df.to_csv(csv_path, index=False)

    print("\nGradient Accumulation Results:")
    print(df.to_string(index=False))

    plt.figure(figsize=(9, 5))

    for name, group in history_df.groupby("method"):

        plt.plot(
            group["epoch"],
            group["test_accuracy"],
            marker="o",
            label=name
        )

    plt.title("Gradient Accumulation vs Normal Training")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.xticks(range(1, EPOCHS + 1))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    accuracy_path = (
        RESULTS_DIR / "task6_accuracy.png"
    )

    plt.savefig(accuracy_path, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))

    plt.bar(
        df["method"],
        df["peak_memory_mb"]
    )

    plt.title("Peak Memory Comparison")
    plt.ylabel("Peak Process Memory (MiB)")
    plt.tight_layout()

    memory_path = (
        RESULTS_DIR / "task6_memory.png"
    )

    plt.savefig(memory_path, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))

    plt.bar(
        df["method"],
        df["throughput_samples_per_second"]
    )

    plt.title("Training Throughput Comparison")
    plt.ylabel("Throughput (samples/s)")
    plt.tight_layout()

    throughput_path = (
        RESULTS_DIR / "task6_throughput.png"
    )

    plt.savefig(throughput_path, dpi=300)
    plt.close()

    config = load_config()

    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"]
    )

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    with mlflow.start_run(
        run_name="Task6_Accumulation_Summary"
    ):

        for path in [
            csv_path,
            accuracy_path,
            memory_path,
            throughput_path
        ]:

            mlflow.log_artifact(str(path))

    print(
        "\nGradient accumulation study completed successfully."
    )


if __name__ == "__main__":

    if len(sys.argv) == 2:
        run_experiment(sys.argv[1])
    else:
        main()
