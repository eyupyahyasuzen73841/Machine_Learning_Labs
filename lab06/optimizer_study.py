import time
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from torch.utils.data import DataLoader, random_split

from train import (
    load_config,
    set_seed,
    get_dataloaders,
    train_one_epoch,
    evaluate
)

from model import EyupLeNetVariant


def create_optimizer(name, model, learning_rate):

    if name == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

    if name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9
        )

    raise ValueError(f"Unknown optimizer: {name}")


def create_scheduler(name, optimizer, config):

    if name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["scheduler"]["step_size"],
            gamma=config["scheduler"]["gamma"]
        )

    if name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=0.00001
        )

    if name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=1
        )

    raise ValueError(f"Unknown scheduler: {name}")


def main():

    config = load_config()

    config["data"]["batch_size"] = 64
    config["data"]["num_workers"] = 0

    seed = config["project"]["seed"]
    epochs = 10
    device = torch.device("cpu")

    optimizer_names = ["Adam", "SGD"]

    scheduler_names = [
        "StepLR",
        "CosineAnnealingLR",
        "ReduceLROnPlateau"
    ]

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    all_history = []
    final_results = []

    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"]
    )

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    for optimizer_name in optimizer_names:

        for scheduler_name in scheduler_names:

            set_seed(seed)

            experiment_name = (
                f"{optimizer_name}_{scheduler_name}"
            )

            print(
                f"\nTesting {experiment_name}",
                flush=True
            )

            original_loader, test_loader = get_dataloaders(
                config
            )

            dataset = original_loader.dataset

            validation_size = 6000
            training_size = len(dataset) - validation_size

            train_dataset, validation_dataset = random_split(
                dataset,
                [training_size, validation_size],
                generator=torch.Generator().manual_seed(seed)
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=64,
                shuffle=True,
                num_workers=0,
                generator=torch.Generator().manual_seed(seed)
            )

            validation_loader = DataLoader(
                validation_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=0
            )

            model = EyupLeNetVariant(
                num_classes=config["model"]["num_classes"]
            ).to(device)

            criterion = nn.CrossEntropyLoss()

            learning_rate = (
                0.001 if optimizer_name == "Adam" else 0.01
            )

            optimizer = create_optimizer(
                optimizer_name,
                model,
                learning_rate
            )

            scheduler = create_scheduler(
                scheduler_name,
                optimizer,
                config
            )

            total_time = 0.0

            with mlflow.start_run(
                run_name=f"Task5_{experiment_name}"
            ):

                mlflow.log_params({
                    "optimizer": optimizer_name,
                    "scheduler": scheduler_name,
                    "initial_learning_rate": learning_rate,
                    "momentum": (
                        0.9 if optimizer_name == "SGD" else 0
                    ),
                    "epochs": epochs,
                    "batch_size": 64,
                    "num_workers": 0,
                    "seed": seed,
                    "device": "cpu",
                    "training_samples": training_size,
                    "validation_samples": validation_size
                })

                for epoch in range(1, epochs + 1):

                    current_lr = optimizer.param_groups[0]["lr"]

                    start = time.perf_counter()

                    train_loss, train_accuracy = train_one_epoch(
                        model,
                        train_loader,
                        criterion,
                        optimizer,
                        device
                    )

                    epoch_time = time.perf_counter() - start
                    total_time += epoch_time

                    val_loss, val_accuracy = evaluate(
                        model,
                        validation_loader,
                        criterion,
                        device
                    )

                    if scheduler_name == "ReduceLROnPlateau":
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                    metrics = {
                        "train_loss": train_loss,
                        "train_accuracy": train_accuracy,
                        "validation_loss": val_loss,
                        "validation_accuracy": val_accuracy,
                        "learning_rate": current_lr,
                        "epoch_time_seconds": epoch_time
                    }

                    mlflow.log_metrics(
                        metrics,
                        step=epoch
                    )

                    all_history.append({
                        "experiment": experiment_name,
                        "optimizer": optimizer_name,
                        "scheduler": scheduler_name,
                        "epoch": epoch,
                        **metrics
                    })

                    print(
                        f"{experiment_name} | "
                        f"Epoch {epoch}/{epochs} | "
                        f"Train Accuracy: {train_accuracy:.4f} | "
                        f"Validation Accuracy: {val_accuracy:.4f} | "
                        f"LR: {current_lr:.6f}",
                        flush=True
                    )

                test_loss, test_accuracy = evaluate(
                    model,
                    test_loader,
                    criterion,
                    device
                )

                final_result = {
                    "experiment": experiment_name,
                    "optimizer": optimizer_name,
                    "scheduler": scheduler_name,
                    "epochs": epochs,
                    "initial_learning_rate": learning_rate,
                    "final_train_accuracy": train_accuracy,
                    "final_validation_accuracy": val_accuracy,
                    "final_test_accuracy": test_accuracy,
                    "final_test_loss": test_loss,
                    "training_time_seconds": total_time
                }

                final_results.append(final_result)

                mlflow.log_metrics({
                    "final_validation_accuracy": val_accuracy,
                    "final_test_accuracy": test_accuracy,
                    "final_test_loss": test_loss,
                    "training_time_seconds": total_time
                })

    history_df = pd.DataFrame(all_history)
    results_df = pd.DataFrame(final_results)

    history_path = results_dir / "task5_epoch_history.csv"
    results_path = results_dir / "task5_optimizer_study.csv"

    history_df.to_csv(history_path, index=False)
    results_df.to_csv(results_path, index=False)

    print("\nOptimizer and Scheduler Results:")
    print(results_df.to_string(index=False))

    best = results_df.loc[
        results_df["final_validation_accuracy"].idxmax()
    ]

    print(f"\nBest combination: {best['experiment']}")
    print(
        f"Validation Accuracy: "
        f"{best['final_validation_accuracy']:.4f}"
    )
    print(
        f"Test Accuracy: "
        f"{best['final_test_accuracy']:.4f}"
    )

    plots = [
        (
            "validation_accuracy",
            "Validation Accuracy Comparison",
            "Validation Accuracy",
            "task5_accuracy.png"
        ),
        (
            "validation_loss",
            "Validation Loss Comparison",
            "Validation Loss",
            "task5_loss.png"
        ),
        (
            "learning_rate",
            "Learning Rate Schedule Comparison",
            "Learning Rate",
            "task5_learning_rate.png"
        )
    ]

    plot_paths = []

    for column, title, ylabel, filename in plots:

        plt.figure(figsize=(11, 6))

        for name, group in history_df.groupby("experiment"):

            plt.plot(
                group["epoch"],
                group[column],
                marker="o",
                label=name
            )

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.xticks(range(1, 11))
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        path = results_dir / filename

        plt.savefig(path, dpi=300)
        plt.close()

        plot_paths.append(path)

    with mlflow.start_run(
        run_name="Task5_Optimizer_Summary"
    ):

        mlflow.log_param(
            "best_combination",
            str(best["experiment"])
        )

        mlflow.log_metric(
            "best_validation_accuracy",
            float(best["final_validation_accuracy"])
        )

        mlflow.log_metric(
            "selected_model_test_accuracy",
            float(best["final_test_accuracy"])
        )

        mlflow.log_artifact(str(history_path))
        mlflow.log_artifact(str(results_path))

        for path in plot_paths:
            mlflow.log_artifact(str(path))

    print(
        "\nOptimizer and scheduler study completed successfully."
    )


if __name__ == "__main__":
    main()
