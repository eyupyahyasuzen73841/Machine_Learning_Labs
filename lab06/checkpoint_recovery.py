import sys
import subprocess
import random
from pathlib import Path

import numpy as np
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

CHECKPOINT_PATH = Path(
    "checkpoints/recovery_epoch_3.pt"
)

TOTAL_EPOCHS = 10
INTERRUPT_EPOCH = 3


def run_training(mode):

    config = load_config()

    config["data"]["batch_size"] = 64
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
        lr=config["training"]["learning_rate"]
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["scheduler"]["step_size"],
        gamma=config["scheduler"]["gamma"]
    )

    history = []

    start_epoch = 1
    end_epoch = TOTAL_EPOCHS

    if mode == "interrupt":

        end_epoch = INTERRUPT_EPOCH

    elif mode == "resume":

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        torch.set_rng_state(
            checkpoint["torch_rng_state"]
        )

        train_loader.generator.set_state(
            checkpoint["loader_rng_state"]
        )

        random.setstate(
            checkpoint["python_rng_state"]
        )

        np.random.set_state(
            checkpoint["numpy_rng_state"]
        )

        history = checkpoint["history"]

        start_epoch = checkpoint["epoch"] + 1

        print(
            f"Checkpoint loaded successfully.\n"
            f"Resuming training from epoch {start_epoch}.",
            flush=True
        )

    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"]
    )

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    RESULTS_DIR.mkdir(exist_ok=True)

    CHECKPOINT_PATH.parent.mkdir(
        exist_ok=True
    )

    with mlflow.start_run(
        run_name=f"Task7_{mode}"
    ):

        mlflow.log_params({
            "mode": mode,
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
            "batch_size": 64,
            "num_workers": 0,
            "optimizer": "Adam",
            "scheduler": "StepLR",
            "seed": seed,
            "device": "cpu"
        })

        for epoch in range(
            start_epoch,
            end_epoch + 1
        ):

            current_lr = optimizer.param_groups[0]["lr"]

            train_loss, train_accuracy = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )

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
                "learning_rate": current_lr
            }

            history.append({
                "epoch": epoch,
                **metrics
            })

            mlflow.log_metrics(
                metrics,
                step=epoch
            )

            print(
                f"{mode} | "
                f"Epoch {epoch}/{TOTAL_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Accuracy: {test_accuracy:.4f}",
                flush=True
            )

            if (
                mode == "interrupt"
                and epoch == INTERRUPT_EPOCH
            ):

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "loader_rng_state": (
                        train_loader.generator.get_state()
                    ),
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "history": history
                }, CHECKPOINT_PATH)

                mlflow.log_artifact(
                    str(CHECKPOINT_PATH)
                )

                print(
                    "\nCheckpoint saved successfully."
                )

                print(
                    "Simulated interruption after epoch 3."
                )

        history_path = (
            RESULTS_DIR / f"task7_{mode}_history.csv"
        )

        pd.DataFrame(history).to_csv(
            history_path,
            index=False
        )

        mlflow.log_artifact(
            str(history_path)
        )

        if mode in ["continuous", "resume"]:

            model_path = (
                RESULTS_DIR / f"task7_{mode}_model.pt"
            )

            torch.save(
                model.state_dict(),
                model_path
            )

            mlflow.log_artifact(
                str(model_path)
            )

            mlflow.log_metric(
                "final_test_accuracy",
                test_accuracy
            )


def main():

    RESULTS_DIR.mkdir(exist_ok=True)

    modes = [
        "continuous",
        "interrupt",
        "resume"
    ]

    for mode in modes:

        print(
            f"\nStarting {mode} training...",
            flush=True
        )

        subprocess.run(
            [sys.executable, __file__, mode],
            check=True
        )

    continuous = pd.read_csv(
        RESULTS_DIR / "task7_continuous_history.csv"
    )

    resumed = pd.read_csv(
        RESULTS_DIR / "task7_resume_history.csv"
    )

    comparison = continuous.merge(
        resumed,
        on="epoch",
        suffixes=("_continuous", "_resumed"),
        validate="one_to_one"
    )

    comparison["accuracy_difference"] = (
        comparison["test_accuracy_continuous"]
        - comparison["test_accuracy_resumed"]
    ).abs()

    comparison["loss_difference"] = (
        comparison["test_loss_continuous"]
        - comparison["test_loss_resumed"]
    ).abs()

    comparison_path = (
        RESULTS_DIR / "task7_checkpoint_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    max_accuracy_difference = comparison[
        "accuracy_difference"
    ].max()

    max_loss_difference = comparison[
        "loss_difference"
    ].max()

    continuous_weights = torch.load(
        RESULTS_DIR / "task7_continuous_model.pt",
        map_location="cpu",
        weights_only=True
    )

    resumed_weights = torch.load(
        RESULTS_DIR / "task7_resume_model.pt",
        map_location="cpu",
        weights_only=True
    )

    max_weight_difference = max(
        (
            continuous_weights[key]
            - resumed_weights[key]
        ).abs().max().item()
        for key in continuous_weights
    )

    print("\nCheckpoint Recovery Results:")

    print(comparison.to_string(index=False))

    print(
        f"\nContinuous final accuracy: "
        f"{continuous['test_accuracy'].iloc[-1]:.6f}"
    )

    print(
        f"Resumed final accuracy: "
        f"{resumed['test_accuracy'].iloc[-1]:.6f}"
    )

    print(
        f"Maximum accuracy difference: "
        f"{max_accuracy_difference:.10f}"
    )

    print(
        f"Maximum loss difference: "
        f"{max_loss_difference:.10f}"
    )

    print(
        f"Maximum model weight difference: "
        f"{max_weight_difference:.10f}"
    )

    plots = [
        (
            "test_accuracy",
            "Checkpoint Recovery - Test Accuracy",
            "Test Accuracy",
            "task7_accuracy.png"
        ),
        (
            "test_loss",
            "Checkpoint Recovery - Test Loss",
            "Test Loss",
            "task7_loss.png"
        ),
        (
            "train_loss",
            "Checkpoint Recovery - Training Loss",
            "Training Loss",
            "task7_train_loss.png"
        )
    ]

    plot_paths = []

    for column, title, ylabel, filename in plots:

        plt.figure(figsize=(10, 5))

        plt.plot(
            continuous["epoch"],
            continuous[column],
            marker="o",
            label="Continuous Training"
        )

        plt.plot(
            resumed["epoch"],
            resumed[column],
            marker="x",
            linestyle="--",
            label="Interrupted and Resumed"
        )

        plt.axvline(
            x=3,
            linestyle=":",
            color="red",
            label="Checkpoint at Epoch 3"
        )

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)

        plt.xticks(
            range(1, TOTAL_EPOCHS + 1)
        )

        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        path = RESULTS_DIR / filename

        plt.savefig(path, dpi=300)
        plt.close()

        plot_paths.append(path)

    config = load_config()

    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"]
    )

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    with mlflow.start_run(
        run_name="Task7_Recovery_Summary"
    ):

        mlflow.log_params({
            "interrupt_epoch": INTERRUPT_EPOCH,
            "resume_epoch": INTERRUPT_EPOCH + 1,
            "total_epochs": TOTAL_EPOCHS
        })

        mlflow.log_metrics({
            "continuous_final_accuracy": float(
                continuous["test_accuracy"].iloc[-1]
            ),
            "resumed_final_accuracy": float(
                resumed["test_accuracy"].iloc[-1]
            ),
            "max_accuracy_difference": float(
                max_accuracy_difference
            ),
            "max_loss_difference": float(
                max_loss_difference
            ),
            "max_weight_difference": float(
                max_weight_difference
            )
        })

        mlflow.log_artifact(
            str(comparison_path)
        )

        for path in plot_paths:

            mlflow.log_artifact(str(path))

    print(
        "\nCheckpoint recovery study completed successfully."
    )


if __name__ == "__main__":

    if len(sys.argv) == 2:

        run_training(sys.argv[1])

    else:

        main()
