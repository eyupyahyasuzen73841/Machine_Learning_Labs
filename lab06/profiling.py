from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from torch.profiler import profile, record_function, ProfilerActivity
from train import load_config, set_seed, get_dataloaders
from model import EyupLeNetVariant


def main():
    config = load_config()
    set_seed(config["project"]["seed"])

    device = torch.device("cpu")
    train_loader, _ = get_dataloaders(config)

    model = EyupLeNetVariant(
        num_classes=config["model"]["num_classes"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    stages = [
        "Data Loading",
        "Forward Pass",
        "Backward Pass",
        "Optimizer Step"
    ]

    timings = {stage: 0.0 for stage in stages}
    summaries = []

    total_batches = len(train_loader)
    chunk_size = 20
    completed_batches = 0

    model.train()
    iterator = iter(train_loader)

    print("Profiling one full training epoch...")
    print(f"Total batches: {total_batches}")

    while completed_batches < total_batches:

        current_chunk = min(
            chunk_size,
            total_batches - completed_batches
        )

        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=False,
            profile_memory=False,
            with_stack=False
        ) as prof:

            for _ in range(current_chunk):

                with record_function("Data Loading"):
                    images, labels = next(iterator)
                    images = images.to(device)
                    labels = labels.to(device)

                with record_function("Forward Pass"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                with record_function("Backward Pass"):
                    loss.backward()

                with record_function("Optimizer Step"):
                    optimizer.step()
                    optimizer.zero_grad()

        events = prof.key_averages()

        for event in events:
            if event.key in timings:
                timings[event.key] += (
                    event.cpu_time_total / 1_000_000
                )

        if completed_batches == 0:
            summaries.append(
                events.table(
                    sort_by="self_cpu_time_total",
                    row_limit=20
                )
            )

        del events
        del prof

        completed_batches += current_chunk

        print(
            f"Profiled {completed_batches}/{total_batches} batches",
            flush=True
        )

    total_time = sum(timings.values())

    rows = []

    for stage, seconds in timings.items():
        rows.append({
            "Stage": stage,
            "Time_seconds": seconds,
            "Percentage": seconds / total_time * 100
        })

    df = pd.DataFrame(rows)

    csv_path = results_dir / "task2_profiling.csv"
    df.to_csv(csv_path, index=False)

    print("\nProfiling Results:")
    print(df.to_string(index=False))

    summary_path = results_dir / "task2_profiler_summary.txt"

    summary_path.write_text(
        "Representative operator summary (first 20 batches)\n"
        + "\n".join(summaries)
        + "\n\nFull epoch stage breakdown:\n"
        + df.to_string(index=False),
        encoding="utf-8"
    )

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        df["Stage"],
        df["Percentage"],
        color=["steelblue", "orange", "seagreen", "purple"]
    )

    plt.bar_label(bars, fmt="%.2f%%", padding=3)
    plt.title("Training Time Breakdown - Full Epoch")
    plt.ylabel("Percentage of Measured Time (%)")
    plt.xticks(rotation=15)
    plt.ylim(0, max(df["Percentage"]) * 1.2)
    plt.tight_layout()

    plot_path = results_dir / "task2_profiling.png"

    plt.savefig(plot_path, dpi=300)
    plt.close()

    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"]
    )

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    with mlflow.start_run(run_name="Task2_Profiling"):

        mlflow.log_params({
            "device": "cpu",
            "batch_size": config["data"]["batch_size"],
            "num_workers": config["data"]["num_workers"],
            "profiled_epochs": 1,
            "profiled_batches": total_batches,
            "profiling_chunk_size": chunk_size
        })

        for row in rows:
            name = row["Stage"].lower().replace(" ", "_")

            mlflow.log_metric(
                f"{name}_seconds",
                row["Time_seconds"]
            )

            mlflow.log_metric(
                f"{name}_percentage",
                row["Percentage"]
            )

        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(plot_path))

    print("\nProfiling completed successfully.")


if __name__ == "__main__":
    main()
