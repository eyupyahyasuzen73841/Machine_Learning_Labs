import random
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import EyupLeNetVariant


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def get_dataloaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(
        root=config["data"]["root"],
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root=config["data"]["root"],
        train=False,
        download=True,
        transform=transform
    )

    generator = torch.Generator()
    generator.manual_seed(config["project"]["seed"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        generator=generator
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"]
    )

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    config = load_config()
    set_seed(config["project"]["seed"])

    device = torch.device(config["training"]["device"])

    print("Starting Lab 6 training...")
    print(f"Device: {device}")

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

    checkpoint_dir = Path(config["checkpoint"]["directory"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"]
    )

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    with mlflow.start_run(run_name="CNN_Baseline"):

        mlflow.log_params({
            "model": config["model"]["name"],
            "dataset": config["data"]["dataset"],
            "seed": config["project"]["seed"],
            "batch_size": config["data"]["batch_size"],
            "num_workers": config["data"]["num_workers"],
            "epochs": config["training"]["epochs"],
            "learning_rate": config["training"]["learning_rate"],
            "optimizer": config["training"]["optimizer"],
            "scheduler": config["scheduler"]["name"],
            "step_size": config["scheduler"]["step_size"],
            "gamma": config["scheduler"]["gamma"],
            "device": str(device)
        })

        mlflow.log_artifact("config.yaml")

        for epoch in range(1, config["training"]["epochs"] + 1):

            current_lr = optimizer.param_groups[0]["lr"]

            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )

            test_loss, test_accuracy = evaluate(
                model, test_loader, criterion, device
            )

            scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "learning_rate": current_lr
            }, step=epoch)

            print(
                f"Epoch {epoch}/{config['training']['epochs']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Accuracy: {train_accuracy:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Accuracy: {test_accuracy:.4f}"
            )

            if (
                config["checkpoint"]["enabled"]
                and epoch % config["checkpoint"]["save_every"] == 0
            ):
                checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "loader_rng_state": train_loader.generator.get_state()
                }, checkpoint_path)

        final_model_path = checkpoint_dir / "final_model.pt"

        torch.save(model.state_dict(), final_model_path)
        mlflow.log_artifact(str(final_model_path))

        print("Training completed successfully.")


if __name__ == "__main__":
    main()
