import logging
import os
import json
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torchvision.models as models
from download import load_and_split_data

import glob
import shutil
from sklearn.model_selection import train_test_split



output_dir = "/content/artifacts"
os.makedirs(output_dir, exist_ok=True)

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract the directory path from the log file path
log_dir = os.path.dirname(config["logging"]["file"])

# Create the directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=config["logging"]["level"],
    format=config["logging"]["format"],
    handlers=[
        logging.FileHandler(config["logging"]["file"]),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Type hints
Dataset = Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]
ModelOutput = Tuple[nn.Module, optim.Optimizer, nn.CrossEntropyLoss]

# Create the directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

from torch.utils.data import random_split

def split_into_batches(dataset: Dataset) -> list[Dataset]:
    """Split a dataset into batches.

    Args:
        dataset (Dataset): The dataset to split.

    Returns:
        List[Dataset]: A list of batches as subsets of the original dataset.
    """
    num_batches = config["data"]["num_batches"]
    batch_sizes = [len(dataset) // num_batches] * num_batches
    batch_sizes[-1] += len(dataset) % num_batches  # Adjust the last batch size if dataset length is not divisible
    return random_split(dataset, batch_sizes, generator=torch.Generator().manual_seed(config["dataset"]["random_seed"]))


def create_model() -> ModelOutput:
    """Create the model, optimizer, and loss function."""
    # Create your model
    model = models.resnet50(pretrained=config["model"]["pretrained"])
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config["model"]["num_classes"])

    
    # Freeze base layers if specified
    if config["model"].get("freeze_base", False):
        for param in model.parameters():
            param.requires_grad = False

    # Create the optimizer
    optimizer_config = config["training"]["optimizer"]
    optimizer = getattr(optim, optimizer_config["name"])(
        model.parameters(), lr=optimizer_config["lr"]
    )

    # Create the loss function
    loss_fn = getattr(nn, config["training"]["loss"])()

    return model, optimizer, loss_fn


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.CrossEntropyLoss,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_batches: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    """Train the model and evaluate on the validation set.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        optimizer (optim.Optimizer): The optimizer for the model.
        loss_fn (nn.CrossEntropyLoss): The loss function.
        train_loader (DataLoader): The data loader for the training set.
        val_loader (DataLoader): The data loader for the validation set.
        device (str, optional): The device to use for training (CPU or GPU). Defaults to "cuda" if available, else "cpu".

    Returns:
        Dict[str, float]: A dictionary containing the evaluation metrics on the validation set.
    """
    model.to(device)
    best_accuracy = 0.0
    test_best_accuracy = 0.0

    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = False

            for param in model.parameters():
                param.requires_grad = True
                
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        # Evaluate on test set
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        train_accuracy = 100.0 * train_correct / train_total
        val_accuracy = 100.0 * val_correct / val_total
        test_accuracy = 100.0 * test_correct / test_total

        logger.info(f"Batch {num_batches}: Epoch {epoch+1}: Train Loss={train_loss/train_total:.4f}, Train Accuracy={train_accuracy:.2f}%")
        logger.info(f"Batch {num_batches}: Epoch {epoch+1}: Val Loss={val_loss/val_total:.4f}, Val Accuracy={val_accuracy:.2f}%")
        logger.info(f"Batch {num_batches}: Epoch {epoch+1}: Test Loss={test_loss/test_total:.4f}, Test Accuracy={test_accuracy:.2f}%")

                
        if test_accuracy > test_best_accuracy:
            test_best_accuracy = test_accuracy
            if config["artifacts"]["save_best_model"]:
                torch.save(model.state_dict(), os.path.join(config["artifacts"]["output_dir"], "best_model.pth"))

    return {"test_accuracy": test_best_accuracy / 100.0}


from sklearn.model_selection import train_test_split

def split_batch_into_train_val(batch_dataset):
    """Split a batch dataset into train and validation sets.

    Args:
        batch_dataset (torch.utils.data.Dataset): The batch dataset to split.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: The train and validation datasets.
    """
    train_dataset, val_dataset = train_test_split(
        batch_dataset,
        test_size=config["dataset"]["val_split"],
        random_state=config["dataset"]["random_seed"]
    )
    return train_dataset, val_dataset


def main() -> None:
    output_dir = config["artifacts"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    trainval_dataset, test_dataset = load_and_split_data(config["data"]["local_dir"])

    batches = split_into_batches(trainval_dataset)

    for num_batches in range(1, len(batches) + 1):
        combined_dataset = torch.utils.data.ConcatDataset(batches[:num_batches])
        train_dataset, val_dataset = split_batch_into_train_val(combined_dataset)

        train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"])
        test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"])

        # Create the model, optimizer, and loss function
        model, optimizer, loss_fn = create_model()

        # Train the model
        metrics = train(model, optimizer, loss_fn, train_loader, val_loader, test_loader, num_batches)
        logger.info(f"Validation metrics for {num_batches} batches: {metrics}")

        # Save artifacts
        os.makedirs(config["artifacts"]["output_dir"], exist_ok=True)
        if config["artifacts"]["save_best_model"]:
            torch.save(model.state_dict(), os.path.join(config["artifacts"]["output_dir"], f"best_model_{num_batches}.pth"))
        if config["artifacts"]["save_logs"]:
            # Save logs
            with open(os.path.join(config["artifacts"]["output_dir"], f"training_{num_batches}.log"), "w") as log_file:
                for handler in logger.handlers:
                    if isinstance(handler, logging.FileHandler):
                        log_file.write(handler.stream.getvalue())
        if config["artifacts"]["save_metrics"]:
            # Save metrics
            with open(os.path.join(config["artifacts"]["output_dir"], f"metrics_{num_batches}.json"), "w") as metrics_file:
                json.dump(metrics, metrics_file)
    


if __name__ == "__main__":
    main()
