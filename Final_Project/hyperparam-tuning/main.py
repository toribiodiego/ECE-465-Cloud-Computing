import os
import json
import ray
from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.air import RunConfig
from ray.tune.schedulers import ASHAScheduler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import wandb
from tqdm import tqdm
import time

# Load environment variables from the .env file
from dotenv import load_dotenv
load_dotenv()

# Global configuration dictionary; keys come exclusively from the environment.
config = {
    "wandb_api_key": os.getenv("WANDB_API_KEY"),
    "resource": {
        "total_cpus": 2,
        "num_tasks": 2,
        "cpus_per_task": 1,
        "gpus_per_task": 0
    },
    "tune_params": {
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
        "optimizer": tune.choice(["adam", "sgd"]),
        "layer_size": tune.randint(64, 256),
        "dropout_rate": tune.uniform(0.1, 0.5),
        "epochs": 3
    }
}

def setup_environment(config):
    """
    Initialize wandb. If no API key is set, switch to offline mode.
    """
    key = config.get("wandb_api_key")
    if key:
        wandb.login(key=key)
    else:
        print("WANDB_API_KEY not set; running in offline mode.")
        os.environ["WANDB_MODE"] = "offline"

class CustomModel(nn.Module):
    """
    A simple feed-forward neural network for MNIST classification.
    """
    def __init__(self, layer_size=128, dropout_rate=0.3, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        self.fc1 = nn.Linear(784, layer_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(layer_size, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def get_data_loaders(batch_size):
    """
    Load the MNIST dataset and return training and testing data loaders.
    """
    train_dataset = MNIST(root="data", train=True, transform=ToTensor(), download=True)
    test_dataset = MNIST(root="data", train=False, transform=ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    return train_loader, test_loader

def build_model_and_optimizer(tune_config, device):
    """
    Instantiate the model, select an optimizer, and define the loss criterion.
    """
    model = CustomModel(layer_size=tune_config["layer_size"], dropout_rate=tune_config["dropout_rate"])
    model.to(device)
    optimizer_cls = optim.Adam if tune_config["optimizer"] == "adam" else optim.SGD
    optimizer = optimizer_cls(model.parameters(), lr=tune_config["lr"])
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train the model for one epoch and return the average loss and accuracy.
    """
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    return total_loss / len(train_loader), total_correct / total_samples

def validate_model(model, test_loader, criterion, device):
    """
    Evaluate the model on the test data and return the average loss and accuracy.
    """
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    return total_loss / len(test_loader), total_correct / total_samples

def log_and_report(epoch, train_loss, train_accuracy, val_loss, val_accuracy):
    """
    Log training and validation metrics to wandb and report them to Ray Tune.
    """
    metrics = {
        "loss": train_loss,
        "accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "epoch": epoch + 1
    }
    wandb.log(metrics)
    tune.report(metrics)

def train_model(tune_config):
    """
    Main training loop that initializes wandb, prepares data, trains the model,
    and logs metrics at each epoch.
    """
    run = wandb.init(project="mnist_ray_tune", config=tune_config)
    train_loader, test_loader = get_data_loaders(tune_config["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer, criterion = build_model_and_optimizer(tune_config, device)
    for epoch in range(tune_config["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_model(model, test_loader, criterion, device)
        log_and_report(epoch, train_loss, train_acc, val_loss, val_acc)
    run.finish()

def initialize_ray(config):
    """
    Shutdown any existing Ray instance and initialize Ray with specified resource parameters.
    """
    ray.shutdown()
    ray.init(num_cpus=config["resource"]["total_cpus"],
             ignore_reinit_error=True,
             logging_level="ERROR",
             include_dashboard=True)

def run_ray_tune(cfg):
    """
    Set up Ray, create the Tune tuner, and execute the hyperparameter tuning run.
    """
    initialize_ray(cfg)
    tuner = Tuner(
        tune.with_resources(train_model, resources={
            "cpu": cfg["resource"]["cpus_per_task"],
            "gpu": cfg["resource"]["gpus_per_task"]
        }),
        param_space=cfg["tune_params"],
        tune_config=TuneConfig(num_samples=cfg["resource"]["num_tasks"], scheduler=None),
        run_config=RunConfig(name="mnist_ray_tune", verbose=0)
    )
    results = tuner.fit()
    timeline_data = ray.timeline()
    with open("timeline.json", "w") as f:
        f.write(json.dumps(timeline_data, indent=2))
    return results

def main():
    """
    Main entry point: set up the environment and run the Ray Tune experiment.
    """
    setup_environment(config)
    run_ray_tune(config)
    ray.shutdown()
    print("Tuning complete; entering idle loop so dashboard stays up on :8265")
    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()
