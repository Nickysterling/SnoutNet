# optuna_tuning.py

import os
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import optuna

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model import SnoutNet
from training.custom_dataset import PetNoseDataset
from training.train import train_model, get_dataloaders, setup_model


# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ANNOTATION_FILE = os.path.join(PROJECT_ROOT, "data/train_noses.txt")
VALIDATION_FILE = os.path.join(PROJECT_ROOT, "data/test_noses.txt")
IMG_DIR = os.path.join(PROJECT_ROOT, "data/images")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "outputs", "optuna_trials_results.txt")


# ----------------------------
# Objective function
# ----------------------------
def objective(trial):
    # Suggest hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use the setup_model function from train.py
    model, optimizer, scheduler, loss_fn = setup_model(device=device, lr=lr, weight_decay=weight_decay)

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(batch_size)

    # Train model without saving or plotting
    val_loss = train_model(
        model, optimizer, scheduler, loss_fn,
        train_loader, val_loader,
        device=device,
        n_epochs=50,
        model_path=None,
        plot_path=None
    )

    return val_loss


# ----------------------------
# Logging callback
# ----------------------------
def log_trial_results(study, trial):
    with open(OUTPUT_FILE, 'a') as f:
        f.write(f"Trial {trial.number}:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Validation loss: {trial.value:.4f}\n\n")
    print(f"Trial {trial.number} logged with validation loss: {trial.value:.4f}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=25, callbacks=[log_trial_results])

    print("\nHyperparameter optimization completed!")
    print("Best hyperparameters:", study.best_params)
    print("Best validation loss:", study.best_value)