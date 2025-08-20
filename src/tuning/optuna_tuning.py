# optuna_tuning.py

import os
import torch
import optuna

from src.training.train import train_model, get_dataloaders, setup_model
from src.utils import file_paths

paths = file_paths()

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = paths["PROJECT_ROOT"]
DATA_DIR = paths["DATA_DIR"]
IMG_DIR = paths["IMG_DIR"]
OUTPUT_DIR = paths["OUTPUT_DIR"]

ANNOTATION_FILE = os.path.join(DATA_DIR, "train_noses.txt")
VALIDATION_FILE = os.path.join(DATA_DIR, "test_noses.txt")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "optuna_trials_results.txt")


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
        n_epochs=5,
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