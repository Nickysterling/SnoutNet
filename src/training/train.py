# train.py

import os
import time
import datetime
import argparse

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import ConcatDataset
from torchsummary import summary

from src.model.model_arch import SnoutNet 
from src.training.custom_dataset import PetNoseDataset
from src.utils import file_paths


paths = file_paths()

# -----------------------------
# Default configuration
# -----------------------------
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 0.0005
DEFAULT_WEIGHT_DECAY = 0.00014651200567136424

PROJECT_ROOT = paths["PROJECT_ROOT"]
DATA_DIR = paths["DATA_DIR"]
IMG_DIR = paths["IMG_DIR"]
OUTPUT_DIR = paths["OUTPUT_DIR"]
MODEL_DIR = paths["MODEL_DIR"]
PLOT_DIR = paths["PLOT_DIR"]

ANNOTATION_FILE = os.path.join(DATA_DIR, "train_noses.txt")
VALIDATION_FILE = os.path.join(DATA_DIR, "test_noses.txt")

# ------------------------------
# Utility Functions
# ------------------------------

# Initialize model weights with Xavier uniform
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


# Initialize model, optimizer, scheduler, and loss function
def setup_model(device='cpu', lr=0.001, weight_decay=0.0):
    model = SnoutNet()
    model.to(device)
    model.apply(init_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_fn = nn.MSELoss()
    
    return model, optimizer, scheduler, loss_fn


# Create training and validation datasets with optional augmentations
def get_dataloaders(batch_size, hflip=False, color=False):
    # Train dataset
    datasets = [PetNoseDataset(annotations_file=ANNOTATION_FILE, img_dir=IMG_DIR)]
    
    # Horizontal flip and colour jitter augmentations
    if hflip:
        datasets.append(PetNoseDataset(annotations_file=ANNOTATION_FILE, img_dir=IMG_DIR, apply_flip=True))
    if color:
        datasets.append(PetNoseDataset(annotations_file=ANNOTATION_FILE, img_dir=IMG_DIR, apply_color_jitter=True))
    
    # Complete train and validation datasets
    train_dataset = ConcatDataset(datasets)
    val_dataset = PetNoseDataset(annotations_file=VALIDATION_FILE, img_dir=IMG_DIR)
    
    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


# Compute Euclidean distance between predicted and ground truth coordinates
def calculate_euclidean_distance(predictions, ground_truths):
    predictions = predictions.to(ground_truths.device)
    
    # Compute the squared differences
    diff = predictions - ground_truths
    squared_diff = diff ** 2

    # Sum the squared differences and then take the square root
    distance = torch.sqrt(torch.sum(squared_diff, dim=1))
    
    return distance


# Compute localization error stats and timing info for an epoch
def get_localization_stats(distances):
    distances_np = np.array(distances)

    return {
        'mean': distances_np.mean(),
        'min': distances_np.min(),
        'max': distances_np.max(),
        'std': distances_np.std(),
    }


# Plot training and validation losses with localization statistics.
def plot_losses(train_losses, val_losses, total_val_images, total_training_time, localization_stats, plot_path):
    plt.figure(figsize=(12, 7))
    plt.clf()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=1)

    minutes, seconds = divmod(total_training_time, 60)
    avg_time_per_image = (total_training_time / total_val_images) * 1000

    stats_text = (
        f"Total Runtime: {int(minutes)} min, {int(seconds)} sec\n"
        f"Avg Time/Test Image: {avg_time_per_image:.2f} ms, Avg Loss: {train_losses[-1]:.4f}\n"
        f"Localization Error: ["
        f"Mean: {localization_stats[-1]['mean']:.2f}, "
        f"Min: {localization_stats[-1]['min']:.2f}, "
        f"Max: {localization_stats[-1]['max']:.2f}, "
        f"Std: {localization_stats[-1]['std']:.2f}]"
    )

    plt.figtext(0.5, -0.05, stats_text, wrap=True, horizontalalignment='center', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.5))
    
    # relative_path = os.path.relpath(plot_path, start=PROJECT_ROOT)
    plt.title(f"Training Results\nLoss Plot: {plot_path}")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()


# ------------------------------
# Training Loop
# ------------------------------

# Train the model and optionally save checkpoints and plots
def train_model(model, optimizer, scheduler, loss_fn, train_loader, val_loader,
                device='cpu', n_epochs=30, model_path=None, plot_path=None):
    
    # Lists to hold average training and validation losses for each epoch
    train_losses = []
    val_losses = []

    # Lists to hold localization error statistics (mean, min, max, std) for each epoch
    localization_stats = []

    # Start timer for the whole training process
    training_start_time = time.time()

    # Loop through the number of epochs
    for epoch in range(1, n_epochs + 1):
        timestamp = datetime.datetime.now().strftime("%I:%M:%S %p")
        print(f"\n[{timestamp}] Epoch {epoch}/{n_epochs}")

        # Epoch variables
        running_train_loss = 0.0  # Accumulates training loss
        running_val_loss = 0.0    # Accumulates validation loss
        distances = []            # Store the Euclidean distances for the current epoch

        # Training loop
        model.train()
        for data in train_loader:
            img, lbl = data
            img = img.float().to(device=device)
            lbl = lbl.float().to(device=device)
            
            optimizer.zero_grad()         # Clear previous gradients
            outputs = model(img)          # Forward pass
            loss = loss_fn(outputs, lbl)  # Compute loss
            loss.backward()               # Backward pass (compute gradients)
            optimizer.step()              # Update model parameters based on gradients
            
            running_train_loss += loss.item()  # Accumulate training loss

        # Validation loop
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                img, lbl = data
                img = img.float().to(device=device)
                lbl = lbl.float().to(device=device)

                outputs = model(img)             # Forward pass      
                loss = loss_fn(outputs, lbl)     # Compute validation loss
                running_val_loss += loss.item()  # Accumulate validation loss

                # Calculate Euclidean distances between the predictions and the ground truth
                distances += calculate_euclidean_distance(outputs, lbl).cpu().tolist()
        
        # End timer
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time

        # Store average train and validation losses for this epoch
        train_losses += [running_train_loss / len(train_loader)]
        val_losses += [running_val_loss / len(val_loader)]
        
        stats = get_localization_stats(distances)
        localization_stats.append(stats)

        timestamp = datetime.datetime.now().strftime("%I:%M:%S %p")

        print(
            f"[{timestamp}] "
            f"Epoch {epoch}/{n_epochs} | "
            f"Train Loss: {train_losses[-1]:.4f} | "
            f"Val Loss: {val_losses[-1]:.4f} | "
            f"Localization Error: ["
            f"Mean: {localization_stats[-1]['mean']:.2f}, "
            f"Min: {localization_stats[-1]['min']:.2f}, "
            f"Max: {localization_stats[-1]['max']:.2f}, "
            f"Std: {localization_stats[-1]['std']:.2f}]"
        )

        # Update the scheduler based on training loss
        scheduler.step(val_losses[-1])

        if model_path is not None:
            # Update model
            torch.save(model.state_dict(), model_path)

        total_val_images = len(val_loader.dataset)

        if plot_path is not None:
            # Update plot
            plot_losses(train_losses, val_losses, total_val_images, total_training_time, localization_stats, plot_path)
    
    return val_losses[-1]


# ------------------------------
# Command-line Argument Parsing
# ------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train SnoutNet model for PetNoseDataset")
    
    parser.add_argument('-e', type=int, default=DEFAULT_EPOCHS, help="Number of epochs to train")
    parser.add_argument('-b', type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument('-lr', type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument('-wd', type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay")
    
    parser.add_argument('-hflip', action='store_true', help='Apply horizontal flip transformation')
    parser.add_argument('-color', action='store_true', help='Apply color jitter transformation')
    
    return parser.parse_args()


# ------------------------------
# Main
# ------------------------------

def main():
    args = parse_args()
    
    print(f"\033[1mRunning main with the following parameters:\033[0m\n"
          f"Epochs: {args.e}, Batch: {args.b}, LR: {args.lr}, WD: {args.wd}\n"
          f"Horizontal Flip: {args.hflip}, Color Jitter: {args.color}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running Model using Device: ', device)
    
    # Load model
    model, optimizer, scheduler, loss_fn = setup_model(device=device, lr=args.lr, weight_decay=args.wd)
    
    # Model summary
    summary(model, input_size=(3, 227, 227))
    
    # Load data
    train_loader, val_loader = get_dataloaders(args.b, args.hflip, args.color)
    
    # File paths
    suffix = ""
    if args.hflip:
        suffix += "_hflip"
    if args.color:
        suffix += "_colorj"

    model_path = os.path.join(MODEL_DIR, f'snoutnet_E{args.e}_B{args.b}{suffix}.pth')
    plot_path = os.path.join(PLOT_DIR, f'loss_E{args.e}_B{args.b}{suffix}.png')
    
    # Train
    train_model(model, optimizer, scheduler, loss_fn,
                train_loader, val_loader,
                device=device, n_epochs=args.e,
                model_path=model_path, plot_path=plot_path)

if __name__ == '__main__':
    main()
