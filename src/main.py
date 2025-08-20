# main.py

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from src.model.model_arch import SnoutNet 
from src.training.custom_dataset import PetNoseDataset
from src.training.train import calculate_euclidean_distance
from src.utils import file_paths

paths = file_paths()

PROJECT_ROOT = paths["PROJECT_ROOT"]
DATA_DIR = paths["DATA_DIR"]
IMG_DIR = paths["IMG_DIR"]

ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "archive")
VALIDATION_FILE = os.path.join(DATA_DIR, "test_noses.txt") 

# Function to visualize predictions vs ground truth using matplotlib
def visualize_predictions(image, pred_nose, gt_nose, distance, mean_error):
    image = image.permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array

    # Extract coordinates for ground truth and predicted points
    gt_x, gt_y = gt_nose
    pred_x, pred_y = pred_nose

    # Print the ground truth and predicted coordinates
    print(f"Ground Truth Coordinates: (x: {gt_x:.2f}, y: {gt_y:.2f})")
    print(f"Predicted Coordinates:   (x: {pred_x:.2f}, y: {pred_y:.2f})")
    print(f"Euclidean Distance (Localization Error) for this image: {distance:.2f} pixels")

    # Compare the current error with the mean error for the test set
    if distance > mean_error:
        print(f"This error is higher than the average localization error ({mean_error:.2f} pixels).")
    elif distance < mean_error:
        print(f"This error is lower than the average localization error ({mean_error:.2f} pixels).")
    else:
        print(f"This error is equal to the average localization error ({mean_error:.2f} pixels).")

    # Plot the image
    plt.imshow(image)
    
    # Plot ground truth in green
    plt.scatter([gt_x], [gt_y], color='green', label='Ground Truth')
    
    # Plot predicted in red
    plt.scatter([pred_x], [pred_y], color='red', label='Prediction')

    # Add text annotations above the image using figtext
    plt.figtext(0.5, 0.98, f"Ground Truth Coordinates: (x: {gt_x:.2f}, y: {gt_y:.2f})", ha='center', fontsize=10, color='green')
    plt.figtext(0.5, 0.94, f"Predicted Coordinates: (x: {pred_x:.2f}, y: {pred_y:.2f})", ha='center', fontsize=10, color='red')
    plt.figtext(0.5, 0.90, f"Euclidean Distance (Localization Error): {distance:.2f} pixels", ha='center', fontsize=10)

    # Show legend and image
    plt.legend()
    plt.show()

# Function to plot histogram of localization errors
def plot_error_histogram(all_errors):
    all_error_values = [e[0] for e in all_errors]
    plt.figure(figsize=(10, 6))

    # Create the histogram
    plt.hist(all_error_values, bins=50, color='gray', alpha=0.9, edgecolor='black', linewidth=1.2)

    # Titles and labels
    plt.title('Prediction Errors Distribution', fontsize=16)
    plt.xlabel('Distance (Error in pixels)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    # Grid
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Show
    plt.tight_layout()
    plt.show()

# Function to compute localization errors for the entire test set
def compute_overall_errors(model, val_loader, device):
    model.eval()
    errors = []

    with torch.no_grad():
        for index, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass to get predictions
            outputs = model(images)
            
            # Calculate Euclidean distance error for each image
            for pred, gt in zip(outputs, labels):
                distance = calculate_euclidean_distance(pred.unsqueeze(0), gt.unsqueeze(0))
                errors.append((distance.item(), index))

    # Sort errors based on the localization error
    errors_sorted = sorted(errors, key=lambda x: x[0])

    # Get the 10 best and 10 worst images
    best_images = errors_sorted[:10]
    worst_images = errors_sorted[-10:]

    # Calculate overall localization error statistics
    errors_np = np.array([e[0] for e in errors])
    min_error = errors_np.min()
    max_error = errors_np.max()
    mean_error = errors_np.mean()
    std_error = errors_np.std()

    print(f"\nLocalization Error Statistics for Test Set:")
    print(f"Mean Error: {mean_error:.2f} pixels")
    print(f"Min Error: {min_error:.2f} pixels")
    print(f"Max Error: {max_error:.2f} pixels")
    print(f"Standard Deviation: {std_error:.2f} pixels")

    # Print the indices of the 10 best and worst images with their respective errors
    print("\nTop 10 images with best localization accuracy (smallest errors):")
    for error, idx in best_images:
        print(f"Index: {idx}, Error: {error:.2f} pixels")

    print("\nTop 10 images with worst localization accuracy (largest errors):")
    for error, idx in worst_images:
        print(f"Index: {idx}, Error: {error:.2f} pixels")

    return min_error, max_error, mean_error, std_error, errors, best_images, worst_images

# Function to test the model and allow user to view a specific image by index
def test_model(model, val_loader, device, overall_mean_error):
    model.eval()

    # Get the total number of test images
    total_images = len(val_loader.dataset)

    # Prompt the user for an image index to view
    while True:
        try:
            index = int(input(f"\nEnter an image index (0 to {total_images - 1}) to view: "))
            if index < 0 or index >= total_images:
                raise ValueError("Index out of range.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")

    # Load the specific image and label using the index
    with torch.no_grad():
        images, labels = val_loader.dataset[index]
        images = images.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)

        # Forward pass to get predictions
        outputs = model(images)

        # Calculate Euclidean distance for this image
        pred = outputs.squeeze(0)
        gt = labels.squeeze(0)
        distance = calculate_euclidean_distance(pred.unsqueeze(0), gt.unsqueeze(0)).item()

        # Visualize the predictions and compare the error with overall statistics
        visualize_predictions(images.squeeze(0), pred.cpu().numpy(), gt.cpu().numpy(), distance, overall_mean_error)

def interactive_loop(model, val_loader, device, mean_error):
    print("\nYou can now enter a specific image index to view and compare its localization error with the overall test set.")
    while True:
        test_model(model, val_loader, device, mean_error)
        while True:
            user_input = input("\nPress enter to test another image or type 'quit' or 'exit' to stop: ").strip().lower()
            if user_input in ('quit', 'exit'):
                print("Exiting the program.")
                return
            elif user_input == '':
                break
            else:
                print("Invalid input. Please press enter to continue or type 'quit' or 'exit' to stop.")

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Test SnoutNet model for PetNoseDataset")
    parser.add_argument('-mp', type=str, default=None, help="Path to the trained model weights")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Select model
    model_path = os.path.abspath(os.path.join(PROJECT_ROOT, args.mp)
    ) if args.mp else os.path.join(ARCHIVE_DIR, "model_weights", "snoutnet_weights_E30_B16_hflip_colorj.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SnoutNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    val_dataset = PetNoseDataset(annotations_file=VALIDATION_FILE, img_dir=IMG_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print("Calculating overall localization error statistics for the test set...")
    min_error, max_error, mean_error, std_error, errors, best_images, worst_images = compute_overall_errors(model, val_loader, device)

    plot_error_histogram(errors)
    interactive_loop(model, val_loader, device, mean_error)


if __name__ == '__main__':
    main()