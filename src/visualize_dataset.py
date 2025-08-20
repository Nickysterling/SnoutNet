# visualize_dataset.py

import os
import matplotlib.pyplot as plt
from torchvision import transforms
from src.training.custom_dataset import PetNoseDataset
from src.utils import file_paths

paths = file_paths()

PROJECT_ROOT = paths["PROJECT_ROOT"]
DATA_DIR = paths["DATA_DIR"]
IMG_DIR = paths["IMG_DIR"]

ANNOTATION_FILE = os.path.join(DATA_DIR, "train_noses.txt")
VALIDATION_FILE = os.path.join(DATA_DIR, "test_noses.txt")

# Function to plot an image with annotation and a title
def plot_image(image_tensor, annotation, index):
    image_pil = transforms.ToPILImage()(image_tensor)
    plt.imshow(image_pil)
    plt.scatter(annotation[0], annotation[1], color='red', s=50, marker='x')
    plt.title(f'Sample {index+1} (Nose Coordinates: {annotation[0]:.1f}, {annotation[1]:.1f})')
    plt.axis('on')
    plt.show()

# Function to display images by single index or range
def display_images(dataset, start=None, end=None, idx=None):
    if idx is not None:
        rows_to_display = [(idx, dataset[idx])]
    elif start is not None and end is not None:
        rows_to_display = [(i, dataset[i]) for i in range(start, end)]
    else:
        raise ValueError("Specify either idx or start and end.")

    for i, (image, label) in rows_to_display:
        plot_image(image, label, i)


def ask_choice(prompt, options, quit_keys=("q",)):
    while True:
        choice = input(prompt).strip().lower()
        if choice in quit_keys:
            return None
        if choice in options:
            return options[choice]
        print(f"Invalid choice. Options: {', '.join(options.keys())}")


def main():
    while True:
        # Choose dataset
        annotations_file = ask_choice(
            "\nWould you like to view the train or val dataset? (t/v) or 'q' to quit: ",
            {"t": ANNOTATION_FILE, "v": VALIDATION_FILE}
        )
        if annotations_file is None:
            print("Exiting visualization.")
            break

        # Flip transform
        flip = ask_choice(
            "\nApply horizontal flip? (y/n) or 'q' to quit: ",
            {"y": True, "n": False}
        )
        if flip is None:
            print("Exiting visualization.")
            break

        # Colour jitter transform
        colour = ask_choice(
            "\nApply colour jitter? (y/n) or 'q' to quit: ",
            {"y": True, "n": False}
        )
        if colour is None:
            print("Exiting visualization.")
            break

        # Build dataset
        dataset = PetNoseDataset(
            annotations_file=annotations_file,
            img_dir=IMG_DIR,
            apply_flip=flip,
            apply_color_jitter=colour
        )

        # Choose viewing mode
        while True:
            mode = ask_choice(
                "\nView a single image or a range? (s/r) or 'b' to go back: ",
                {"s": "single", "r": "range"},
                quit_keys=("b",)
            )
            if mode is None:
                break

            if mode == "single":
                idx = int(input(f"Enter index (0 to {len(dataset)-1}): "))
                display_images(dataset, idx=idx)
            elif mode == "range":
                start = int(input(f"Start index (0 to {len(dataset)-1}): "))
                end = int(input(f"End index (exclusive ≤ {len(dataset)}): "))
                display_images(dataset, start=start, end=end)

if __name__ == "__main__":
    main()