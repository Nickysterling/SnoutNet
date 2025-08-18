# visualize_dataset.py

import os
import matplotlib.pyplot as plt
from torchvision import transforms
from training.custom_dataset import PetNoseDataset

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

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    img_dir = os.path.join(project_root, "data/images")

    while True:
        dataset_choice = input("\nWhich dataset do you want to visualize? (train/val) or 'q' to quit: ").strip().lower()
        if dataset_choice == 'q':
            print("Exiting visualization.")
            break

        if dataset_choice == 'train':
            annotations_file = os.path.join(project_root, "data/train_noses.txt")
        elif dataset_choice == 'val':
            annotations_file = os.path.join(project_root, "data/test_noses.txt")
        else:
            print("Invalid choice. Please type 'train', 'val', or 'q'.")
            continue

        dataset = PetNoseDataset(
            annotations_file=annotations_file,
            img_dir=img_dir,
            apply_flip=True,
            apply_color_jitter=False
        )

        while True:
            choice = input("\nDo you want to view a single image or a range? (single/range) or 'b' to go back: ").strip().lower()
            if choice == 'b':
                break
            if choice == 'single':
                idx = int(input(f"Enter the index of the image (0 to {len(dataset)-1}): "))
                display_images(dataset, idx=idx)
            elif choice == 'range':
                start = int(input(f"Enter the start index (0 to {len(dataset)-1}): "))
                end = int(input(f"Enter the end index (exclusive, up to {len(dataset)}): "))
                display_images(dataset, start=start, end=end)
            else:
                print("Invalid choice. Please type 'single', 'range', or 'b'.")
