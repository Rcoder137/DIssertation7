import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np



# Placeholder transformations
train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
batch_size = 32  # Define a batch size

def create_filtered_dataset(allowed_classes, base_dir, subset, transform):
    datasets_list = []
    for cls in allowed_classes:
        class_subset_path = os.path.join(base_dir, cls, subset)  # e.g., 'crop_dataset/Bell Pepper/train'
        if not os.path.exists(class_subset_path):
            raise FileNotFoundError(f"No directory found for class {cls} in subset {subset} at {class_subset_path}")

        # Use ImageFolder directly to utilize the structure under each subset
        class_dataset = datasets.ImageFolder(class_subset_path, transform=transform)
        datasets_list.append(class_dataset)

    # Combine datasets from each class
    combined_dataset = ConcatDataset(datasets_list)
    
    return combined_dataset

extract_folder = r'C:\Users\robin\Projects\CropClassificationProject\Datasets\crop_dataset'
allowed_classes = ['Bell Pepper', 'Peach']
print(allowed_classes)

# Creating filtered datasets for each subset
train_dataset_filtered = create_filtered_dataset(allowed_classes, extract_folder, 'train', train_transform)
val_dataset_filtered = create_filtered_dataset(allowed_classes, extract_folder, 'val', val_transform)
test_dataset_filtered = create_filtered_dataset(allowed_classes, extract_folder, 'test', test_transform)

# Creating DataLoaders
train_loader_filtered = DataLoader(train_dataset_filtered, batch_size=batch_size, shuffle=True)
val_loader_filtered = DataLoader(val_dataset_filtered, batch_size=batch_size, shuffle=False)
test_loader_filtered = DataLoader(test_dataset_filtered, batch_size=batch_size, shuffle=False)

# Print classes from each dataset included in ConcatDataset
print("Training classes:", [train_dataset.classes for train_dataset in train_dataset_filtered.datasets])
print("Validation classes:", [val_dataset.classes for val_dataset in val_dataset_filtered.datasets])
print("Testing classes:", [test_dataset.classes for test_dataset in test_dataset_filtered.datasets])
  # Fetch and print dataset sizes
print(len(train_dataset_filtered), len(val_dataset_filtered), len(test_dataset_filtered))

train_loader = DataLoader(train_dataset_filtered, batch_size=batch_size, shuffle=True)

def imshow(img, ax, title=None):
    """Imshow for Tensor."""
    img = img.numpy().transpose((1, 2, 0))  # Convert tensor image to numpy and change from CxHxW to HxWxC
    mean = np.array([0.485, 0.456, 0.406])  # These should match the values used during normalization
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean  # Unnormalize
    img = np.clip(img, 0, 1)  # Clip values to ensure they are valid [0,1] pixels
    ax.imshow(img)
    ax.axis('off')  # Hide axes
    if title is not None:
        ax.set_title(title)

def show_images_batch(dataloader, num_images=24, images_per_row=6):
    images, _ = next(iter(dataloader))  # Get one batch of images
    num_rows = (num_images + images_per_row - 1) // images_per_row
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(images_per_row * 8, num_rows * 4))
    axes = axes.flatten()

    for i, img in enumerate(images):
        if i >= num_images:
            break
        imshow(img, axes[i])

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# Now call the function with your DataLoader
show_images_batch(train_loader)
