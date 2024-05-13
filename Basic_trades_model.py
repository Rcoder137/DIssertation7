import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import sys
import torch.optim as optim
import matplotlib.pyplot as plt 
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet18, ResNet18_Weights


def trades_loss(model, x_nat, y, optimizer, device, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=6.0):
    model.eval()  # Set the model to evaluation mode for generating adversarial examples
    x_adv = x_nat.detach() + 0.001 * torch.randn(x_nat.shape).to(device)  # Initial adversarial perturbation
    x_adv.requires_grad_()

    for _ in range(perturb_steps):
        with torch.enable_grad():
            outputs_adv = model(x_adv)
            loss_kl = F.kl_div(F.log_softmax(outputs_adv, dim=1), F.softmax(model(x_nat), dim=1), reduction='batchmean')
            loss_kl.backward()  # Compute gradients based on KL divergence
        grad = x_adv.grad.data
        x_adv = x_adv.detach() + step_size * torch.sign(grad)  # Gradient ascent step
        x_adv = torch.min(torch.max(x_adv, x_nat - epsilon), x_nat + epsilon)  # Clip to epsilon neighborhood
        x_adv = torch.clamp(x_adv, 0.0, 1.0)  # Additional clipping to valid pixel range
        x_adv.requires_grad_()  # Re-enable gradient computation for next iteration

    model.train()  # Switch back to training mode for the rest of the training process

    # Compute final losses for natural and adversarial examples
    outputs_nat = model(x_nat)
    outputs_adv = model(x_adv)
    loss_natural = F.cross_entropy(outputs_nat, y)
    loss_robust = F.kl_div(F.log_softmax(outputs_adv, dim=1), F.softmax(outputs_nat, dim=1), reduction='batchmean')

    # Calculate TRADES loss: sum of natural and robustness losses
    total_loss = loss_natural + beta * loss_robust
    return total_loss




base_dir = "C:/Users/robin/Projects/CropClassificationProject/Datasets/crop_dataset"
image_size = 224
batch_size = 32  # Set the batch size you want to u


# Define transformations for the training, validation, and testing sets
def train_transforms():
    return transforms.Compose([
           transforms.Resize((256, 256)),  # Resize to a square of 256x256 pixels
           transforms.RandomRotation(10),  # Random rotation of images
           transforms.RandomHorizontalFlip(),  # Random horizontal flip
           transforms.ToTensor(),  # Convert images to tensors
           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats  # Standard normalization for ImageNet
    ])

def val_transforms():
    return transforms.Compose([
           transforms.Resize((image_size, image_size)),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
def load_datasets(base_dir, batch_size):
    train_transform = train_transforms()
    val_transform = val_transforms()
 
    # Paths to the dataset directories
    train_path = os.path.join(base_dir, 'Bell Pepper', 'Train')
    val_path = os.path.join(base_dir, 'Bell Pepper', 'Val')
    
    # Create datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
    return train_loader, val_loader



def main():

    # Register hook
    activations = {}
    def get_activation(name):
        """A function to create a forward hook function to capture activations."""
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
        
    train_loader, val_loader = load_datasets(base_dir, batch_size)
    # Load a pre-trained ResNet18 model
    weights = ResNet18_Weights.DEFAULT                       # Use DEFAULT for the most up-to-date weights
    model = resnet18(weights=weights)
    num_classes = 2                                          # We have 2 classes (Bacterial Spot and Healthy) for the Bell Pepper 
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the final fully connected layer of the model to match the number of classes
        
    model.layer4.register_forward_hook(get_activation('layer4'))
    
    # Freeze all layers except the final layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True    # Unfreeze the last layer
    
    
    # Check if CUDA is available and move the model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize the optimizer and scheduler
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)   # Example: Decay LR every 5 epochs
    
    # Training and evaluation loop
    num_epochs = 10
    epochs_no_improve = 0  # Initialize epochs_no_improve
    n_epochs_stop = 5  # Number of epochs to stop after no improvement
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float('inf')
    
    # Parameters for the TRADES loss
    step_size = 0.003
    epsilon = 0.002
    perturb_steps = 10
    beta = 6.0
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        model.train()  # Set model to training mode
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            features = activations['layer4']
            loss = trades_loss(model, images, labels, optimizer, device, step_size, epsilon, perturb_steps, beta)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
        train_losses.append(total_loss / total_samples)
        train_accuracies.append(100 * total_correct / total_samples)
        
        # Evaluation on the validation set
        
        model.eval()  # Set model to evaluation mode
        validation_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                validation_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        validation_loss /= total
        accuracy = 100 * correct / total
        val_losses.append(validation_loss)
        val_accuracies.append(accuracy)
        
        scheduler.step()  # Adjust learning rate based on scheduler
        
        
       # Check if there was an improvement
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            epochs_no_improve = 0
            print(f"New best validation loss: {best_val_loss}")
        else:
            epochs_no_improve += 1
        print(f'Epoch {epoch+1}, Validation Loss: {validation_loss:.4f}, Accuracy: {accuracy:.2f}%')
        print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}, Val Acc: {val_accuracies[-1]:.2f}')

        if epochs_no_improve == n_epochs_stop:
            print("Early stopping!")
            break
            
    # Plotting training and validation losses
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'bo-', label='Training Loss')
    plt.plot(val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting training and validation accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    
    torch.save(model.state_dict(), 'Basic_trained_model.pth')
    print('Model saved successfully as Basic_trained_model.pth.')
 

if __name__ == '__main__':
    main()
