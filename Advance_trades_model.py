import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt 
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR
import torchattacks


def save_checkpoint(state, filename="checkpoint_epoch_1.pth"):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename="checkpoint_epoch_1.pth"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from '{filename}'")
    else:
        print(f"No checkpoint found at '{filename}'")


# Define TRADES Loss Class
class TradesLoss(nn.Module):
    def __init__(self, beta=1.0):
        super(TradesLoss, self).__init__()
        self.beta = beta
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, model, x_nat, x_adv, y):
        model.eval()  # Ensure model is in evaluation mode for consistent predictions
        clean_preds = model(x_nat)
        adv_preds = model(x_adv)
        
        model.train()  # Switch back to train mode if necessary elsewhere in your code
        clean_loss = self.cross_entropy(clean_preds, y)
        kl_div = nn.KLDivLoss(reduction='batchmean')(
            torch.log_softmax(adv_preds, dim=1),
            torch.softmax(clean_preds, dim=1)
        )
        total_loss = clean_loss + self.beta * kl_div
        return total_loss

base_dir = "C:/Users/robin/Projects/CropClassificationProject/Datasets/crop_dataset"
image_size = 224
batch_size = 32  # Set the batch size you want to use

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
    weights = ResNet18_Weights.DEFAULT  # Use DEFAULT for the most up-to-date weights
    model = resnet18(weights=weights)
    num_classes = 2  # We have 2 classes (Bacterial Spot and Healthy) for the Bell Pepper
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the final fully connected layer of the model to match the number of classes
    
    model.layer4.register_forward_hook(get_activation('layer4'))
   
    # Freeze all layers except the final layer
    for param in model.parameters():
        param.requires_grad = False   # Freeze all parameters
    for param in model.fc.parameters():
        param.requires_grad = True    # Unfreeze the last layer
    
   
   # Check if CUDA is available and move the model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize the optimizer and scheduler
    optimizer = optim.Adam(model.fc.parameters(), lr=0.01, weight_decay=1e-3) # Decay LR every 5 epochs
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  
    load_checkpoint(model, optimizer) 
    loss_func = TradesLoss(beta=6.0)
    
    # Define Multiple Attacks
    fgsm_attack = torchattacks.FGSM(model, eps=0.03)
    pgd_attack = torchattacks.PGD(model, eps=0.007, alpha=0.001, steps=7)
    cw_attack = torchattacks.CW(model, c=1, kappa=0, steps=1000, lr=0.01)
    deepfool_attack = torchattacks.DeepFool(model, steps=50)


    # Training and evaluation loop
    num_epochs = 1
    epochs_no_improve = 0  # Initialize epochs_no_improve
    n_epochs_stop = 5  # Number of epochs to stop after no improvement
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        model.train()  # Set model to training mode
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
           # Apply FGSM, PGD, CW, and DeepFool attacks
            images_adv_fgsm = fgsm_attack(images, labels)
            images_adv_pgd = pgd_attack(images, labels)
            images_adv_cw = cw_attack(images, labels)
            images_adv_deepfool = deepfool_attack(images, labels)

            optimizer.zero_grad()
            outputs = model(images_adv_deepfool)
            features = activations['layer4']
            loss = loss_func(model, images, images_adv_deepfool, labels)
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
        save_checkpoint({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, filename=f"checkpoint_epoch_{epoch+1}.pth")
        
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
    plt.title('Training and Validation Loss- FGSM Attack')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting training and validation accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy - FGSM Attack ')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), 'Advance_trained_model.pth')
    print('Model saved successfully as Advance_trained_model.pth.')
 

if __name__ == '__main__':
    main()
