import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc



def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

extract_folder = r'C:\Users\robin\Projects\CropClassificationProject\Datasets\crop_dataset'
batch_size = 32  # Set the batch size you want to use
image_size = 224

# Define transformations for the training, validation, and testing sets
def test_transforms():
    return transforms.Compose([
           transforms.Resize((256, 256)),  # Resize to a square of 256x256 pixels
           transforms.RandomRotation(10),  # Random rotation of images
           transforms.RandomHorizontalFlip(),  # Random horizontal flip
           transforms.ToTensor(),  # Convert images to tensors
           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats  # Standard normalization for ImageNet
    ])
    
def test_datasets(extract_folder, batch_size):
    test_transform = test_transforms()
    test_path = os.path.join(extract_folder, 'Bell Pepper', 'Test')
    test_dataset = ImageFolder(root=test_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return test_loader


def load_model(model_path, device='cuda'):
    # Ensure the device is available and fallback to CPU if necessary
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available! Falling back to CPU.")
        device = 'cpu'
    
    try:
        weights = ResNet18_Weights.DEFAULT  # Use DEFAULT for the most up-to-date weights
        model = resnet18(weights=weights)
        num_features = model.fc.in_features
        # Assuming you know the number of classes, this should be set dynamically based on your dataset
        num_classes = 2 
        model.fc = nn.Linear(num_features, num_classes)
        # Load the model state dictionary
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print("Model loaded successfully.")  # Success message
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"The model file was not found at {model_path}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the model: {str(e)}")

def evaluate_clean_data(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0   # Counter for correct predictions
    total = 0     # Total number of predictions
    total_loss = 0.0  # Total loss for all batches
    y_true = []
    y_pred = []
    y_scores = []  # Store probability scores for ROC calculation
    with torch.no_grad():                                      # No gradients needed as we are only testing
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)           # Compute the batch loss
            total_loss += loss.item() * images.size(0)           # Multiply batch loss by batch size to scale up
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[:, 1]  # binary classification
            _, predicted = torch.max(outputs, 1)
           
            y_true.extend(labels.tolist())  # Collect true labels
            y_pred.extend(predicted.tolist())  # Collect predicted labels
            y_scores.extend(probabilities.tolist())  # Append the probabilities for each batch
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f'Average loss on test set: {avg_loss:.4f}')
    return y_true, y_pred, y_scores
 
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show() 
    


def plot_roc_curve(y_true, y_scores, class_labels):
    # Calculate the ROC curve and AUC for the positive class
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

epsilons = [0, 0.05, 0.1, 0.15, 0.2]
   
def test_adversarial(model, device, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    adv_examples = []
    model.eval()

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True  # Enable gradient tracking on images
        outputs = model(images)
        init_pred = outputs.max(1, keepdim=True)[1]  # Get the indices of the max log-probability

        # Mask to filter out correctly predicted samples
        mask = (init_pred.squeeze() == labels)
        if torch.sum(mask) == 0:
            continue
        
        # Calculate loss only on correctly predicted samples
        loss = F.cross_entropy(outputs[mask], labels[mask])
        model.zero_grad()  # Zero gradients
        loss.backward()  # Backpropagate to calculate gradients

        # Apply mask to gradients
        data_grad = images.grad.data
        perturbed_data = fgsm_attack(images, epsilon, data_grad)

        # Re-classify the perturbed image
        outputs = model(perturbed_data)
        final_pred = outputs.max(1, keepdim=True)[1]

        # Calculate how many were still correctly predicted after the attack
        correct_mask = (final_pred.squeeze() == labels) & mask
        correct += correct_mask.sum().item()

        # Optionally save some adversarial examples for visualization
        if len(adv_examples) < 5 and correct_mask.any():
            index = correct_mask.nonzero(as_tuple=True)[0][0]
            adv_ex = perturbed_data[index].squeeze().detach().cpu().numpy()
            adv_examples.append((init_pred[index].item(), final_pred[index].item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader.dataset))
    print(f'Epsilon: {epsilon}\tTest Accuracy = {final_acc * 100:.2f}%')

    return final_acc, adv_examples



def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1



if __name__ == '__main__':
    test_loader = test_datasets(extract_folder, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = r'C:\Users\robin\dessertation\cropclassification\Training\Basic_model\Basic_trained_model.pth'
    model = load_model(model_path, device=device)
    class_labels = ['Bell Pepper Healthy', 'Bell Pepper Bacterial Spot']
    # Evaluate the model on clean data
    y_true, y_pred, y_scores = evaluate_clean_data(model, test_loader, device)
    accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)

         
    for eps in epsilons:
        test_adversarial(model, device, test_loader, eps)
    print(f"Calculated Metrics -> Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
    plot_confusion_matrix(y_true, y_pred, classes=['Healthy', 'Bacterial Spot'])
    plot_roc_curve(y_true, y_scores, class_labels)
