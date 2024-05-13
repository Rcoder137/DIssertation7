import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, roc_curve, auc
import seaborn as sns
import torchattacks


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
    test_path = os.path.join(extract_folder, 'Peach', 'Test')
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
        


# Initialize a dictionary to store the results
results = {
    "Clean": {"accuracy": [], "precision": [], "recall": [], "f1": [], "loss": []},
    "Adversarial": {"accuracy": [], "precision": [], "recall": [], "f1": [], "loss": []}
}

classes = ['Bell Pepper Healthy', 'Bell Pepper Bacterial Spot']
def plot_confusion_matrix_seaborn(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def evaluate_clean_data(model, test_loader, device, results):
    model.eval()  # Set the model to evaluation mode
    correct = 0   # Counter for correct predictions
    total = 0     # Total number of predictions
    total_loss = 0.0  # Total loss for all batches
    y_true = []
    y_pred = []
    y_scores = []  # Store probability scores for ROC calculation
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item() * images.size(0)
            probabilities = F.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)
           
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
            y_scores.extend(probabilities.tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        if total > 0:
           fpr, tpr, _ = roc_curve(y_true, y_scores)
           roc_auc = auc(fpr, tpr)
           avg_loss = total_loss / total
           accuracy = 100 * correct / total
           cm = confusion_matrix(y_true, y_pred)
           precision, recall, f1 = precision_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred, average='macro')
           results["Clean"].update({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "loss": avg_loss, "confusion_matrix": cm})
           print("Clean Data Results:")
           print(f'Accuracy on test set: {accuracy:.2f}%')
           print(f'Average loss on test set: {avg_loss:.4f}')
           print(f"Accuracy: {results['Clean']['accuracy']:.2f}%")
           print(f"Precision: {results['Clean']['precision']:.2f}")
           print(f"Recall: {results['Clean']['recall']:.2f}")
           print(f"F1 Score: {results['Clean']['f1']:.2f}")
           print(f"Average Loss: {results['Clean']['loss']:.4f}")
           # Plot the confusion matrix
           plot_confusion_matrix_seaborn(cm, classes, title='Confusion Matrix - Clean Data')

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
           
    return y_true, y_pred, y_scores
    
   
    
def test_adversarial(model, device, test_loader, attack, results, attack_name):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    y_true = []
    y_pred = []
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attack(images, labels)
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
            
        correct += (predicted == labels).sum().item()
        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())
        total += labels.size(0)
            
    if total > 0:
        avg_loss = total_loss / total
        accuracy = correct / total
        cm = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        
        results["Adversarial"][attack_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "loss": avg_loss,
            "confusion_matrix": cm
        }
        print(f"{attack_name} Attack Results:")
        print(f'{attack_name} Attack - Processed {total} images - Accuracy: {accuracy*100:.2f}%, Loss: {avg_loss:.4f}')
        print(f"{attack_name} Attack Results:")
        print(f"Accuracy: {results['Adversarial'][attack_name]['accuracy']:.2f}%")
        print(f"Precision: {results['Adversarial'][attack_name]['precision']:.2f}")
        print(f"Recall: {results['Adversarial'][attack_name]['recall']:.2f}")
        print(f"F1 Score: {results['Adversarial'][attack_name]['f1']:.2f}")
        print(f"Average Loss: {results['Adversarial'][attack_name]['loss']:.4f}")
        plot_confusion_matrix_seaborn(cm, classes, title=f'Confusion Matrix - {attack_name}')
    else:
        print(f"No data processed in {attack_name} attack. No metrics to calculate.")


   
    
if __name__ == '__main__':
    test_loader = test_datasets(extract_folder, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = r'C:\Users\robin\dessertation\cropclassification\Training\Advance_trained_model.pth'
    model = load_model(model_path, device=device)
    class_labels = ['Bell Pepper Healthy', 'Bell Pepper Bacterial Spot']

    attacks = {
        'FGSM': torchattacks.FGSM(model, eps=0.03),
        'PGD': torchattacks.PGD(model, eps=0.007, alpha=0.001, steps=7),
        'CW': torchattacks.CW(model, c=1, kappa=0, steps=1000, lr=0.01),
        'DeepFool': torchattacks.DeepFool(model, steps=50)
    }

    # Collect metrics for each attack and clean data
    results = {
        "Clean": {"accuracy": [], "precision": [], "recall": [], "f1": [], "loss": []},
        "Adversarial": {}
    }
    
    # Evaluate clean data
    y_true, y_pred, y_scores = evaluate_clean_data(model, test_loader, device, results)
    
    # Evaluate adversarial attacks
    for name, attack in attacks.items():
        test_adversarial(model, device, test_loader, attack, results, name)
     