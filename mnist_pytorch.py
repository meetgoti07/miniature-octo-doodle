"""
MNIST Digit Recognition - Method 3: PyTorch Implementation
Using PyTorch with CNN architecture for digit classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class MNISTNet(nn.Module):
    """CNN architecture for MNIST classification"""
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layers
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        # Second conv block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        
        # Third conv block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout1(x)
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

def load_data():
    """Load and preprocess MNIST data"""
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, device, epochs=20):
    """Train the PyTorch model"""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.6f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Avg Loss: {epoch_loss:.6f}, '
              f'Accuracy: {epoch_accuracy:.2f}%')
        
        scheduler.step()
    
    return train_losses, train_accuracies

def test_model(model, test_loader, device):
    """Test the PyTorch model"""
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest Results:')
    print(f'Average Loss: {test_loss:.4f}')
    print(f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    
    return accuracy, all_preds, all_targets

def visualize_results(train_losses, train_accuracies, all_preds, all_targets):
    """Visualize training results and confusion matrix"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot training accuracy
    ax2.plot(train_accuracies)
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title('Confusion Matrix - PyTorch Model')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    
    # Classification accuracy per digit
    accuracy_per_digit = []
    for i in range(10):
        mask = np.array(all_targets) == i
        if mask.sum() > 0:
            acc = (np.array(all_preds)[mask] == i).sum() / mask.sum() * 100
            accuracy_per_digit.append(acc)
        else:
            accuracy_per_digit.append(0)
    
    ax4.bar(range(10), accuracy_per_digit)
    ax4.set_title('Accuracy per Digit')
    ax4.set_xlabel('Digit')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_xticks(range(10))
    
    plt.tight_layout()
    plt.savefig('pytorch_results.png')
    plt.show()

def visualize_sample_predictions(model, test_loader, device):
    """Visualize some sample predictions"""
    model.eval()
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    data, target = next(data_iter)
    data, target = data.to(device), target.to(device)
    
    # Make predictions
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1)
    
    # Plot some samples
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(10):
        row = i // 5
        col = i % 5
        
        # Convert tensor to numpy for plotting
        img = data[i].cpu().squeeze().numpy()
        
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'True: {target[i].item()}, Pred: {pred[i].item()}')
        axes[row, col].axis('off')
        
        # Color the title based on correctness
        color = 'green' if target[i].item() == pred[i].item() else 'red'
        axes[row, col].title.set_color(color)
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)')
    plt.tight_layout()
    plt.savefig('pytorch_sample_predictions.png')
    plt.show()

def main():
    """Main function"""
    print("MNIST Digit Recognition - PyTorch Implementation")
    print("=" * 55)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading and preprocessing data...")
    train_loader, test_loader = load_data()
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating PyTorch model...")
    model = MNISTNet().to(device)
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Train model
    print("\nTraining model...")
    train_losses, train_accuracies = train_model(model, train_loader, device, epochs=20)
    
    # Test model
    print("\nTesting model...")
    accuracy, all_preds, all_targets = test_model(model, test_loader, device)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds))
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(train_losses, train_accuracies, all_preds, all_targets)
    visualize_sample_predictions(model, test_loader, device)
    
    # Save model
    torch.save(model.state_dict(), 'pytorch_mnist_model.pth')
    print("\nModel saved as 'pytorch_mnist_model.pth'")
    
    return accuracy

if __name__ == "__main__":
    main()