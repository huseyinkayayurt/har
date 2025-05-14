import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from model import CNNWithAttention, FocalLoss
from dataloader import create_dataloaders

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    
    return train_losses, test_accuracies

def plot_results(train_losses, test_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    # Plot test accuracy
    ax2.plot(test_accuracies)
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

def main(dev_mode=True):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    data_path = 'UCI HAR Dataset'
    train_loader, test_loader = create_dataloaders(
        data_path, 
        batch_size=32, 
        dev_mode=dev_mode
    )
    
    # Initialize model, loss, and optimizer
    model = CNNWithAttention(num_classes=6).to(device)
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Set number of epochs based on mode
    num_epochs = 10 if dev_mode else 50
    
    # Train model
    train_losses, test_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, 
        num_epochs=num_epochs, device=device
    )
    
    # Plot and save results
    plot_results(train_losses, test_accuracies)
    
    # Save model
    model_name = 'har_cnn_attention_model_dev.pth' if dev_mode else 'har_cnn_attention_model.pth'
    torch.save(model.state_dict(), model_name)
    print(f"Training completed and model saved as {model_name}!")

if __name__ == '__main__':
    main(dev_mode=True)  # Set to True for development, False for full training 