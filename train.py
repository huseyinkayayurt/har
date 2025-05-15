import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
from model import CNNWithAttention, FocalLoss
from dataloader import create_dataloaders

def setup_output_dir(dev_mode=True):
    """Çıktı klasörlerini oluştur"""
    # Ana output klasörü
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Timestamp ile benzersiz run klasörü
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "dev" if dev_mode else "full"
    run_dir = os.path.join(output_dir, f"run_{timestamp}_{mode}")
    os.makedirs(run_dir)
    
    # Alt klasörler
    models_dir = os.path.join(run_dir, "models")
    plots_dir = os.path.join(run_dir, "plots")
    logs_dir = os.path.join(run_dir, "logs")
    
    os.makedirs(models_dir)
    os.makedirs(plots_dir)
    os.makedirs(logs_dir)
    
    return run_dir, models_dir, plots_dir, logs_dir

class TrainingMonitor:
    def __init__(self, model, optimizer, log_dir, patience=7, early_stop_patience=7,
                 min_delta=0.001, lr_patience=3, lr_factor=0.5, min_lr=1e-6,
                 dropout_inc_threshold=2.0, dropout_dec_threshold=0.5):
        """
        Args:
            model: Model instance
            optimizer: Optimizer instance
            log_dir: Log dosyalarının kaydedileceği dizin
            patience: Genel izleme için sabır
            early_stop_patience: Early stopping için sabır
            min_delta: Minimum değişim miktarı
            lr_patience: Learning rate düşürme için sabır
            lr_factor: Learning rate düşürme faktörü
            min_lr: Minimum learning rate
            dropout_inc_threshold: Dropout artırma eşiği (train/val loss oranı)
            dropout_dec_threshold: Dropout azaltma eşiği (train/val loss oranı)
        """
        self.model = model
        self.optimizer = optimizer
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "training_log.txt")
        
        # Early stopping için
        self.patience = patience
        self.min_delta = min_delta
        self.early_stop_patience = early_stop_patience
        
        # Learning rate ayarlama için
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.min_lr = min_lr
        self.lr_counter = 0
        self.best_lr_loss = float('inf')
        
        # Dropout ayarlama için
        self.dropout_inc_threshold = dropout_inc_threshold
        self.dropout_dec_threshold = dropout_dec_threshold
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.test_accuracies = []
        
        # Early stopping için
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
        # Log dosyasını başlat
        with open(self.log_file, 'w') as f:
            f.write("Epoch,Train Loss,Val Loss,Accuracy,Status,Learning Rate\n")
    
    def log_metrics(self, epoch, train_loss, val_loss, accuracy, status, lr):
        """Metrikleri log dosyasına kaydet"""
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{accuracy:.2f},{status},{lr:.6f}\n")
    
    def check_overfitting(self, train_loss, val_loss):
        """Overfitting kontrolü ve dropout ayarlama"""
        ratio = train_loss / val_loss if val_loss > 0 else float('inf')
        
        if ratio < self.dropout_dec_threshold:  # Underfitting
            self._adjust_dropout(decrease=True)
            return "underfitting"
        elif ratio > self.dropout_inc_threshold:  # Overfitting
            self._adjust_dropout(decrease=False)
            return "overfitting"
        return "good"
    
    def _adjust_dropout(self, decrease=True):
        """Model'in dropout oranlarını ayarla"""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                current_p = module.p
                if decrease:
                    module.p = max(0.1, current_p - 0.1)
                else:
                    module.p = min(0.9, current_p + 0.1)
                if current_p != module.p:
                    print(f"Dropout rate adjusted from {current_p:.2f} to {module.p:.2f}")
    
    def _adjust_learning_rate(self):
        """Learning rate'i düşür"""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.lr_factor, self.min_lr)
            param_group['lr'] = new_lr
            print(f"Learning rate adjusted from {old_lr:.6f} to {new_lr:.6f}")
            
        self.lr_counter = 0
        
    def __call__(self, epoch, train_loss, val_loss, accuracy):
        """Her epoch sonunda çağrılacak ana fonksiyon"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.test_accuracies.append(accuracy)
        
        # Early stopping kontrolü
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model = self.model.state_dict().copy()
            self.counter = 0
        else:
            self.counter += 1
        
        # Learning rate ayarlama kontrolü
        if val_loss < self.best_lr_loss - self.min_delta:
            self.best_lr_loss = val_loss
            self.lr_counter = 0
        else:
            self.lr_counter += 1
            
        if self.lr_counter >= self.lr_patience:
            self._adjust_learning_rate()
        
        # Overfitting/Underfitting kontrolü
        status = self.check_overfitting(train_loss, val_loss)
        
        # Early stopping kontrolü
        if self.counter >= self.early_stop_patience:
            self.early_stop = True
            print("Early stopping triggered")
            return True
            
        # Durum raporu
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Status: {status.upper()}, Current LR: {current_lr:.6f}")
        
        # Log metrics
        self.log_metrics(epoch, train_loss, val_loss, accuracy, status, current_lr)
        
        return False

def plot_results(train_losses, val_losses, test_accuracies, plots_dir):
    # Loss plot
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'loss_plot.png'))
    plt.close()
    
    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, test_accuracies, 'g-')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'accuracy_plot.png'))
    plt.close()
    
    # Save metrics as CSV
    metrics_file = os.path.join(plots_dir, 'metrics.csv')
    with open(metrics_file, 'w') as f:
        f.write("Epoch,Train Loss,Val Loss,Test Accuracy\n")
        for i, (tl, vl, ta) in enumerate(zip(train_losses, val_losses, test_accuracies)):
            f.write(f"{i+1},{tl:.6f},{vl:.6f},{ta:.2f}\n")

def plot_confusion_matrix(y_true, y_pred, plots_dir):
    """
    Confusion matrix oluştur ve görselleştir
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Model tahminleri
        plots_dir: Görsel çıktıların kaydedileceği dizin
    """
    # Aktivite etiketleri
    activities = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 
                 'SITTING', 'STANDING', 'LAYING']
    
    # Confusion matrix hesapla
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize et
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Görselleştirme
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=activities, yticklabels=activities)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Classification report'u kaydet
    report = classification_report(y_true, y_pred, target_names=activities)
    report_path = os.path.join(plots_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)

def train_model(model, train_loader, test_loader, criterion, optimizer, 
                run_dir, num_epochs=50, device='cuda', patience=7, scheduler=None):
    # Training monitor başlat
    monitor = TrainingMonitor(
        model=model,
        optimizer=optimizer,
        log_dir=os.path.join(run_dir, "logs"),
        patience=patience,
        early_stop_patience=patience
    )
    
    # Initial learning rate
    last_lr = optimizer.param_groups[0]['lr']
    
    for epoch in range(num_epochs):
        # Training phase
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
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # Calculate validation loss
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Collect predictions and labels for confusion matrix
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_epoch_loss = running_val_loss / len(test_loader)
        accuracy = 100 * correct / total
        
        # Learning rate scheduler step
        if scheduler:
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_epoch_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            # Log if learning rate changed
            if new_lr != current_lr:
                print(f'Learning rate adjusted from {current_lr:.6f} to {new_lr:.6f}')
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, '
              f'Val Loss: {val_epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
        
        # Monitor'ü çağır ve early stopping kontrolü yap
        if monitor(epoch, epoch_loss, val_epoch_loss, accuracy):
            print(f"Training stopped at epoch {epoch+1}")
            model.load_state_dict(monitor.best_model)
            break
    
    # Son epoch'taki tahminlerle confusion matrix oluştur
    plots_dir = os.path.join(os.path.dirname(monitor.log_dir), "plots")
    plot_confusion_matrix(all_labels, all_predictions, plots_dir)
    
    return monitor.train_losses, monitor.val_losses, monitor.test_accuracies, model

def main(dev_mode=True):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Output klasörlerini oluştur
    run_dir, models_dir, plots_dir, logs_dir = setup_output_dir(dev_mode)
    print(f"Outputs will be saved to: {run_dir}")
    
    # Create data loaders
    data_path = 'UCI HAR Dataset'
    train_loader, test_loader = create_dataloaders(
        data_path, 
        batch_size=32, 
        dev_mode=dev_mode
    )
    
    # Initialize model with slightly higher dropout for regularization
    model = CNNWithAttention(num_classes=6, dropout_rate=0.6).to(device)
    criterion = FocalLoss(gamma=2.0)  # Focal Loss with gamma=2.0
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,  # L2 regularization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    
    # Set number of epochs and patience based on mode
    num_epochs = 10 if dev_mode else 50
    patience = 3 if dev_mode else 7
    
    # Train model
    train_losses, val_losses, test_accuracies, model = train_model(
        model, train_loader, test_loader, criterion, optimizer,
        run_dir=run_dir,
        num_epochs=num_epochs, 
        device=device, 
        patience=patience,
        scheduler=scheduler  # Pass scheduler to train_model
    )
    
    # Plot and save results
    plot_results(train_losses, val_losses, test_accuracies, plots_dir)
    
    # Save model
    model_name = 'model_dev.pth' if dev_mode else 'model_full.pth'
    model_path = os.path.join(models_dir, model_name)
    torch.save({
        'epoch': len(train_losses),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'test_accuracy': test_accuracies[-1]
    }, model_path)
    
    print(f"Training completed! All outputs saved to: {run_dir}")
    print(f"Final test accuracy: {test_accuracies[-1]:.2f}%")

if __name__ == '__main__':
    main(dev_mode=True) 