import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
from model import create_model, CNNWithAttention, TSTransformerEncoder, FocalLoss
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
    def __init__(self, model, optimizer, log_dir, patience=3, early_stop_patience=3,
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
        dropout_modules = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                dropout_modules.append((name, module))
        
        if not dropout_modules:
            return
            
        # Dropout değişikliğinden önce tüm değerleri topla
        old_rates = [module.p for _, module in dropout_modules]
        avg_old_rate = sum(old_rates) / len(old_rates)
        
        # Tüm dropout modüllerinin oranlarını değiştir
        for _, module in dropout_modules:
            current_p = module.p
            if decrease:
                module.p = max(0.1, current_p - 0.1)
            else:
                module.p = min(0.9, current_p + 0.1)
        
        # Değişiklikten sonra tüm değerleri topla
        new_rates = [module.p for _, module in dropout_modules]
        avg_new_rate = sum(new_rates) / len(new_rates)
        
        # Sadece ortalama değişikliği rapor et
        if avg_old_rate != avg_new_rate:
            if decrease:
                print(f"Dropout rates decreased: {avg_old_rate:.2f} -> {avg_new_rate:.2f} (avg of {len(dropout_modules)} layers)")
            else:
                print(f"Dropout rates increased: {avg_old_rate:.2f} -> {avg_new_rate:.2f} (avg of {len(dropout_modules)} layers)")
    
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
            if self.counter >= self.early_stop_patience * 0.5 and self.counter < self.early_stop_patience:
                remaining = self.early_stop_patience - self.counter
                print(f"WARNING: Validation loss did not improve for {self.counter} epochs. "
                      f"Early stopping in {remaining} more epochs if no improvement.")
        
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
            return True
            
        # Durum raporu
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Status: {status.upper()}, Current LR: {current_lr:.6f}")
        
        # Log metrics
        self.log_metrics(epoch, train_loss, val_loss, accuracy, status, current_lr)
        
        return False

def plot_results(train_losses, val_losses, test_accuracies, plots_dir, method_name):
    # Loss plot
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title(f'Training and Validation Loss - {method_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'loss_plot_{method_name.lower()}.png'))
    plt.close()
    
    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, test_accuracies, 'g-')
    plt.title(f'Test Accuracy - {method_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'accuracy_plot_{method_name.lower()}.png'))
    plt.close()
    
    # Save metrics as CSV
    metrics_file = os.path.join(plots_dir, f'metrics_{method_name.lower()}.csv')
    with open(metrics_file, 'w') as f:
        f.write(f"Method: {method_name}\n")
        f.write("Epoch,Train Loss,Val Loss,Test Accuracy\n")
        for i, (tl, vl, ta) in enumerate(zip(train_losses, val_losses, test_accuracies)):
            f.write(f"{i+1},{tl:.6f},{vl:.6f},{ta:.2f}\n")

def plot_confusion_matrix(y_true, y_pred, plots_dir, method_name):
    """
    Confusion matrix oluştur ve görselleştir
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Model tahminleri
        plots_dir: Görsel çıktıların kaydedileceği dizin
        method_name: Kullanılan yöntemin adı
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
    plt.title(f'Normalized Confusion Matrix - {method_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{method_name.lower()}.png'))
    plt.close()
    
    # Hiç tahmin yapılmayan sınıfları kontrol et
    pred_classes = np.unique(y_pred)
    missing_classes = [i for i in range(len(activities)) if i not in pred_classes]
    
    if missing_classes:
        print("\nWARNING: Model did not predict these classes:", 
              ", ".join([activities[i] for i in missing_classes]))
        print("This may cause 'Precision is ill-defined' warnings.\n")
    
    # Classification report'u kaydet - zero_division parametresi ekledik
    report = classification_report(y_true, y_pred, target_names=activities, 
                                  zero_division=0)
    
    report_path = os.path.join(plots_dir, f'classification_report_{method_name.lower()}.txt')
    with open(report_path, 'w') as f:
        f.write(f"Method: {method_name}\n\n")
        # Eksik sınıflar hakkında bilgi ekle
        if missing_classes:
            f.write("NOTE: The model did not predict the following classes:\n")
            f.write(", ".join([activities[i] for i in missing_classes]) + "\n\n")
        f.write(report)

def train_model(model, train_loader, test_loader, criterion, optimizer, 
                method_name, run_dir, num_epochs=50, device='cuda', patience=3, scheduler=None):
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
    
    # Method specific training
    is_transformer = method_name == "Transformer+Contrastive"
    is_one_cycle_lr = isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)
    
    # Maximum gradient norm for clipping
    max_grad_norm = 1.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if is_transformer:
                # For transformer with contrastive loss
                outputs = model(inputs, return_projection=True)
                loss = criterion(outputs, labels)
            else:
                # For CNN with attention
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            # NaN kontrolü
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: Loss is {loss.item()}, skipping batch")
                continue
                
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            # OneCycleLR scheduler steps per batch, not per epoch
            if scheduler is not None and is_one_cycle_lr:
                scheduler.step()
            
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
                
                if is_transformer:
                    # For transformer, we need to get just the class outputs
                    outputs, _ = model(inputs, return_projection=True)
                else:
                    outputs = model(inputs)
                
                # Calculate validation loss
                val_loss = criterion(outputs if not is_transformer else (outputs, _), labels)
                
                # NaN kontrolü
                if torch.isnan(val_loss) or torch.isinf(val_loss):
                    print(f"WARNING: Validation loss is {val_loss.item()}, using previous value")
                    # Bir önceki batch'in değerini kullanmak için val_loss'u değiştirme
                    continue
                    
                running_val_loss += val_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Collect predictions and labels for confusion matrix
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Prevent division by zero or NaN
        if running_val_loss == 0 or len(test_loader) == 0:
            print("WARNING: Validation loss calculation failed. Using a default value.")
            val_epoch_loss = 1.0  # Default value
        else:
            val_epoch_loss = running_val_loss / len(test_loader)
        
        accuracy = 100 * correct / total if total > 0 else 0
        
        # Learning rate scheduler step - but only for ReduceLROnPlateau
        if scheduler is not None and not is_one_cycle_lr:
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_epoch_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            # Log if learning rate changed
            if new_lr != current_lr:
                print(f'Learning rate adjusted from {current_lr:.6f} to {new_lr:.6f}')
        
        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, '
              f'Val Loss: {val_epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%, '
              f'LR: {current_lr:.6f}')
        
        # Validation loss ratio - yüksekse underfitting, çok düşükse overfitting tehlikesi
        loss_ratio = epoch_loss / val_epoch_loss if val_epoch_loss > 0 else float('inf')
        if is_transformer and epoch > 0:
            if loss_ratio < 0.2:
                print(f"UYARI: Training/Validation loss oranı çok düşük ({loss_ratio:.4f}). "
                      f"Overfitting tehlikesi olabilir.")
            elif loss_ratio > 0.9:
                print(f"UYARI: Training/Validation loss oranı çok yüksek ({loss_ratio:.4f}). "
                      f"Modelin validation setinde zorlandığını gösterir.")
        
        # Monitor'ü çağır ve early stopping kontrolü yap
        if monitor(epoch, epoch_loss, val_epoch_loss, accuracy):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Reason: Validation loss did not improve for {patience} epochs")
            print(f"Restoring best model from epoch {epoch+1-monitor.counter}")
            print(f"Best validation loss: {monitor.best_loss:.4f}\n")
            model.load_state_dict(monitor.best_model)
            break
    
    # Son epoch'taki tahminlerle confusion matrix oluştur
    plots_dir = os.path.join(os.path.dirname(monitor.log_dir), "plots")
    plot_confusion_matrix(all_labels, all_predictions, plots_dir, method_name)
    
    return monitor.train_losses, monitor.val_losses, monitor.test_accuracies, model

def main(dev_mode=True, method="cnn_attention"):
    """
    Ana eğitim fonksiyonu
    
    Args:
        dev_mode (bool): Geliştirme modu için True, tam eğitim için False
        method (str): 'cnn_attention', 'transformer_contrastive', veya 'multi_branch_cnn'
    """
    # Yöntem adını insan okunabilir formata çevir
    if method == "cnn_attention":
        method_name = "CNN+Attention"
        loss_name = "Focal Loss"
    elif method == "transformer_contrastive":
        method_name = "Transformer+Contrastive"
        loss_name = "Contrastive+CE Loss"
    elif method == "multi_branch_cnn":
        method_name = "Multi-Branch CNN"
        loss_name = "CE Loss"
    else:
        raise ValueError(f"Unknown method: {method}. Use 'cnn_attention', 'transformer_contrastive', or 'multi_branch_cnn'")
    
    print(f"Starting training with {method_name} method ({loss_name})...")
    
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
        batch_size=64, 
        dev_mode=dev_mode,
        method=method  # Pass method to data loader for proper preprocessing
    )
    
    # Initialize model and criterion based on method
    model, criterion = create_model(method, num_classes=6)
    model = model.to(device)
    
    # Metoda özel optimizasyon parametreleri
    if method == "transformer_contrastive":
        # Transformer için daha düşük learning rate ve farklı weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.0005,  # Daha düşük LR
            weight_decay=0.03,  # Daha yüksek weight decay
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Transformer için farklı scheduler (OneCycleLR)
        total_steps = len(train_loader) * (50 if not dev_mode else 10)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.001,  # Maximum learning rate
            total_steps=total_steps,
            pct_start=0.1,  # İlk %10'luk adımda warm-up
            anneal_strategy='cos',  # Cosine annealing
            div_factor=25.0,  # initial_lr = max_lr/div_factor
            final_div_factor=1000.0  # min_lr = initial_lr/final_div_factor
        )
    elif method == "multi_branch_cnn":
        # Multi-Branch CNN için özel ayarlar
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=4,
            min_lr=1e-7
        )
    else:
        # CNN için önceki optimizer ve scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=4,
            min_lr=1e-7
        )
    
    # Set number of epochs and patience based on mode
    num_epochs = 10 if dev_mode else 50
    patience = 2 if dev_mode else 3
    
    # Format the method name with loss type for display
    display_method_name = f"{method_name} ({loss_name})"
    
    # Train model
    train_losses, val_losses, test_accuracies, model = train_model(
        model, train_loader, test_loader, criterion, optimizer,
        method_name=display_method_name,
        run_dir=run_dir,
        num_epochs=num_epochs, 
        device=device, 
        patience=patience,
        scheduler=scheduler  # Pass scheduler to train_model
    )
    
    # Plot and save results
    plot_results(train_losses, val_losses, test_accuracies, plots_dir, display_method_name)
    
    # Save model
    model_name = f'{method.lower()}_model_{"dev" if dev_mode else "full"}.pth'
    model_path = os.path.join(models_dir, model_name)
    torch.save({
        'epoch': len(train_losses),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'test_accuracy': test_accuracies[-1],
        'method': method
    }, model_path)
    
    print(f"Training completed! All outputs saved to: {run_dir}")
    print(f"Method: {display_method_name}")
    print(f"Final test accuracy: {test_accuracies[-1]:.2f}%")
    return model, test_accuracies[-1]

if __name__ == '__main__':
    # Default olarak CNN+Attention kullan
    # Diğer yöntemi kullanmak için: main(dev_mode=True, method="transformer_contrastive")
    # main(dev_mode=True, method="cnn_attention")
    main(dev_mode=False, method="transformer_contrastive")