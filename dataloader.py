import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os

def load_data(data_path, split):
    """
    UCI HAR Dataset'ini yükle
    
    Args:
        data_path (str): Veri setinin yolu
        split (str): 'train' veya 'test'
    
    Returns:
        X (numpy.ndarray): Özellikler
        y (numpy.ndarray): Etiketler
    """
    # Veri dosyalarının yolları
    X_data_path = os.path.join(data_path, split, f'X_{split}.txt')
    y_data_path = os.path.join(data_path, split, f'y_{split}.txt')
    
    # Özellikleri ve etiketleri yükle
    X = np.loadtxt(X_data_path)
    y = np.loadtxt(y_data_path).astype(int) - 1  # 1-6'dan 0-5'e dönüştür
    
    return X, y

def reshape_data(X, method):
    """
    Veriyi model için uygun boyutlara dönüştür
    
    Args:
        X (numpy.ndarray): Özellikler
        method (str): Kullanılan model yöntemi
    
    Returns:
        X (numpy.ndarray): Dönüştürülmüş özellikler
    """
    # Her örnek 561 özelliğe sahip
    # Bu özellikler 9 sensörden geliyor
    # Her sensör için 561/9 = 62 zaman adımı var
    
    if method == "transformer_contrastive":
        # (batch_size, 33, 17)
        return X.reshape(X.shape[0], 33, 17)
    elif method in ["multi_branch_cnn", "cnn_lstm"]:
        # (batch_size, 9, 128)
        # Eğer 561 < 9*128 ise, kalan kısmı sıfırla doldur
        X_new = np.zeros((X.shape[0], 9*128))
        X_new[:, :561] = X
        return X_new.reshape(X.shape[0], 9, 128)
    else:
        # (batch_size, 1, 33, 17)
        return X.reshape(X.shape[0], 1, 33, 17)

class HARDataset(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            features (numpy.ndarray): Özellikler
            labels (numpy.ndarray): Etiketler
        """
        self.features = features
        self.labels = labels
        
        # Normalize features
        self.features = (self.features - np.mean(self.features)) / (np.std(self.features) + 1e-8)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label

def create_dataloaders(data_path, batch_size=64, dev_mode=True, method=None):
    """
    Veri yükleyicileri oluştur
    
    Args:
        data_path (str): Veri setinin yolu
        batch_size (int): Batch boyutu
        dev_mode (bool): Geliştirme modu için True, tam eğitim için False
        method (str): Kullanılan model yöntemi ('cnn_attention', 'transformer_contrastive', 'multi_branch_cnn', 'cnn_lstm')
    """
    # Her model için veriyi yeniden yükle ve dönüştür
    X_train, y_train = load_data(data_path, 'train')
    X_test, y_test = load_data(data_path, 'test')
    
    # Geliştirme modunda veri setini küçült
    if dev_mode:
        X_train = X_train[:1000]
        y_train = y_train[:1000]
        X_test = X_test[:200]
        y_test = y_test[:200]
    
    # Veriyi model için uygun boyutlara dönüştür
    X_train = reshape_data(X_train, method)
    X_test = reshape_data(X_test, method)
    
    # Veri setlerini oluştur
    train_dataset = HARDataset(X_train, y_train)
    test_dataset = HARDataset(X_test, y_test)
    
    # DataLoader'ları oluştur
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader 