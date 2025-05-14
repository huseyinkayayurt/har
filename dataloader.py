import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os

class HARDataset(Dataset):
    def __init__(self, data_path, split='train', dev_mode=False):
        """
        Args:
            data_path (str): Path to UCI HAR Dataset
            split (str): 'train' or 'test'
            dev_mode (bool): If True, use only a small portion of data for development
        """
        self.data_path = data_path
        self.split = split
        
        # Load data
        X_data_path = os.path.join(data_path, split, 'X_' + split + '.txt')
        y_data_path = os.path.join(data_path, split, 'y_' + split + '.txt')
        
        # Load features and labels
        self.features = np.loadtxt(X_data_path)
        self.labels = np.loadtxt(y_data_path).astype(int) - 1  # Convert 1-6 to 0-5
        
        if dev_mode:
            # Use only 10% of the data in dev mode
            dev_size = len(self.features) // 10
            self.features = self.features[:dev_size]
            self.labels = self.labels[:dev_size]
        
        # Reshape features to (N, 1, 33, 17) for CNN
        # Original shape is (N, 561), we'll reshape it to be compatible with our CNN
        N = self.features.shape[0]
        # 561 = 33 * 17
        self.features = self.features.reshape(N, 1, 33, 17)
        
        # Normalize features
        self.features = (self.features - np.mean(self.features)) / np.std(self.features)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label

def create_dataloaders(data_path, batch_size=32, num_workers=4, dev_mode=False):
    """
    Create train and test dataloaders for UCI HAR Dataset
    
    Args:
        data_path (str): Path to UCI HAR Dataset
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        dev_mode (bool): If True, use only a small portion of data for development
    
    Returns:
        train_loader, test_loader (DataLoader): PyTorch dataloaders
    """
    # Create datasets
    train_dataset = HARDataset(data_path, split='train', dev_mode=dev_mode)
    test_dataset = HARDataset(data_path, split='test', dev_mode=dev_mode)
    
    print(f"Dataset sizes (dev_mode={dev_mode}):")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Feature shape: {train_dataset[0][0].shape}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader 