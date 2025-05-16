import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os

class HARDataset(Dataset):
    def __init__(self, data_path, split='train', dev_mode=False, method='cnn_attention'):
        """
        Args:
            data_path (str): Path to UCI HAR Dataset
            split (str): 'train' or 'test'
            dev_mode (bool): If True, use only a small portion of data for development
            method (str): 'cnn_attention', 'transformer_contrastive', or 'multi_branch_cnn'
        """
        self.data_path = data_path
        self.split = split
        self.method = method
        
        # Load data
        X_data_path = os.path.join(data_path, split, 'X_' + split + '.txt')
        y_data_path = os.path.join(data_path, split, 'y_' + split + '.txt')
        
        # Load features and labels
        self.features = np.loadtxt(X_data_path)
        self.labels = np.loadtxt(y_data_path).astype(int) - 1  # Convert 1-6 to 0-5
        
        if dev_mode:
            # Use only 5% of the data in dev mode
            dev_size = len(self.features) // 20
            self.features = self.features[:dev_size]
            self.labels = self.labels[:dev_size]
        
        # Process features based on method
        if method in ['cnn_attention', 'transformer_contrastive']:
            # Reshape features to (N, 1, 33, 17) for CNN
            # Original shape is (N, 561), we'll reshape it to be compatible with our CNN
            N = self.features.shape[0]
            # 561 = 33 * 17
            self.features = self.features.reshape(N, 1, 33, 17)
        
        elif method == 'multi_branch_cnn':
            # For multi-branch CNN, reshape to (N, 9, 128)
            # This is a simplification - in a real scenario, you would extract acc and gyro data
            # Reshaped features: First 6 channels (0-5) for acc+gyro, last 3 (6-8) for total_acc
            N = self.features.shape[0]
            # Simulating 9-channel data with 128 time steps from 561 features
            transformed_features = np.zeros((N, 9, 128))
            
            # Fill the simulated data (just for demonstration)
            # In real implementation, you would properly extract these values from raw signals
            for i in range(N):
                # Reshape the 561 features into a format suitable for multi-branch CNN
                feature = self.features[i]
                
                # Reshape to 9 channels
                # First 6 channels: acc_xyz (3) + gyro_xyz (3)
                # Last 3 channels: total_acc_xyz (3)
                for c in range(9):
                    # Extract approximately 128 values for each channel
                    # This is just a mock implementation - adjust for your actual data
                    start_idx = c * (561 // 9)
                    end_idx = min(start_idx + 128, 561)
                    
                    # Pad with zeros if needed
                    actual_length = end_idx - start_idx
                    transformed_features[i, c, :actual_length] = feature[start_idx:end_idx]
            
            self.features = transformed_features
        
        # Normalize features
        self.features = (self.features - np.mean(self.features)) / (np.std(self.features) + 1e-8)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label

def create_dataloaders(data_path, batch_size=32, num_workers=4, dev_mode=False, method='cnn_attention'):
    """
    Create train and test dataloaders for UCI HAR Dataset
    
    Args:
        data_path (str): Path to UCI HAR Dataset
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        dev_mode (bool): If True, use only a small portion of data for development
        method (str): 'cnn_attention', 'transformer_contrastive', or 'multi_branch_cnn'
    
    Returns:
        train_loader, test_loader (DataLoader): PyTorch dataloaders
    """
    # Create datasets
    train_dataset = HARDataset(data_path, split='train', dev_mode=dev_mode, method=method)
    test_dataset = HARDataset(data_path, split='test', dev_mode=dev_mode, method=method)
    
    print(f"Dataset sizes (dev_mode={dev_mode}, method={method}):")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Feature shape: {train_dataset[0][0].shape}")
    
    # MPS (Apple Silicon) device check
    use_pin_memory = torch.backends.cuda.is_built() and torch.cuda.is_available()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    return train_loader, test_loader 