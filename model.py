import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(x_cat))
        return x * attention_map

class CNNWithAttention(nn.Module):
    def __init__(self, num_classes=6):
        super(CNNWithAttention, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.attention = SpatialAttention()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = None
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.is_first_forward = True

    def _initialize_fc1(self, x):
        with torch.no_grad():
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.attention(x)
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.attention(x)
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = self.attention(x)
            flattened_size = x.view(x.size(0), -1).shape[1]
            self.fc1 = nn.Linear(flattened_size, 512).to(x.device)

    def forward(self, x):
        if self.is_first_forward:
            self._initialize_fc1(x)
            self.is_first_forward = False
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.attention(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.attention(x)
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.attention(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean() 