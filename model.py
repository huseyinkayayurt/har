import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

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

# MultiBranchCNN sınıfını ekle
class MultiBranchCNN(nn.Module):
    """
    Multi-branch CNN for processing accelerometer and gyroscope data separately.
    
    The model has two branches:
    1. Branch 1: Processes acc + gyro data (channels 0-5)
    2. Branch 2: Processes total_acc data (channels 6-8)
    
    Input shape: (batch_size, 9, 128) where:
    - First 6 channels (0-5): acc + gyro (2×3 axes)
    - Last 3 channels (6-8): total_acc
    """
    def __init__(self, num_classes=6, dropout_rate=0.5):
        super(MultiBranchCNN, self).__init__()
        
        # Branch 1: ACC + GYRO (channels 0-5)
        self.branch1 = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate * 0.7),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)
        )
        
        # Branch 2: TOTAL ACC (channels 6-8)
        self.branch2 = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate * 0.7),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)
        )
        
        # Calculate output size after convolutional layers
        # Original size: 128
        # After 3 MaxPool1d(2) layers: 128 // (2^3) = 16
        self.branch1_out_features = 256 * 16
        self.branch2_out_features = 128 * 16
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.branch1_out_features + self.branch2_out_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch_size, 9, 128)
        
        # Split input into branches
        x_branch1 = x[:, :6, :]  # ACC + GYRO (channels 0-5)
        x_branch2 = x[:, 6:, :]  # TOTAL ACC (channels 6-8)
        
        # Process each branch
        x_branch1 = self.branch1(x_branch1)
        x_branch2 = self.branch2(x_branch2)
        
        # Flatten
        x_branch1 = x_branch1.view(x_branch1.size(0), -1)
        x_branch2 = x_branch2.view(x_branch2.size(0), -1)
        
        # Concatenate branch outputs
        x_concat = torch.cat((x_branch1, x_branch2), dim=1)
        
        # Fully connected layers
        x = self.fc1(x_concat)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class CNNWithAttention(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.5):
        super(CNNWithAttention, self).__init__()
        
        # CNN layers with L2 regularization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32, momentum=0.1, eps=1e-5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.1, eps=1e-5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.1, eps=1e-5)
        
        self.attention = SpatialAttention()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layers with different rates
        self.dropout1 = nn.Dropout(dropout_rate * 0.5)  # Lighter dropout early
        self.dropout2 = nn.Dropout(dropout_rate * 0.7)  # Medium dropout
        self.dropout3 = nn.Dropout(dropout_rate)        # Full dropout late
        
        # Fully connected layers
        self.fc1 = None
        self.bn_fc1 = None  # Batch norm for FC1
        self.fc2 = nn.Linear(512, num_classes)
        self.is_first_forward = True

    def _initialize_fc1(self, x):
        with torch.no_grad():
            x = self.forward_features(x)
            flattened_size = x.view(x.size(0), -1).shape[1]
            self.fc1 = nn.Linear(flattened_size, 512).to(x.device)
            self.bn_fc1 = nn.BatchNorm1d(512, momentum=0.1, eps=1e-5).to(x.device)

    def forward_features(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.attention(x)
        x = self.dropout1(x)
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.attention(x)
        x = self.dropout2(x)
        
        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.attention(x)
        x = self.dropout3(x)
        
        return x

    def forward(self, x):
        if self.is_first_forward:
            self._initialize_fc1(x)
            self.is_first_forward = False
        
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

class TSTransformerEncoder(nn.Module):
    def __init__(self, feature_dim=561, d_model=128, nhead=8, 
                 num_layers=3, dim_feedforward=256, dropout=0.1, 
                 num_classes=6, projection_dim=64):
        super(TSTransformerEncoder, self).__init__()
        
        # Reshape and input projection
        self.input_proj = nn.Linear(feature_dim // 33, d_model)  # Assume input is reshaped
        self.input_norm = nn.LayerNorm(d_model)  # Input normalization
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Pre-classifier normalization
        self.pre_classifier_norm = nn.LayerNorm(d_model)
        
        # Classification head with smaller hidden layer to prevent overfitting
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.BatchNorm1d(dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, projection_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Weight initialization
        self._init_weights()
        
    def _init_weights(self):
        # Initialize weights with small values to prevent instability
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward_features(self, x):
        # x is expected to be (batch_size, 1, 33, 17)
        batch_size = x.shape[0]
        
        # Reshape from 2D to sequence (batch_size, seq_len=33, features=17)
        x = x.squeeze(1)
        
        # Project each time step
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)
        x = self.input_norm(x)  # Apply normalization before positional encoding
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, seq_len+1, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder layers
        for layer in self.layers:
            x = layer(x.transpose(0, 1)).transpose(0, 1)
        
        # Get class token representation
        cls_representation = x[:, 0]  # (batch_size, d_model)
        
        # Apply pre-classifier normalization
        cls_representation = self.pre_classifier_norm(cls_representation)
        
        return cls_representation
    
    def forward(self, x, return_projection=False):
        features = self.forward_features(x)
        
        if return_projection:
            # For contrastive learning
            projection = self.projection(features)
            projection = F.normalize(projection, dim=1)
            return self.classifier(features), projection
        
        # Just classification
        return self.classifier(features)

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

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, projections, targets):
        # projections: [batch_size, projection_dim]
        # targets: [batch_size]
        
        device = projections.device
        batch_size = projections.shape[0]
        
        # Normalize projections
        projections = F.normalize(projections, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(projections, projections.t()) / self.temperature
        
        # Create mask for positive pairs (same class)
        mask = targets.unsqueeze(1) == targets.unsqueeze(0)
        mask.fill_diagonal_(False)  # Remove self-contrast cases
        
        # For each anchor, find positive samples
        pos_mask = mask.float()
        
        # Get log_prob for all samples
        exp_logits = torch.exp(sim_matrix)
        
        # Exclude self-similarity
        log_prob = sim_matrix - torch.log(
            exp_logits.sum(dim=1, keepdim=True) - exp_logits.diag().unsqueeze(1)
        )
        
        # Compute mean of log-likelihood for positive samples
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1).clamp(min=1e-8)
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()
        
        return loss

# CNN+LSTM sınıfını ekle
class CNNLSTM(nn.Module):
    """
    CNN+LSTM model for time series classification.
    
    The model first applies 1D CNN to extract features, then passes them through
    LSTM to capture temporal dependencies, followed by a fully connected layer for classification.
    
    Input shape: (batch_size, 9, 128) where:
    - 9 channels: sensor data
    - 128: time steps
    """
    def __init__(self, num_classes=6, input_channels=9, seq_length=128, dropout_rate=0.5):
        super(CNNLSTM, self).__init__()
        
        # CNN parameters
        self.input_channels = input_channels
        self.seq_length = seq_length
        
        # CNN feature extraction layers
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Output: (B, 64, 64)
        self.dropout1 = nn.Dropout(dropout_rate * 0.5)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # Output: (B, 128, 32)
        self.dropout2 = nn.Dropout(dropout_rate * 0.7)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # Output: (B, 256, 16)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Calculate CNN output size
        # Original: 128 -> After 3 pooling layers: 128/(2^3) = 16
        cnn_output_size = seq_length // 8
        
        # LSTM parameters
        self.lstm_hidden_size = 128
        self.lstm_num_layers = 2
        
        # LSTM layer for temporal processing
        self.lstm = nn.LSTM(
            input_size=256,  # CNN output channels
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if self.lstm_num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers for classification
        lstm_output_size = self.lstm_hidden_size * 2  # Bidirectional
        self.fc1 = nn.Linear(lstm_output_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch_size, 9, 128)
        
        # CNN Feature Extraction
        # Process with CNN layers: (B, C, T) format
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Reshape for LSTM: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)
        
        # LSTM Sequence Processing
        output, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state from final layer of both directions for classification
        # Shape: (num_layers * 2, batch_size, hidden_size) -> (batch_size, hidden_size * 2)
        h_n = h_n.view(self.lstm_num_layers, 2, -1, self.lstm_hidden_size)
        h_n = h_n[-1]  # Take the last layer
        h_n = h_n.transpose(0, 1).contiguous().view(-1, self.lstm_hidden_size * 2)
        
        # Fully connected layers for classification
        x = self.fc1(h_n)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x

def create_model(method, num_classes=6):
    """
    Create model based on method
    
    Args:
        method (str): 'cnn_attention', 'transformer_contrastive', or 'multi_branch_cnn'
        num_classes (int): Number of classes
        
    Returns:
        model: PyTorch model
        criterion: Loss function
    """
    if method == "cnn_attention":
        model = CNNWithAttention(num_classes=num_classes, dropout_rate=0.6)
        criterion = FocalLoss(gamma=2.0)
        return model, criterion
    
    elif method == "transformer_contrastive":
        model = TSTransformerEncoder(
            feature_dim=561,  # Total feature dimension
            d_model=128,      # Model dimension
            nhead=8,          # Number of attention heads
            num_layers=3,     # Number of transformer layers
            dim_feedforward=256,  # Feedforward dimension
            dropout=0.1,      # Dropout rate
            num_classes=num_classes,  # Number of classes
            projection_dim=64  # Projection dimension for contrastive learning
        )
        # We'll use both losses and combine them
        supervised_contrastive = SupervisedContrastiveLoss(temperature=0.07)
        classification = nn.CrossEntropyLoss()
        
        # Contrastive loss ağırlığı
        contrastive_weight = 0.3
        
        # Custom criterion that combines both losses with weighting
        def combined_criterion(outputs, targets):
            # Handle different input cases - outputs can be single tensor or tuple
            if isinstance(outputs, tuple):
                logits, projections = outputs  # Unpack outputs from tuple
            else:
                # This is not expected, but handle gracefully to prevent crashes
                print("WARNING: Expected tuple output but got tensor. Using only classification loss.")
                logits = outputs
                projections = None
                
            # Always compute classification loss
            classification_loss = classification(logits, targets)
            
            # Only compute contrastive loss if projections are available
            if projections is not None:
                try:
                    contrastive_loss = supervised_contrastive(projections, targets)
                    # NaN kontrolü
                    if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
                        print("WARNING: Contrastive loss is NaN/Inf! Using only classification loss.")
                        contrastive_loss = torch.tensor(0.0, device=logits.device)
                except Exception as e:
                    print(f"Error in contrastive loss: {e}")
                    contrastive_loss = torch.tensor(0.0, device=logits.device)
                    
                # Toplam loss hesaplanırken güvenli ağırlıklandırma
                return classification_loss + contrastive_weight * contrastive_loss
            else:
                # No projections, just return classification loss
                return classification_loss
        
        return model, combined_criterion
    
    elif method == "multi_branch_cnn":
        model = MultiBranchCNN(num_classes=num_classes, dropout_rate=0.5)
        criterion = nn.CrossEntropyLoss()
        return model, criterion
    
    elif method == "cnn_lstm":
        model = CNNLSTM(num_classes=num_classes, input_channels=9, seq_length=128, dropout_rate=0.5)
        criterion = nn.CrossEntropyLoss()
        return model, criterion
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'cnn_attention', 'transformer_contrastive', 'multi_branch_cnn', or 'cnn_lstm'") 