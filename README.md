# Human Activity Recognition with Multiple Deep Learning Approaches

This project implements four different deep learning approaches for Human Activity Recognition (HAR) using accelerometer and gyroscope sensor data from smartphones:

1. **CNN+Attention**: Combination of Convolutional Neural Network and Spatial Attention mechanism (Focal Loss)
2. **Transformer+Contrastive**: TS-TCC-like Transformer encoder architecture and Supervised Contrastive Learning (Contrastive+CE Loss)
3. **Multi-Branch CNN**: CNN architecture that processes accelerometer and gyroscope data through separate branches (CE Loss)
4. **CNN+LSTM**: Architecture combining 1D CNN and LSTM layers to process time series data (CE Loss)

All four approaches are used to classify 6 different activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) and their results can be compared.

## Dataset

The project uses the [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), which includes:
- Data collected from 30 volunteers
- 6 different activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)
- 561-feature vector
- 10,299 samples

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd <project_directory>
```

2. (Recommended) Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
# or
.\venv\Scripts\activate  # For Windows
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the UCI HAR Dataset and extract it into the project folder. The folder structure should be:
```
project_root/
├── UCI HAR Dataset/
│   ├── train/
│   │   ├── X_train.txt
│   │   └── y_train.txt
│   └── test/
│       ├── X_test.txt
│       └── y_test.txt
├── model.py
├── train.py
├── dataloader.py
└── main.py
```

## Usage

### Training with a Single Method

To run a specific method from the command line:

```bash
# CNN+Attention in development mode
python main.py --method cnn_attention --dev

# Transformer+Contrastive with full training
python main.py --method transformer_contrastive

# Multi-Branch CNN with full training
python main.py --method multi_branch_cnn

# CNN+LSTM with full training
python main.py --method cnn_lstm
```

### Comparing Methods

To run and compare all methods:

```bash
# Comparison in development mode (faster)
python main.py --method compare --dev

# Comparison with full dataset (more accurate results)
python main.py --method compare
```

This command trains all methods sequentially, compares accuracy rates, and generates a comparison chart.

## Model Architectures

### 1. CNN+Attention Model
- **CNN**: 3 convolutional layers (32, 64, 128 filters)
- **Attention**: Spatial attention after each convolutional block
- **Dropout**: Progressive dropout strategy (0.3, 0.42, 0.6)
- **Fully Connected**: 512-neuron intermediate layer
- **Output**: Softmax for 6 classes
- **Loss**: Focal Loss (γ=2.0)

### 2. Transformer+Contrastive Model
- **Input Projection**: Converts input data to embedding dimension
- **Positional Encoding**: Position information for transformer
- **Encoder**: 3 encoder layers (Multi-head Attention + Feed Forward)
- **Class Token**: Special token for classification
- **Projection Head**: Projection for contrastive learning
- **Loss**: Supervised Contrastive Loss + Cross Entropy Loss

### 3. Multi-Branch CNN Model
- **Dual Branch Architecture**:
  - **Branch 1**: CNN branch processing ACC + GYRO (6 channels) data
  - **Branch 2**: CNN branch processing TOTAL_ACC (3 channels) data
- **Each Branch**: 3 convolutional layers + batch normalization + pooling
- **Merge Layer**: Layer combining the two branches
- **Fully Connected**: 512-neuron merge layer
- **Loss**: Cross Entropy Loss (CE Loss)
- **Input Format**: (batch_size, 9, 128) - First 6 channels for acc+gyro, last 3 channels for total_acc

### 4. CNN+LSTM Model
- **CNN Frontend**: 1D CNN layers for feature extraction from raw signals
- **LSTM Core**: Bidirectional LSTM to capture temporal dependencies
- **Fully Connected**: Classification layers
- **Loss**: Cross Entropy Loss (CE Loss)
- **Input Format**: (batch_size, 9, 128) - Time series data

## Data Preprocessing

Each model requires different data formats:

1. **CNN+Attention and Transformer**: 2D tensor in shape (batch_size, 1, 33, 17)
2. **Multi-Branch CNN and CNN+LSTM**: 1D tensor in shape (batch_size, 9, 128)
   - First 6 channels (0-5): accelerometer and gyroscope data (2×3 axes)
   - Last 3 channels (6-8): total acceleration data

## Output Directory Structure

A separate directory is created for each training run:

```
output/
├── run_YYYYMMDD_HHMMSS_[dev/full]_[method]/  # Unique folder for each run
│   ├── models/                                # Trained model files
│   │   └── method_model_[dev/full].pth        # Model state and parameters
│   ├── plots/                                 # Visualizations
│   │   ├── loss_plot_[method].png             # Training and validation loss graphs
│   │   ├── accuracy_plot_[method].png         # Test accuracy graph
│   │   ├── confusion_matrix_[method].png      # Confusion matrix visualization
│   │   ├── classification_report_[method].txt  # Detailed classification report
│   │   └── metrics_[method].csv               # Detailed metrics
│   └── logs/                                  # Training logs
│       └── training_log.txt                   # Detailed information for each epoch
│
└── comparison_YYYYMMDD_HHMMSS/                # Comparison study folder
    ├── method_comparison.png                  # Comparison chart of all methods
    └── comparison_results.txt                 # Comparison results and best method
```

## Development Notes

- In development mode, 5% of the dataset is used
- Number of epochs is 10 in development mode, 50 in full training
- Early stopping: 2 epochs in dev mode, 3 epochs in full mode
- Each run includes the following mechanisms:
  - Learning rate scheduling (ReduceLROnPlateau or OneCycleLR)
  - Weight decay (L2 regularization)
  - Dropout adjustment
  - Stability optimizations
  - Gradient clipping

## Performance Evaluation

The following metrics are calculated and compared for each method:
- Accuracy (%)
- Confusion Matrix
- Precision, Recall, F1-Score (per class)
- Training/validation loss curves

## Method Comparison

The outputs created for comparing methods are:
- Bar chart: Visual comparison of accuracy rates for each method
- Text report: Accuracy rates for each method and the best performing method
- Detailed performance metrics: Confusion matrix and classification reports for each method

## Contributing

You can send pull requests to add new features, fix bugs, or improve models.

## License

This project is licensed under the MIT License. See the LICENSE file for details.