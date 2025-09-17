# SADH: State-Aware Dynamic Hypergraph Learning



###  SADH Core Algorithm
- **State-Aware Mechanism**: Dynamically adjusts hypergraph weights to adapt to data state changes
- **Dynamic Hypergraph Construction**: Intelligent hyperedge generation based on k-NN and category information
- **Multi-View Fusion**: Effectively integrates complementary information from different views
- **Missing Data Handling**: Robust processing of view missing and noisy data

### Supported Datasets
- **Handwritten**: 6-view handwritten digit recognition dataset
- **Scene15**: 15-class scene classification dataset
- **BRCA**: Breast cancer gene expression dataset
- **Animal**: 10-class animal classification dataset, containing 10,158 images across 50 animal classes with two feature views.
- **Scene15**: The 15-Scenes dataset comprises 4,485 images across 15 categories, with multiple visual feature views (e.g., GIST/PHOG).
- **Custom Datasets**: Supports .mat format multi-view data



## Project Structure
                
├── README.md                    
├── requirements.txt                # Project dependencies
├── SADH/                          # SADH original implementation
│   ├── main.py                    # Main entry script
│   ├── src/                       # Source code
│   │   ├── models/                # Model definitions
│   │   ├── trainers/              # Trainers
│   │   ├── utils/                 # Utility functions
│   │   └── experiments/           # Experiment scripts
│   └── configs/                   # Configuration files




### Requirements

- Python 3.7+
- PyTorch 1.9.0+
- Other dependencies see `requirements.txt`


### Installation
# Clone the project
git clone <repository-url>
cd UBE-main

# Install dependencies
pip install -r requirements.txt


### Running Experiments

#### 1. SADH Complete Experiment
```bash
cd SADH
python main.py --experiment all --dataset handwritten





### Hyperedge Types
1. **Instance-View Hyperedges (IVH)**: Connect different views of the same instance
2. **Category Hyperedges (CH)**: Connect instances of the same category
3. **View-View Hyperedges (VVH)**: Connect similar instances across different views
4. **Low-Confidence Hyperedges (LCH)**: Handle uncertain instance relationships
5. **Information-Transfer Hyperedges (ITH)**: Promote information transfer between views


## Configuration

### Main Parameters
```yaml
# Model parameters
embed_dim: 128          # Embedding dimension
learning_rate: 0.001    # Learning rate
epochs: 100             # Training epochs
dropout: 0.5            # Dropout rate

# Hypergraph parameters
k: 10                   # k-NN neighborhood size
method: 'knn'           # Hypergraph construction method
use_state_aware: true   # Whether to use state awareness

# Data parameters
missing_rate: 0.0       # Missing rate
smote: false            # Whether to use SMOTE augmentation
```


## Extension Features

### Custom Datasets
Supports adding new multi-view datasets with the following format:
```python
# Data format requirements
X_views: List[np.ndarray]  # Multi-view feature list
y: np.ndarray              # Labels
mask: np.ndarray           # Missing view mask





### Common Issues
1. **PyTorch installation failed**: Ensure using 64-bit Python
2. **Insufficient memory**: Reduce batch_size or embed_dim
3. **Dataset path error**: Check if dataset files exist
4. **Dependency conflicts**: Use virtual environment to isolate dependencies





## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



**Project Status**: Active Development  
**Version**: v2.0.0
