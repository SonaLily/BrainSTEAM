# BrainSTEAM

A Graph Neural Network framework for brain connectivity analysis using fMRI data. This project applies dynamic edge convolution networks with self-attention graph pooling to classify neuroimaging data from autism (ABIDE) and healthy subjects (HCP).

## Overview

BrainSTEAM transforms fMRI time series data into brain connectivity graphs and uses graph neural networks to learn discriminative patterns for classification tasks. The framework combines:

- **Dynamic Edge Convolution**: Adapts to local graph structure using k-nearest neighbors
- **Self-Attention Graph Pooling (SAGPool)**: Learns to select the most informative brain regions
- **Multi-task Learning**: Joint optimization of classification, feature reconstruction, and degree prediction
- **MixUp Augmentation**: Improves generalization through graph-level data augmentation

## Architecture

### Data Pipeline

```
fMRI Time Series → Correlation Matrices → Graph Construction → GNN Processing → Classification
```

1. **Input**: fMRI time series data with 39 brain regions of interest (ROIs)
2. **Preprocessing**: Compute correlation and partial correlation matrices
3. **Graph Construction**:
   - Nodes: Brain ROIs with correlation matrix as features
   - Edges: Derived from partial correlation between regions
4. **GNN Processing**: 3 layers of DynamicEdgeConv with SAGPool
5. **Output**: Binary classification (e.g., ASD vs. Control)

### Model Variants

- **DynamicGNN**: Basic 3-layer edge convolution model
- **DynamicGNNdec**: Adds feature decoder and degree prediction for regularization
- **DynamicGNNdec2**: Alternative pooling strategy with max+mean aggregation

## Project Structure

```
.
├── models/
│   └── model_edgeconv.py          # GNN model architectures
├── layers/
│   └── SAGPool.py                 # Self-attention graph pooling layer
├── util/
│   └── prepossess.py              # MixUp augmentation and utilities
├── BrainDataset_st.py             # PyTorch Geometric dataset for brain graphs
├── BrainedgeconvdecstTrain.py     # Training and evaluation functions
├── main_abide_*.py                # Training scripts for ABIDE dataset
├── main_hcp_*.py                  # Training scripts for HCP dataset
└── LICENSE                        # License information
```

## Key Features

### Multi-Task Learning
The model jointly optimizes three objectives:
- **Classification**: Primary task (disease vs. control)
- **Feature Reconstruction**: Decoder reconstructs original node features (weight: 0.1)
- **Degree Prediction**: Predicts node degree distribution (weight: 0.1)

### MixUp Augmentation
Graph-level MixUp augmentation applied to both node features and edge structures to improve model generalization and reduce overfitting.

### Spatial-Temporal Processing
Scripts with `_st` suffix perform spatial-temporal analysis by:
- Randomly sampling temporal windows during training
- Building graphs from windowed fMRI segments
- Aggregating spatial patterns across time

## Training

### ABIDE Dataset (Autism Classification)

```python
# Basic model
python main_abide_edgeconv.py

# With MixUp augmentation
python main_abide_edgeconv_mixup.py

# Full model with decoder and spatial-temporal processing
python main_abide_edgeconv_mixup_dec_st.py
```

### HCP Dataset

```python
# Full model (recommended)
python main_hcp_edgeconv_mixup_dec_st_GOOD.py
```

### Model Configuration

Key hyperparameters (typically configured within each training script):
- `hidden_channels`: GNN hidden layer size
- `pooling_ratio`: 0.8 (retains 80% of nodes after pooling)
- `k`: 10 (number of nearest neighbors for edge convolution)
- `learning_rate`: Optimizer learning rate
- `n_epochs`: Number of training epochs
- `batch_size`: Training batch size

### Loss Formulation

```
Total Loss = Classification Loss + 0.1 × Feature Reconstruction Loss + 0.1 × Degree Prediction Loss
```

## Dependencies

- PyTorch
- PyTorch Geometric
- nilearn (neuroimaging utilities)
- networkx (graph construction)
- scikit-learn (metrics and cross-validation)
- numpy, pandas, matplotlib
- torch-sparse

## Evaluation Metrics

The framework computes comprehensive classification metrics:
- Accuracy
- Balanced Accuracy
- Sensitivity (Recall)
- Specificity
- Precision
- F1 Score
- ROC-AUC

Results are automatically saved to CSV files in `data/results/` with detailed experiment metadata.

## Implementation Details

### Graph Construction
- **Nodes**: 39 brain regions (ROIs)
- **Node Features**: 39×39 correlation matrix (flattened)
- **Edges**: All non-zero partial correlation connections
- **Edge Weights**: Partial correlation values

### Training Process
1. Load fMRI time series data
2. Randomly sample temporal window (for spatial-temporal models)
3. Compute correlation matrices using nilearn
4. Build graph structure with networkx
5. Convert to PyTorch Geometric Data object
6. Apply MixUp augmentation
7. Forward pass through GNN + decoders
8. Compute combined loss and backpropagate
9. Evaluate on validation/test set

### Reproducibility
Random seed is set to 42 throughout the codebase for reproducible results.

## Acknowledgments

This project incorporates modified code from the following open-source projects:

- **MixUp Implementation**: [mixup-cifar10](https://github.com/facebookresearch/mixup-cifar10) by Facebook Research (CC BY-NC 4.0)
- **ST-GCN Model**: [st-gcn](https://github.com/yysijie/st-gcn) by CUHK Multimedia Lab (BSD 2-Clause)
- **SAGPool**: [SAGPool](https://github.com/inyeoplee77/SAGPool) by Inyeop Lee

## License

See [LICENSE](LICENSE) file for details. Note that this project uses modified code from multiple sources with different licenses - please review individual component licenses.

## Citation

If you use this code in your research, please cite the original papers for MixUp, ST-GCN, and SAGPool, as well as any publications associated with this project.
