# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BrainSTEAM is a Graph Neural Network (GNN) research project for brain connectivity analysis using fMRI data. The project applies dynamic edge convolution networks with self-attention graph pooling (SAGPool) to classify brain imaging data from ABIDE (Autism Brain Imaging Data Exchange) and HCP (Human Connectome Project) datasets.

## Core Architecture

### Model Pipeline
1. **Data Processing**: fMRI time series → correlation matrices → graph structures
2. **Graph Construction**: Uses correlation and partial correlation to build brain connectivity graphs
3. **GNN Processing**: 3-layer DynamicEdgeConv with SAGPool between layers
4. **Decoder Branch**: Feature reconstruction decoder + degree prediction decoder for regularization
5. **Augmentation**: MixUp augmentation applied to node features and edge indices
6. **Classification**: Binary classification (typically disease vs. control)

### Key Components

**models/model_edgeconv.py**: Contains three GNN architectures
- `DynamicGNN`: Basic 3-layer dynamic edge convolution model
- `DynamicGNNdec`: Model with feature decoder and degree decoder for multi-task learning
- `DynamicGNNdec2`: Variant with different pooling strategy (concatenates max and mean pooling)

**BrainDataset_st.py**: Custom PyTorch Geometric dataset class
- Converts fMRI time series to correlation/partial correlation matrices
- Constructs graph Data objects with node features (correlation) and edges (partial correlation)
- Processes data on-the-fly for spatial-temporal analysis
- Creates temporary .pyg directories (automatically cleaned up)

**BrainedgeconvdecstTrain.py**: Training and evaluation functions
- `brainedgedecsttrain()`: Main training loop with MixUp augmentation
- `test_test()`: Evaluation on test set
- `test_train()`: Evaluation on training set
- `compute_metrics()`: Calculates accuracy, sensitivity, specificity, balanced accuracy, ROC-AUC, precision, recall, F1
- Loss combines: classification loss + feature reconstruction (alpha=0.1) + degree prediction (beta=0.1)

**layers/SAGPool.py**: Self-Attention Graph Pooling layer
- Learns to select important nodes based on graph structure
- Returns both pooled features and permutation indices for decoder

**util/prepossess.py**: Data augmentation utilities
- `mixup_data2()`: MixUp augmentation for graph data (mixes node features and edge indices)
- `mixup_criterion()`: Weighted loss for MixUp training

## Training Scripts

### ABIDE Dataset Training
- `main_abide_edgeconv.py`: Basic EdgeConv model
- `main_abide_edgeconv_mixup.py`: With MixUp augmentation
- `main_abide_edgeconv_mixup_dec.py`: With MixUp + decoder branches
- `main_abide_edgeconv_mixup_dec_st.py`: Full model with spatial-temporal processing

### HCP Dataset Training
- `main_hcp_edgeconv.py`: Basic EdgeConv model
- `main_hcp_edgeconv_mixup.py`: With MixUp augmentation
- `main_hcp_edgeconv_mixup_dec.py`: With MixUp + decoder branches
- `main_hcp_edgeconv_mixup_dec_st_GOOD.py`: Full model (marked as "GOOD" - likely best performing)

## Important Implementation Details

### Graph Construction
- **Node features**: 39 x 39 correlation matrix (for 39 ROIs)
- **Edges**: Derived from partial correlation matrix
- **Edge selection**: All non-zero partial correlations are included as edges
- Dynamic edge convolution uses k=10 nearest neighbors

### Model Parameters
- `hidden_channels`: Controls GNN layer width (typically tuned per experiment)
- `pooling_ratio`: 0.8 (keeps 80% of nodes after each pooling layer)
- `dropout_ratio`: 0 (no dropout used)
- `k`: 10 (number of neighbors for dynamic edge convolution)

### Training Configuration
- Binary cross-entropy loss for classification
- MSE loss for feature reconstruction
- Degree prediction uses cross-entropy
- Loss weights: classification (1.0) + feature_reconstruction (0.1) + degree_prediction (0.1)
- Batch size and epochs vary per script
- Random seed: 42 (set for reproducibility)

### Data Flow
1. Load fMRI time series data (shape: [subjects, 1, time_steps, ROIs, 1])
2. Random temporal window selection during training
3. Convert to correlation matrices using nilearn
4. Build graph with networkx
5. Convert to PyTorch Geometric Data object
6. Apply MixUp augmentation
7. Forward pass through GNN + decoders
8. Compute combined loss
9. Temporary .pyg directories are deleted after use (via `shutil.rmtree()`)

## Dependencies
- PyTorch + PyTorch Geometric
- nilearn (brain imaging utilities)
- networkx (graph construction)
- scikit-learn (metrics, cross-validation)
- pandas (results tracking)
- torch-sparse

## Attribution
This project incorporates modified code from:
- MixUp implementation (Facebook Research, CC BY-NC 4.0)
- ST-GCN model (CUHK Multimedia Lab, BSD 2-Clause)
- SAGPool implementation (Inyeop Lee)
