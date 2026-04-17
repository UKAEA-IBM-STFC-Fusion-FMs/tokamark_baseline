# TokaMark Baseline

This is a **deep learning baseline** for the **TokaMark benchmark**, which focuses on fusion plasma prediction tasks using the MAST (Mega Ampere Spherical Tokamak) dataset. The architecture implements a multi-branch CNN encoder-decoder model for time-series prediction.

The code in this repository corresponds to the official implementation of the **TokaMark (CNN) baseline model** introduced in the 
paper [TokaMark: A Comprehensive Benchmark for MAST Tokamak Plasma Models](https://arxiv.org/abs/2602.10132) (submitted
to the 32nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2026).

Companion resources:
* <ins>**TokaMark:**</ins> A Python-based system for preprocessing fusion plasma data from the MAST (Mega Ampere Spherical 
Tokamak) facility and providing benchmark tasks for machine learning models.

---

## Environment Setup

1. Create and activate the required Conda environment and install the preprocessing packages from the **TokaMark** repository:

```bash
git clone <tokamark git repository>
cd <tokamark installation directory>
conda env create -f environment_basic.yml
conda activate tokamark-env
pip install -e .
```

2. Install required **TokaMark Baseline** dependencies:
```bash
pip install line-profiler==5.0.2 torchinfo==1.8.0
```

---

## High-Level Architecture Overview

```mermaid
graph TB
    subgraph "Data Pipeline"
        A[MAST Dataset] --> B[Data Preprocessing]
        B --> C[TimeCNNTransform]
        C --> D[DataLoader with Collation]
    end
    
    subgraph "Model Architecture"
        D --> E[Multi-Branch Encoder]
        E --> F[Backbone Network]
        F --> G[Multi-Branch Decoder]
        G --> H[Predictions]
    end
    
    subgraph "Training & Evaluation"
        H --> I[MultiOutputMSELoss]
        I --> J[BatchStepTrainer]
        J --> K[Model Checkpointing]
        K --> L[Evaluator]
    end
    
    style A fill:#e1f5ff
    style E fill:#ffe1e1
    style F fill:#ffe1e1
    style G fill:#ffe1e1
    style J fill:#e1ffe1
```

## Core Components

### 1. **Entry Points** (`run_training.py` & `run_evaluation.py`)

These scripts orchestrate the entire pipeline:

- **Training**: Initializes datasets, creates model architecture, and runs batch-step training
- **Evaluation**: Loads trained models, performs inference, and computes metrics

Key configuration:
- Task selection (e.g., `task_1-1`, `task_4-5`)
- CNN configuration via YAML files
- Seed management for reproducibility

### 2. **Data Transformation** (`src/time_cnn_transform.py`)

The `TimeCNNTransform` class prepares data for the CNN:

```python
# Transforms raw data into structured format:
{
    'x': [input_signals + actuator_signals],  # Model inputs
    'y': [output_signals]                      # Prediction targets
}
```

**Key features:**
- Handles temporal windowing for Markovian vs non-Markovian tasks
- Adds channel dimension via `np.expand_dims()`
- Moves time axis to first position with `np.moveaxis()`

### 3. **Model Architecture** (`src/time_cnn_model.py`)

#### Multi-Branch CNN Model

The `MultiBranchTimeCNNModel` is the core architecture:

```mermaid
graph LR
    subgraph "Input Branches"
        I1[1D Encoder] --> M[Merge]
        I2[2D Encoder] --> M
        I3[3D Encoder] --> M
    end
    
    subgraph "Backbone"
        M --> B1[Linear + ReLU]
        B1 --> B2[Linear + ReLU]
        B2 --> B3[Linear + ReLU]
    end
    
    subgraph "Output Branches"
        B3 --> O1[1D Decoder]
        B3 --> O2[2D Decoder]
        B3 --> O3[3D Decoder]
    end
    
    style M fill:#ffe1e1
    style B2 fill:#e1ffe1
```

**Encoder Types:**
- **`Conv1DEncoder`**: For time series (shape: `[channels, time]`)
- **`Conv2DEncoder`**: For profiles over time (shape: `[channels, time, height]`)
- **`Conv3DEncoder`**: For images over time (shape: `[channels, time, height, width]`)

**Decoder Types:**
- **`Conv1DDecoder`**: Reconstructs 1D outputs
- **`Conv2DDecoder`**: Reconstructs 2D profiles
- **`Conv3DDecoder`**: Reconstructs 3D volumes

**Architecture Details:**
- Each encoder compresses inputs to a fixed dimension `D` (default: 16)
- Backbone network: 3-layer MLP with dropout (0.2)
- Uses gradient checkpointing to reduce memory usage
- Supports heterogeneous input/output shapes

### 4. **Training System** (`src/trainer.py`)

#### Batch-Step Training

The `BatchStepTrainer` implements incremental training:

```python
# Training loop structure:
while trainer.step_batch():
    # Single batch update
    # Validation every N batches
    # Early stopping check
```

**Key Features:**
- **`MultiOutputMSELoss`**: Handles multiple output branches
- **`cnn_collate_fn`**: Custom collation with NaN handling
- **Validation frequency**: Configurable (default: every 100 batches)
- **Early stopping**: Based on validation loss with patience
- **Optimizer**: Adam with weight decay (1e-4)
- **History tracking**: Saves training/validation loss per step

### 5. **Evaluation System** (`src/evaluator.py`)

Two main evaluation functions:

#### `cnn_unstd_evaluation_per_shot`
- Unstandardizes predictions using saved mean/std
- Computes metrics per window and feature
- Uses `WindowMetricsAccumulator` from MAST benchmark

#### `cnn_safety_vizu_per_shot`
- Generates visualizations for first N shots
- Handles different output dimensionalities (1D, 2D, 3D, 4D)
- Creates comparison plots (true vs predicted)

### 6. **Utilities** (`utils.py` & `globals.py`)

- **`set_seed()`**: Ensures reproducibility across Python, NumPy, PyTorch
- **`seed_worker()`**: Seeds DataLoader workers
- **`globals.py`**: Manages repository paths dynamically

## Data Flow

```mermaid
sequenceDiagram
    participant D as MAST Dataset
    participant T as TimeCNNTransform
    participant L as DataLoader
    participant M as CNN Model
    participant Tr as Trainer
    
    D->>T: Raw shot data
    T->>T: Window segmentation
    T->>L: Transformed windows
    L->>L: Collate batch
    L->>M: Batched tensors
    M->>M: Forward pass
    M->>Tr: Predictions
    Tr->>Tr: Compute loss
    Tr->>M: Backpropagation
    Tr->>Tr: Update weights
```

## Task Configuration

The system supports multiple task types defined in YAML configs:

- **Task 1-x**: Single-step prediction (Markovian)
- **Task 2-x**: Multi-step prediction
- **Task 3-x**: Sequence-to-sequence
- **Task 4-x**: Long-horizon prediction (non-Markovian)

Each task has different:
- Input shapes (combinations of 1D/2D/3D signals)
- Output shapes (scalars, profiles, images)
- Temporal windows and stride settings

## Memory Optimization

The architecture employs several memory-saving techniques:

1. **Gradient Checkpointing**: Used in encoder/decoder forward passes
2. **Mixed Precision**: Configurable (commented out in current version)
3. **Batch Size Management**: Configurable via YAML
4. **Pin Memory**: Enabled for faster GPU transfer
5. **Worker Processes**: Multiprocessing for data loading

## Configuration Files

Located in `src/config/`:
- `config_cnn_test.yaml`: Quick testing configuration
- `config_cnn_iterable_lr_4_work_4.yaml`: Production configuration

Configuration includes:
- Model hyperparameters (D, layers, kernel sizes)
- Training settings (learning rate, max steps, patience)
- DataLoader settings (batch size, workers)
- Path configurations

## Dependencies

The system integrates with the **tokamark** package for:
- Dataset initialization
- Task configuration
- Metric computation
- Data splitting (train/val/test)

## Usage

### Training
```bash
python run_training.py --task task_1-1 --config_cnn /src/config/config_cnn_test.yaml --seed 23
```

### Evaluation
```bash
python run_evaluation.py --task task_1-1 --config_cnn /src/config/config_cnn_iterable_lr_4_work_4.yaml --seed 23
```

## Architecture Summary

This architecture provides a flexible, scalable baseline for fusion plasma prediction tasks with support for heterogeneous multi-modal time-series data. The multi-branch design allows the model to process different types of diagnostic signals (1D time series, 2D profiles, 3D images) simultaneously and produce predictions in various formats matching the physical quantities of interest.

## Companion Resources

| Resource | Link |
|---|---|
| TokaMark paper | [arXiv:2602.10132](https://arxiv.org/abs/2602.10132) |
| TokaMark repository | [UKAEA-IBM-STFC-Fusion-FMs/tokamark](https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark) |

## Citing TokaMark

TokaMark has been submitted to the *32nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2026*, and it is
currently being reviewed. A preprint version of the manuscript is available [here](https://arxiv.org/abs/2602.10132).

If you use TokaMark, please cite our work as:

    @article{rousseau2026tokamark,
      title={TokaMark: A Comprehensive Benchmark for MAST Tokamak Plasma Models},
      author={
        Rousseau, C{\'e}cile and Jackson, Samuel and Ordonez-Hurtado, Rodrigo H. and
        Amorisco, Nicola C. and Boschi, Tobia and Holt, George K and Loreti, Andrea and 
        Sz{\'e}kely, Eszter and Whittle, Alexander and Agnello, Adriano and Pamela, Stanislas and 
        Pascale, Alessandra and Akers, Robert and Bernabe Moreno, Juan and Thorne, Sue and 
        Zayats, Mykhaylo
      },
      journal={arXiv preprint arXiv:2602.10132},
      year={2026}
    }