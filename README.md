# HeatCast

HeatCast is a heat-alert forecasting project that predicts future temperature conditions from historical meteorological time-series data. The project uses a cross-modality forecasting pipeline that combines numerical time-series patterns with prompt-based language model embeddings to support heat event prediction.

The main workflow is provided in `HeatAlert.ipynb`. Supporting Python modules handle dataset loading, prompt-embedding generation, model training, evaluation, and attention visualization.

## Repository Structure

```text
.
├── HeatAlert.ipynb          # Main project notebook
├── train.py                 # Model training and evaluation entry point
├── visualize_attention.py   # Attention visualization utility
├── data_provider/           # Dataset loaders
├── layers/                  # Neural network layers
├── models/                  # HeatCast/TimeCMA model implementation
├── scripts/                 # Shell scripts for embedding storage and training
├── storage/                 # Prompt-embedding generation and storage utilities
├── utils/                   # Metrics, masking, time features, and training helpers
├── dataset/                 # Input dataset files
├── env_windows.yaml         # Conda environment for Windows
└── env_ubuntu.yaml          # Conda environment for Ubuntu/Linux
```

## Setup

Create the Conda environment for your platform:

```bash
conda env create -f env_windows.yaml
```

or:

```bash
conda env create -f env_ubuntu.yaml
```

Then activate the environment:

```bash
conda activate TimeCMA
```

## Usage

The main workflow is documented in `HeatAlert.ipynb`.

To run the training script directly:

```bash
python train.py --data_path heat-wave --seq_len 96 --pred_len 7 --batch_size 32 --num_nodes 15
```

To use the provided shell script on Linux:

```bash
bash scripts/heat-wave.sh
```

If prompt embeddings need to be regenerated, use the embedding storage utilities in `storage/` or the corresponding scripts in `scripts/`.
