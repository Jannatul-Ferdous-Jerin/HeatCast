# HeatCast

HeatCast is a heat-alert forecasting project based on a cross-modality time-series forecasting pipeline. The original project notebook is `HeatAlert.ipynb`; the Python modules and scripts in this repository support data loading, prompt-embedding storage, model training, evaluation, and attention visualization.

This repository is prepared for paper-submission code sharing. Generated outputs, trained checkpoints, embedding files, and the manuscript PDF are intentionally excluded from version control.

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

## Data and Artifacts

Tracked:

- Source code and utility scripts
- Environment files
- Main notebook
- Dataset files required to reproduce the workflow

Ignored:

- `HeatCast_Elsevier.pdf`
- `Embeddings/`
- `Results/`
- `logs/`
- `graphs/`
- Generated prediction CSV files
- Model checkpoints
- Python cache and notebook checkpoint files

## License

See `LICENSE.txt`.
