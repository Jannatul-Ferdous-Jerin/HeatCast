# HeatCast

HeatCast predicts future temperature conditions for heat-alert forecasting using historical meteorological time-series data. The project combines numerical time-series modeling with prompt-based language model embeddings to capture weather patterns that are useful for heat event prediction.

The main workflow is provided in `HeatCast.ipynb`. Supporting Python modules handle data loading, prompt-embedding generation, model training, evaluation, and attention visualization.

## Repository Structure

```text
.
|-- HeatCast.ipynb           # Main project notebook
|-- train.py                 # Model training and evaluation entry point
|-- visualize_attention.py   # Attention visualization utility
|-- data_provider/           # Dataset loaders
|-- dataset/                 # Place local input datasets here
|-- layers/                  # Neural network layers
|-- models/                  # HeatCast/TimeCMA model implementation
|-- scripts/                 # Shell scripts for embedding storage and training
|-- storage/                 # Prompt-embedding generation and storage utilities
|-- utils/                   # Metrics, masking, time features, and training helpers
|-- env_windows.yaml         # Conda environment for Windows
`-- env_ubuntu.yaml          # Conda environment for Ubuntu/Linux
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

The main workflow is documented in `HeatCast.ipynb`.

Place the required dataset file in `dataset/` before running the notebook or training scripts.

To run the training script directly:

```bash
python train.py --data_path heat-wave --seq_len 96 --pred_len 7 --batch_size 32 --num_nodes 15
```

To use the provided shell script on Linux:

```bash
bash scripts/heat-wave.sh
```

If prompt embeddings need to be regenerated, use the embedding storage utilities in `storage/` or the corresponding scripts in `scripts/`.
