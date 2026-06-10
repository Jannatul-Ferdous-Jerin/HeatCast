# visualize_attention.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from types import SimpleNamespace
import os

# Assuming your project structure allows these imports
from models.TimeCMA import Dual
from data_provider.data_loader_emb import Dataset_Custom
from torch.utils.data import DataLoader

def visualize_attention():
    # --- 1. Setup: Use the same configuration as your training script ---
    args = SimpleNamespace(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        data_path='23p75__90p50',
        channel=64,
        num_nodes=13,
        seq_len=96,
        pred_len=96,
        dropout_n=0.4,
        d_llm=768,
        e_layer=1,
        d_layer=1,
        head=8,
        lead_time=3,
        batch_size=8, # A smaller batch is fine for visualization
        num_workers=0,
        # These are needed for model instantiation
        learning_rate=1e-5,
        weight_decay=1e-4,
    )

    # --- 2. Load Your Trained Model ---
    device = torch.device(args.device)
    model = Dual(
        device=device, channel=args.channel, num_nodes=args.num_nodes, seq_len=args.seq_len,
        pred_len=args.pred_len, dropout_n=args.dropout_n, d_llm=args.d_llm, e_layer=args.e_layer,
        d_layer=args.d_layer, head=args.head
    ).to(device)

    # ❗️ UPDATE THIS PATH to your saved model
    model_path = "/home/rmedu/Music/Heat_Alert/logs/23p75__90p50/23p75__90p50/96_64_1_2_1e-05_0.4_2024/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Model path not found at {model_path}")
        print("Please update the 'model_path' variable in the script.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # --- 3. Load a Batch of Data ---
    test_set = Dataset_Custom(
        flag='test', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path,
        train_dates=("1/1/1990", "12/31/2010"), val_dates=("1/1/2011", "5/31/2014"),
        test_dates=("6/1/2014", "12/31/2024"), lead_time=args.lead_time
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    print("Test data loaded.")

    # --- 4. Get Predictions and Attention Weights ---
    x, y, x_mark, y_mark, embeddings = next(iter(test_loader))
    x, x_mark, embeddings = x.to(device), x_mark.to(device), embeddings.to(device)

    with torch.no_grad():
        # Call the forward pass with the new flag
        output, attention_weights = model(x, x_mark, embeddings, return_attn=True)

    print(f"Model output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    # Expected shape: [batch_size, num_heads, query_len, key_len]
    # For your model: [batch_size, 1, channel, d_llm] -> e.g., [8, 1, 256, 768]

    # --- 5. Plot the Attention Map ---
    # Plot for the first sample in the batch and the first (and only) head
    sample_idx = 0
    head_idx = 0
    
    attention_map = attention_weights[sample_idx, head_idx, :, :].cpu().numpy()
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(attention_map, cmap='viridis', robust=True)
    plt.title(f'Cross-Modal Attention Map (Sample {sample_idx}, Head {head_idx})', fontsize=16)
    plt.xlabel('Text Embedding Features (Keys)', fontsize=12)
    plt.ylabel('Time Series Features (Queries)', fontsize=12)
    plt.show()


if __name__ == '__main__':
    visualize_attention()