import torch
import sys
import os
import time
import h5py
import argparse
from torch.utils.data import DataLoader
from data_provider.data_loader_save import Dataset_Custom
from gen_prompt_emb import GenPromptEmb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="ETTh1")
    parser.add_argument("--num_nodes", type=int, default=13)
    parser.add_argument("--input_len", type=int, default=96)
    parser.add_argument("--output_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--l_layers", type=int, default=12)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--divide", type=str, default="train")
    parser.add_argument("--num_workers", type=int, default=min(10, os.cpu_count()))
    parser.add_argument("--lead_time", type=int, default=2, help="Lead time in weeks") # <-- Add lead_time argument
    
    
    # Date range arguments
    parser.add_argument("--train_start", type=str, default="1/1/1990", help="Training start date")
    parser.add_argument("--train_end", type=str, default="12/31/2010", help="Training end date")
    parser.add_argument("--val_start", type=str, default="1/1/2011", help="Validation start date")
    parser.add_argument("--val_end", type=str, default="5/31/2014", help="Validation end date")
    parser.add_argument("--test_start", type=str, default="6/1/2014", help="Test start date")
    parser.add_argument("--test_end", type=str, default="12/31/2024", help="Test end date")
    
    return parser.parse_args()

def get_dataset(data_path, flag, input_len, output_len, train_dates, val_dates, test_dates):
    return Dataset_Custom(
            flag=flag, 
            size=[input_len, 0, output_len], 
            data_path=data_path,
            train_dates=train_dates,
            val_dates=val_dates,
            test_dates=test_dates,
            lead_time=args.lead_time # <-- Pass lead_time
        )

def save_embeddings(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Define date ranges
    train_dates = (args.train_start, args.train_end)
    val_dates = (args.val_start, args.val_end)
    test_dates = (args.test_start, args.test_end)
    
    # Create datasets with date ranges
    train_set = get_dataset(args.data_path, 'train', args.input_len, args.output_len, 
                           train_dates, val_dates, test_dates)
    test_set = get_dataset(args.data_path, 'test', args.input_len, args.output_len, 
                          train_dates, val_dates, test_dates)
    val_set = get_dataset(args.data_path, 'val', args.input_len, args.output_len, 
                         train_dates, val_dates, test_dates)

    data_loader = {
        'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers),
        'test': DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers),
        'val': DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    }[args.divide]

    gen_prompt_emb = GenPromptEmb(
        device=device, # type: ignore
        input_len=args.input_len,
        data_path=args.data_path,
        model_name=args.model_name,
        d_model=args.d_model,
        layer=args.l_layers,
        divide=args.divide
    ).to(device)
    
    print(args.data_path)
    save_path = f"./Embeddings/{args.data_path}/{args.divide}/"
    os.makedirs(save_path, exist_ok=True)

    emb_time_path = f"./Results/emb_logs/"
    os.makedirs(emb_time_path, exist_ok=True)

    for i, (x, y, x_mark, y_mark) in enumerate(data_loader):
        embeddings = gen_prompt_emb.generate_embeddings(x.to(device), x_mark.to(device))

        file_path = f"{save_path}{i}.h5"
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('embeddings', data = embeddings.cpu().numpy())

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(data_loader)} samples")
            
        # # Save and visualize the first sample
        # if i >= 0:
        #     break
    
if __name__ == "__main__":
    args = parse_args()
    t1 = time.time()
    save_embeddings(args)
    # print(torch.device(args.device if torch.cuda.is_available() else "cpu"))
    t2 = time.time()
    print(f"Total time spent: {(t2 - t1)/60:.4f} minutes")