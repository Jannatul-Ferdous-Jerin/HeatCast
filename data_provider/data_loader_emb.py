import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils.tools import StandardScaler
from utils.timefeatures import time_features
import warnings
import h5py
warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, root_path="/home/rmedu/Music/Heat_Alert/dataset", flag='train', size=None, features='M', data_path='23p75__90p50',
                 target='Temperature(C)', scale=True, timeenc=0, freq='d', patch_len=16, percent=100, model_name="gpt2",
                 train_dates=("1/1/1990", "12/31/2010"),
                 val_dates=("1/1/2011", "5/31/2014"),
                 test_dates=("6/1/2014", "12/31/2024"),
                 lead_time=2):  # <-- Add lead_time parameter
        
        self.percent = percent
        self.patch_len = patch_len
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        self.lead_time = lead_time # <-- Store lead_time
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # Store date ranges
        self.train_dates = train_dates
        self.val_dates = val_dates
        self.test_dates = test_dates

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        
        

        if not data_path.endswith('.csv'):
            data_path_file = data_path
            data_path += '.csv' 
   
        self.data_path = os.path.join(root_path, data_path)
        self.data_path_file = data_path_file
        self.model_name = model_name
        self.embed_path = f"/home/rmedu/Music/Heat_Alert/Embeddings/{data_path_file}/{flag}/"

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        
        # --- START OF FIX ---
        # Convert date column to datetime objects
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        # Filter dataframe based on the flag ('train', 'val', or 'test')
        if self.set_type == 0: # train
            start_date, end_date = pd.to_datetime(self.train_dates[0]), pd.to_datetime(self.train_dates[1])
        elif self.set_type == 1: # val
            start_date, end_date = pd.to_datetime(self.val_dates[0]), pd.to_datetime(self.val_dates[1])
        else: # test
            start_date, end_date = pd.to_datetime(self.test_dates[0]), pd.to_datetime(self.test_dates[1])
        
        # Apply the date filter
        df_raw = df_raw[(df_raw['date'] >= start_date) & (df_raw['date'] <= end_date)]
        
        # The old percentage-based split is no longer needed.
        # We use the entire filtered dataframe.
        border1 = 0
        border2 = len(df_raw)
        # --- END OF FIX ---
        
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            # For scaling, we still need to fit on the original training data range
            full_df = pd.read_csv(os.path.join(self.root_path, self.data_path))
            full_cols_data = full_df.columns[1:]
            
            # Use the original training date range to define the scaler's data
            train_start_date, train_end_date = pd.to_datetime(self.train_dates[0]), pd.to_datetime(self.train_dates[1])
            full_df['date'] = pd.to_datetime(full_df['date'])
            scaler_train_data = full_df[(full_df['date'] >= train_start_date) & (full_df['date'] <= train_end_date)][full_cols_data]
            
            self.scaler.fit(scaler_train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']] # No need for [border1:border2] as df_raw is already filtered
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['year'] = df_stamp.date.apply(lambda row: row.year)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        
        # New logic for r_begin and r_end based on lead_time
        gap_days = (self.lead_time - 1) * 7 # Gap in days
        r_begin = s_end + gap_days
        r_end = r_begin + self.pred_len
        
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        embeddings_stack = []
        file_path = os.path.join(self.embed_path, f"{index}.h5")
        if os.path.exists(file_path):
            with h5py.File(file_path, 'r') as hf:
                data = hf['embeddings'][:]
                tensor = torch.from_numpy(data)
                embeddings_stack.append(tensor.squeeze(0))
        else:
            raise FileNotFoundError(f"No embedding file found at {file_path}")
                
        embeddings = torch.stack(embeddings_stack, dim=-1)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, embeddings

    def __len__(self):
        total_length_needed = self.seq_len + self.pred_len + (self.lead_time - 1) * 7
        return len(self.data_x) - total_length_needed + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)