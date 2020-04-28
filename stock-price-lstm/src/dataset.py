import torch
from torch.utils.data import Dataset

import os
import pandas as pd
import numpy as np
from functools import reduce

from src.process_data import ProcessData


class StockDataset(Dataset):
    def __init__(self, device="cpu", window_size=365, prediction_lag=1,
                 start_date='2000-01-01', end_date='2019-01-01', reload_data=False,
                 data_dir='E:/Datasets/YahooFinance/YahooSPData'):
        self.window_size = window_size
        self.prediction_lag = prediction_lag
        self.device = device

        if reload_data:
            process_data = ProcessData(data_dir)
            process_data.process()
            process_data.convert_returns()
            process_data.save_data()

        returns_path = '../data/returns_data.csv'

        self.data = pd.read_csv(returns_path)
        self.data.set_index('Date', inplace=True)

        self.data = self.data[self.data.index >= start_date]
        if end_date:
            self.data = self.data[self.data.index <= end_date]

    def __len__(self):
        return len(self.data) - (self.window_size + self.prediction_lag) + 1

    def __getitem__(self, idx):
        X = torch.tensor(self.data.iloc[idx: idx + self.window_size, :].values, device=self.device)
        y = torch.tensor(self.data.iloc[
                         idx + self.window_size: idx + self.window_size + self.prediction_lag, :
                    ].values, device=self.device).sum(dim=0)
        return X, y

    def target_dates(self, start=0, end=-1):
        y_start = start + self.window_size + self.prediction_lag
        return self.data.index[y_start: y_start + end]

    @property
    def num_features(self):
        return self.data.shape[1]

    @property
    def names(self):
        return self.data.columns

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ds = StockDataset()
    print(ds.data.head())
    ds.data.iloc[:,0].plot()
    plt.show()