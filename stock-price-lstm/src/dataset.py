import torch
from torch.utils.data import Dataset

import os
import pandas as pd
import numpy as np
from functools import reduce


class StockDataset(Dataset):
    def __init__(self, data_dir, device="cpu", interpolation='nearest', window_size=100, num_files=None,
                 start_date='2000-01-01', end_date='2019-01-01'):
        self.window_size = window_size
        self.device = device

        # Hard coded path needs to be changed
        stock_data_list = []
        files_in_data_dir = os.listdir(data_dir)
        if num_files:
            files_in_data_dir = files_in_data_dir[:num_files]
        for fn in files_in_data_dir:
            (series_name, file_type) = fn.split('.')
            if file_type != 'csv':
                continue
            file_path = os.path.join(data_dir, fn)

            df = pd.read_csv(file_path)
            df.set_index('Date', inplace=True)

            # Convert standardised stock price -> log(returns)
            df = pd.to_numeric(
                np.log(df['Adj Close']).diff(),
                errors='coerce'
            ).to_frame()

            # TODO: Implement other interpolation methods to use as hyperparameters.
            if interpolation == 'nearest':
                df = df.interpolate(method='nearest')
            elif interpolation == 'zero':
                df.fillna(0, inplace=True)
            elif interpolation == 'none':
                pass
            else:
                raise NotImplementedError('Only "nearest" interpolation method has been implemented.')

            df = df[(df.index >= start_date) & (df.index <= end_date)]

            df.columns = [series_name]
            stock_data_list.append(df)

        self.df_final = reduce(lambda left, right: pd.merge(left, right, how='outer',
                                                            left_index=True, right_index=True), stock_data_list)
        self.df_final = self.df_final.fillna(0)

    def __len__(self):
        return len(self.df_final) - self.window_size + 1

    def __getitem__(self, idx):
        X = torch.tensor(self.df_final.iloc[idx: idx + self.window_size, :].values, device=self.device)
        y = torch.tensor(self.df_final.iloc[idx + self.window_size + 1, :].values, device=self.device)

        # data = torch.zeros(self.window_size, 2)
        # for i in range(0, self.window_size):
        #     data[i] = torch.tensor(self.df_final.iloc[idx + i, 0:])
        return X, y

    @property
    def num_features(self):
        return self.df_final.shape[1]

    @property
    def names(self):
        return self.df_final.columns

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ds = StockDataset(r"E:\Datasets\YahooFinance\YahooSPData", interpolation='zero')
    print(ds.df_final.head())
    (ds.df_final.notnull().sum(axis=1) / ds.df_final.shape[1]).plot()
    plt.show()