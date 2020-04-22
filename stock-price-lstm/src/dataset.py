import torch
from torch.utils.data import Dataset

import os
import pandas as pd
from functools import reduce

class StockDataset(Dataset):
    def __init__(self, data_dir, interpolation='nearest', window_size=100):
        self.window_size = window_size

        # Hard coded path needs to be changed
        stock_data_list = []
        for fn in os.listdir(data_dir)[:10]:
            (series_name, file_type) = fn.split('.')
            if file_type != 'csv':
                continue
            file_path = os.path.join(data_dir, fn)

            df = pd.read_csv(file_path)
            df.set_index('Date', inplace=True)
            df = pd.to_numeric(df['Adj Close'], errors='coerce').to_frame()

            print(fn, df.dtypes)
            print(df.isnull().sum())
            # TODO: Implement other interpolation methods to use as hyperparameters.
            if interpolation == 'nearest':
                df = df.interpolate(method='nearest')
            elif interpolation == 'none':
                pass
            else:
                raise NotImplementedError('Only "nearest" interpolation method has been implemented.')

            df.columns = [series_name]
            stock_data_list.append(df)

        self.df_final = reduce(lambda left, right: pd.merge(left, right, how='outer',
                                                            left_index=True, right_index=True), stock_data_list)

    def __len__(self):
        return len(self.df_final) - self.window_size + 1

    def __getitem__(self, idx):
        X = torch.tensor(self.df_final.iloc[idx: idx+self.window_size, :].values)
        y = torch.tensor(self.df_final.iloc[idx+self.window_size+1, :].values)
        # data = torch.zeros(self.window_size, 2)
        # for i in range(0, self.window_size):
        #     data[i] = torch.tensor(self.df_final.iloc[idx + i, 0:])
        return X, y

if __name__ == "__main__":
    ds = StockDataset(r"E:\Datasets\YahooFinance\YahooSPData", interpolation='none')
    print(ds[0][0].shape)
    print(ds[0][1].shape)
