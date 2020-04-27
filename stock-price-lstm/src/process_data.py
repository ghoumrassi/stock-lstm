import os
import numpy as np
import pandas as pd
from functools import reduce


class ProcessData:
    def __init__(self, data_dir, save_dir='..\\data', interpolation='nearest'):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.stock_data = None
        self.returns_data = None

    def process(self):
        stock_data_list = []
        files_in_data_dir = os.listdir(self.data_dir)
        for fn in files_in_data_dir:
            (series_name, file_type) = fn.split('.')
            if file_type != 'csv':
                continue
            file_path = os.path.join(self.data_dir, fn)

            df = pd.read_csv(file_path)
            df.set_index('Date', inplace=True)
            df = df['Adj Close'].to_frame()
            df.columns = [series_name]
            stock_data_list.append(df)

        self.stock_data = reduce(lambda left, right: pd.merge(left, right, how='outer',
                                                            left_index=True, right_index=True), stock_data_list)

    def convert_returns(self, interpolate="zero"):
        self.returns_data = pd.DataFrame()
        for col in self.stock_data.columns:
            self.returns_data[col] = pd.to_numeric(
                np.log(self.stock_data[col]).diff(),
                errors='coerce'
            )

        if interpolate == 'zero':
            self.returns_data.fillna(0, inplace=True)
        else:
            raise NotImplementedError("Only zeros interpolation method has been implemented.")

    def save_data(self):
        self.returns_data.to_csv(
            os.path.join(self.save_dir, 'returns_data.csv')
        )
        self.stock_data.to_csv(
            os.path.join(self.save_dir, 'stock_data.csv')
        )

if __name__ == "__main__":
    process_data = ProcessData('E:/Datasets/YahooFinance/YahooSPData')
    process_data.process()
    process_data.convert_returns()
    process_data.save_data()