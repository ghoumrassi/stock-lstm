from src import StockDataset, LSTM

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TrainModel:
    def __init__(self, data_dir):

        self.data_dir = data_dir

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        num_files = 100
        num_epochs = 2
        hidden_dim = 50
        learning_rate = 0.01
        momentum = 0.1
        batch_size = 20
        interpolation = 'zero'

        train_start = '2005-01-01'
        val_start = '2017-01-01'
        val_end = '2019-01-01'

        self.train_data = StockDataset(data_dir, device, interpolation=interpolation, num_files=num_files,
                               start_date=train_start, end_date=val_start)
        self.val_data = StockDataset(data_dir, device, interpolation='zero', num_files=num_files,
                               start_date=val_start, end_date=val_end)

        num_features = self.train_data.num_features

        model = LSTM(input_dim=num_features, hidden_dim=hidden_dim, output_dim=num_features)
        model = model.double()
        model.to(device)
        print(torch.cuda.is_available())
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = nn.MSELoss()

        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=False, drop_last=True)
        val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False, drop_last=True)

        for epoch in range(num_epochs):
            print("Epoch: ", epoch+1)
            training_loss = 0
            val_loss = 0
            for X, y in train_loader:
                model.zero_grad()
                out = model(X)
                y_pred = out[:, -1, :].view(batch_size, -1)
                loss = criterion(y_pred, y)
                training_loss += loss.item()
                loss.backward()
                optimizer.step()

            with torch.set_grad_enabled(False):
                predictions_list = []
                for X, y in val_loader:
                    out = model(X)
                    y_pred = out[:, -1, :].view(batch_size, -1)
                    loss = criterion(y_pred, y)
                    val_loss += loss.item()
                    predictions_list.append(y_pred.data.numpy())
                self.predictions = np.vstack(predictions_list)
            print('Training Loss: %.4g' % training_loss)
            print('Validation Loss: %.4g' % val_loss)

    def show_graph(self, series_name):
        actual = self.val_data.df_final
        predictions = self.predictions
        cols = self.val_data.names

        predictions_frame = pd.DataFrame(predictions, columns=cols)
        fig, ax = plt.subplots(figsize=(8,8))
        predictions_frame[series_name].plot(ax=ax)
        actual[series_name].plot(ax=ax)
        ax.legend(["Prediction", "Actual"])
        plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Directory of stock price data.")
    args = parser.parse_args()

    data_dir = args.data_dir

    training = TrainModel(data_dir)
    training.show_graph('AAPL')