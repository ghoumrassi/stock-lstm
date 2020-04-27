from src import StockDataset, LSTM

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TrainModel:
    def __init__(self, data_dir, model_path, save_path, reload):

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        num_epochs = 150
        hidden_dim = 50
        learning_rate = 0.1
        momentum = 0.1
        batch_size = 20
        prediction_lag = 3

        train_start = '2002-01-01'
        val_start = '2017-01-01'
        val_end = '2019-01-01'

        self.train_data = StockDataset(device=device, start_date=train_start, end_date=val_start,
                                       prediction_lag=prediction_lag, reload_data=reload, data_dir=data_dir)
        self.val_data = StockDataset(device=device, start_date=val_start, end_date=val_end,
                                     prediction_lag=prediction_lag)

        num_features = self.train_data.num_features

        model = LSTM(input_dim=num_features, hidden_dim=hidden_dim, output_dim=num_features)
        model = model.double()

        # Load model if path specified
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

        model.to(device)
        print(torch.cuda.is_available())
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = nn.MSELoss()

        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=False, drop_last=True)
        val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False, drop_last=True)

        self.loss_dict = {'train': [], 'val': []}
        for epoch in range(num_epochs):
            if (epoch + 1) % 50 == 0:
                print("Learning rate reduced.")
                learning_rate *= 0.1
            print("Epoch: ", epoch + 1)
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
                    if device == "cpu":
                        predictions_list.append(y_pred.data.numpy())
                    else:
                        predictions_list.append(y_pred.data.cpu().numpy())
                self.predictions = np.vstack(predictions_list)
            print('Training Loss: %.4g' % training_loss)
            print('Validation Loss: %.4g' % val_loss)
            self.loss_dict['train'].append(training_loss)
            self.loss_dict['val'].append(val_loss)

            # Save model
            torch.save(model.state_dict(), save_path)

        # Save predictions
        self.val_data.data.to_csv('val_data.csv')
        pd.DataFrame(
            self.predictions,
            columns=self.val_data.names
        ).to_csv('predictions.csv')


    def show_graph(self, series_name):
        actual = self.val_data.data
        predictions = self.predictions

        cols = self.val_data.names

        predictions_frame = pd.DataFrame(predictions, columns=cols)
        predictions_frame.index = self.val_data.target_dates()

        fig, ax = plt.subplots(figsize=(8, 8))
        predictions_frame[series_name].plot(ax=ax)
        actual[series_name].plot(ax=ax)
        ax.legend(["Prediction", "Actual"])
        plt.show()

    def loss_graph(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(self.loss_dict['train'])
        ax.plot(self.loss_dict['val'])
        ax.legend(["Prediction", "Actual"])
        plt.show()

if __name__ == "__main__":
    import argparse
    import time

    ts = int(time.time())

    parser = argparse.ArgumentParser()

    parser.add_argument("-l", "--load-model", dest="load_model", default=None,
                        help="Location of previously saved model state.")
    parser.add_argument("-s", "--save-model", dest="save_data", default=f"./model_state_{ts}",
                        help="Save location for model state.")
    parser.add_argument("-r", "--reload-data", dest="reload", action="store_true",
                        help="Reload data from data directory.")
    parser.add_argument("-d", "--data-dir", dest="data_dir", default="E:/Datasets/YahooFinance/YahooSPData",
                        help="Directory of stock price data.")

    args = parser.parse_args()

    model_path = args.load_model
    save_path = args.save_data
    reload = args.reload
    data_dir = args.data_dir

    training = TrainModel(data_dir, model_path, save_path, reload)
    training.show_graph('AAPL')
    training.loss_graph()
