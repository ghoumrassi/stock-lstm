from src import StockDataset, LSTM

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TrainModel:
    def __init__(self, data_dir="E:/Datasets/YahooFinance/YahooSPData",
                 model_path=None, save_path='model', reload=False, num_epochs=0):

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        num_epochs = num_epochs
        hidden_dim = 100
        learning_rate = 0.1
        momentum = 0.1
        num_layers = 2
        prediction_lag = 7
        window_size = 365
        self.batch_size = 20

        train_start = '2002-01-01'
        train_end = '2019-01-01'

        val_start = '2018-01-01'
        val_end = '2021-01-01'

        test_start = '2019-01-01'
        test_end = '2021-01-01'

        self.train_data = StockDataset(device=self.device, window_size=window_size, start_date=train_start,
                                       end_date=train_end, prediction_lag=prediction_lag, reload_data=reload,
                                       data_dir=data_dir)
        self.val_data = StockDataset(device=self.device, window_size=window_size, start_date=val_start,
                                     end_date=val_end, prediction_lag=prediction_lag)
        self.test_data = StockDataset(device=self.device, window_size=window_size, start_date=test_start,
                                      end_date=test_end, prediction_lag=prediction_lag)

        num_features = self.train_data.num_features

        self.model = LSTM(input_dim=num_features, hidden_dim=hidden_dim, output_dim=num_features, num_layers=num_layers)
        self.model = self.model.double()

        # Load model if path specified
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))

        self.model.to(self.device)
        print(torch.cuda.is_available())
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = nn.MSELoss()

        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False, drop_last=True)
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, drop_last=True)

        self.loss_dict = {'train': [], 'val': []}
        last_lr = 0
        for epoch in range(num_epochs):
            if epoch > 2:
                last_val_loss = self.loss_dict['val'][-1]
                prev_val_loss = self.loss_dict['val'][-2]
                if last_val_loss / prev_val_loss > 0.995 and epoch - last_lr > 10:
                    print("Learning rate reduced.")
                    learning_rate /= 2
                    last_lr = epoch
            print("Epoch: ", epoch + 1)
            training_loss = 0
            val_loss = 0
            for X, y in train_loader:
                self.model.zero_grad()
                out = self.model(X)
                y_pred = out[:, -1, :].view(self.batch_size, -1)
                loss = criterion(y_pred, y)
                training_loss += loss.item()
                loss.backward()
                optimizer.step()

            with torch.set_grad_enabled(False):
                predictions_list = []
                for X, y in val_loader:
                    out = self.model(X)
                    y_pred = out[:, -1, :].view(self.batch_size, -1)
                    loss = criterion(y_pred, y)
                    val_loss += loss.item()
                    if self.device == "cpu":
                        predictions_list.append(y_pred.data.numpy())
                    else:
                        predictions_list.append(y_pred.data.cpu().numpy())
                self.predictions = np.vstack(predictions_list)
            print('Training Loss: %.4g' % training_loss)
            print('Validation Loss: %.4g' % val_loss)
            self.loss_dict['train'].append(training_loss)
            self.loss_dict['val'].append(val_loss)

            # Save model
            torch.save(self.model.state_dict(), save_path)

        if num_epochs != 0:
            self.predictions.index = self.val_data.target_dates(end=len(self.predictions))
            self.predictions.columns = self.val_data.names

    def save2csv(self):
        # Save predictions
        self.val_data.data.to_csv('val_data.csv')
        pd.DataFrame(
            self.predictions,
            columns=self.val_data.names
        ).to_csv('predictions.csv')

    def predict(self):
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, drop_last=True)
        with torch.set_grad_enabled(False):
            predictions_list = []
            for X, y in val_loader:
                batch_size = y.shape.item()
                out = self.model(X)
                y_pred = out[:, -1, :].view(batch_size, -1)
                if self.device == "cpu":
                    predictions_list.append(y_pred.data.numpy())
                else:
                    predictions_list.append(y_pred.data.cpu().numpy())
            self.predictions = np.vstack(predictions_list)

        self.predictions = pd.DataFrame(self.predictions)
        self.predictions.index = self.val_data.target_dates(end=len(self.predictions))
        self.predictions.columns = self.val_data.names

    # def show_graph(self, series_name):
    #     actual = self.val_data.data
    #     predictions = self.predictions
    #
    #     cols = self.val_data.names
    #
    #     predictions_frame = pd.DataFrame(predictions, columns=cols)
    #     predictions_frame.index = self.val_data.target_dates()
    #
    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     predictions_frame[series_name].plot(ax=ax)
    #     actual[series_name].plot(ax=ax)
    #     ax.legend(["Prediction", "Actual"])
    #     plt.show()

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
    parser.add_argument("-e", "--epochs", dest="epochs", default=100, type=int,
                        help="Number of epochs to train for.")

    args = parser.parse_args()

    model_path = args.load_model
    save_path = args.save_data
    reload = args.reload
    data_dir = args.data_dir
    epochs = args.epochs

    training = TrainModel(data_dir=data_dir, model_path=model_path, save_path=save_path,
                          reload=reload, num_epochs=epochs)
    # training.show_graph('AAPL')
    training.loss_graph()
