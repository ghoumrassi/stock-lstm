from src import StockDataset, LSTM

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

device = "cuda:0" if torch.cuda.is_available() else "cpu"

num_files = 100
num_epochs = 2
hidden_dim = 50
learning_rate = 0.01
batch_size = 20

train_data = StockDataset(r"E:\Datasets\YahooFinance\YahooSPData", interpolation='zero', num_files=num_files,
                       start_date='2005-01-01', end_date='2017-01-01')
val_data = StockDataset(r"E:\Datasets\YahooFinance\YahooSPData", interpolation='zero', num_files=num_files,
                       start_date='2017-01-01', end_date='2019-01-01')

num_features = train_data.num_features

model = LSTM(input_dim=num_features, hidden_dim=hidden_dim, output_dim=num_features)
model = model.double()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.1)
criterion = nn.MSELoss()

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)

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
        predictions = np.vstack(predictions_list)
    print('Training Loss: %.4g' % (training_loss))
    print('Validation Loss: %.4g' % (val_loss))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
# Testing prediction accuracy on series
    def show_graph(actuals, predictions, series_name):
        cols = val_data.names
        predictions_frame = pd.DataFrame(predictions, columns=cols)
        fig, ax = plt.subplots(figsize=(8,8))
        predictions_frame[series_name].plot(ax=ax)
        actuals[series_name].plot(ax=ax)
        ax.legend(["Prediction", "Actual"])
        plt.show()

show_graph(val_data.df_final, predictions, 'AAPL')