import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialise hidden states
        h0 = torch.zero_(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialise cell states
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(h0.detach(), c0.detach())

        out = self.fc(out[:, -1, :])

        return out

if __name__ == "__main__":
    input_dim = 10
    hidden_dim = 10
    output_dim = 10
    num_layers = 10

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                 num_layers=num_layers)

    loss_fn = nn.MSELoss(size_average=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    print('Model: ', model)
    print("# of Parameters: ", len(list(model.parameters())))
    for param in list(model.parameters()):
        print(param.size())