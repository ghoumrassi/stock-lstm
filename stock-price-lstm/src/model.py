import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialise hidden states
        lstm_out, _ = self.lstm(x)
        dropout_out = F.dropout(lstm_out, p=0.2)
        fc_out = self.fc(dropout_out)
        return fc_out

if __name__ == "__main__":
    input_dim = 10
    hidden_dim = 10
    output_dim = 10
    num_layers = 1

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                 num_layers=num_layers)

    loss_fn = nn.MSELoss(size_average=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    print('Model: ', model)
    print("# of Parameter Matrices: ", len(list(model.parameters())))
    for param in list(model.parameters()):
        print(param.size())