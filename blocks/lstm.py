import torch
import torch.nn as nn
import math


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=-1)
        i = torch.sigmoid(self.W_i(combined))
        f = torch.sigmoid(self.W_f(combined))
        o = torch.sigmoid(self.W_o(combined))
        c_hat = torch.tanh(self.W_c(combined))
        c = f * c_prev + i * c_hat
        h = o * torch.tanh(c)
        return h, c


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            self.cells.append(LSTMCell(input_size if layer == 0 else hidden_size, hidden_size))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            inp = x[:, t, :]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](inp, h[layer], c[layer])
                inp = h[layer]
            outputs.append(h[-1])

        return torch.stack(outputs, dim=1), (h[-1], c[-1])


if __name__ == "__main__":
    x = torch.randn(2, 10, 16)
    model = LSTM(input_size=16, hidden_size=32, num_layers=2)
    output, (h_n, c_n) = model(x)
    print("Output shape:", output.shape)
    print("h_n shape:", h_n.shape)
    print("c_n shape:", c_n.shape)
