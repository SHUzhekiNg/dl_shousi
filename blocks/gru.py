import torch
import torch.nn as nn


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_n = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=-1)
        z = torch.sigmoid(self.W_z(combined))
        r = torch.sigmoid(self.W_r(combined))
        combined_r = torch.cat([x, r * h_prev], dim=-1)
        n = torch.tanh(self.W_n(combined_r))
        h = (1 - z) * h_prev + z * n
        return h


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            self.cells.append(GRUCell(input_size if layer == 0 else hidden_size, hidden_size))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            inp = x[:, t, :]
            for layer in range(self.num_layers):
                h[layer] = self.cells[layer](inp, h[layer])
                inp = h[layer]
            outputs.append(h[-1])

        return torch.stack(outputs, dim=1), h[-1]


if __name__ == "__main__":
    x = torch.randn(2, 10, 16)
    model = GRU(input_size=16, hidden_size=32, num_layers=2)
    output, h_n = model(x)
    print("Output shape:", output.shape)
    print("h_n shape:", h_n.shape)
