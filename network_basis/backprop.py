import torch
import math


class MLP:
    def __init__(self, dims):
        self.weights = []
        self.biases = []
        for i in range(len(dims) - 1):
            k = math.sqrt(1.0 / dims[i])
            w = torch.empty(dims[i], dims[i + 1]).uniform_(-k, k)
            b = torch.zeros(dims[i + 1])
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x):
        return torch.clamp(x, min=0)

    def relu_grad(self, x):
        return (x > 0).float()

    def softmax(self, x):
        e = torch.exp(x - x.max(dim=-1, keepdim=True).values)
        return e / e.sum(dim=-1, keepdim=True)

    def forward(self, x):
        self.activations = [x]
        self.pre_activations = []
        for i in range(len(self.weights)):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.pre_activations.append(z)
            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = self.softmax(z)
            self.activations.append(a)
        return self.activations[-1]

    def cross_entropy_loss(self, pred, target):
        eps = 1e-8
        log_pred = torch.log(pred + eps)
        return -torch.mean(torch.sum(target * log_pred, dim=-1))

    def backward(self, target, lr=0.01):
        batch_size = target.shape[0]
        delta = self.activations[-1] - target

        for i in reversed(range(len(self.weights))):
            dw = (self.activations[i].T @ delta) / batch_size
            db = delta.mean(dim=0)
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.relu_grad(self.pre_activations[i - 1])


if __name__ == "__main__":
    torch.manual_seed(42)
    X = torch.randn(200, 4)
    labels = (X[:, 0] + X[:, 1] > 0).long()
    y = torch.zeros(200, 3)
    y[torch.arange(200), labels] = 1.0

    mlp = MLP([4, 16, 3])
    for epoch in range(500):
        pred = mlp.forward(X)
        loss = mlp.cross_entropy_loss(pred, y)
        mlp.backward(y, lr=0.05)
        if (epoch + 1) % 100 == 0:
            acc = (pred.argmax(dim=-1) == labels).float().mean()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")
