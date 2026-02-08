import torch

X = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.5], [4.0, 5.5], [5.0, 6.0],
                  [5.5, 7.0], [6.0, 8.0], [7.0, 9.0], [8.0, 10.0], [9.0, 11.0]], dtype=torch.float32)
y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.float32)


class LogisticRegression:
    def __init__(self, n_features, lr=0.1):
        self.weights = torch.zeros(n_features)
        self.bias = torch.zeros(1)
        self.lr = lr

    def sigmoid(self, z):
        return 1.0 / (1.0 + torch.exp(-z))

    def fit(self, X, y, epochs=1000):
        n_samples = X.shape[0]
        for _ in range(epochs):
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)
            dw = (1.0 / n_samples) * (X.T @ (y_pred - y))
            db = (1.0 / n_samples) * torch.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        return self.sigmoid(X @ self.weights + self.bias)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).float()

    def bce_loss(self, y_pred, y_true):
        eps = 1e-7
        return -torch.mean(y_true * torch.log(y_pred + eps) + (1 - y_true) * torch.log(1 - y_pred + eps))


if __name__ == "__main__":
    model = LogisticRegression(n_features=2, lr=0.1)
    model.fit(X, y, epochs=1000)
    proba = model.predict_proba(X)
    preds = model.predict(X)
    print("Probabilities:", proba)
    print("Predictions:", preds)
    print("BCE Loss:", model.bce_loss(proba, y).item())

    test_sample = torch.tensor([[6.5, 8.5]], dtype=torch.float32)
    print(f"Prediction for {test_sample.tolist()}: {model.predict(test_sample).item()}")
