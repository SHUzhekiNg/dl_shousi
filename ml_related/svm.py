import torch


class SVM:
    def __init__(self, n_features, lr=0.01, lambda_reg=0.01):
        self.weights = torch.zeros(n_features)
        self.bias = torch.zeros(1)
        self.lr = lr
        self.lambda_reg = lambda_reg

    def fit(self, X, y, epochs=1000):
        y_ = torch.where(y <= 0, -1.0, 1.0)
        for _ in range(epochs):
            scores = X @ self.weights + self.bias
            margin = y_ * scores
            mask = (margin < 1).float()
            dw = self.lambda_reg * self.weights - (1.0 / X.shape[0]) * (X.T @ (mask * y_))
            db = -(1.0 / X.shape[0]) * torch.sum(mask * y_)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        scores = X @ self.weights + self.bias
        return torch.sign(scores)

    def hinge_loss(self, X, y):
        y_ = torch.where(y <= 0, -1.0, 1.0)
        scores = X @ self.weights + self.bias
        return torch.mean(torch.clamp(1 - y_ * scores, min=0)) + 0.5 * self.lambda_reg * torch.sum(self.weights ** 2)


if __name__ == "__main__":
    torch.manual_seed(42)
    X = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [6.0, 5.0], [7.0, 8.0], [8.0, 7.0]], dtype=torch.float32)
    y = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=torch.float32)

    model = SVM(n_features=2, lr=0.01, lambda_reg=0.01)
    model.fit(X, y, epochs=1000)
    preds = model.predict(X)
    print("Predictions:", preds)
    print("Hinge loss:", model.hinge_loss(X, y).item())
