import torch

X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0],
                  [6.0], [7.0], [8.0], [9.0], [10.0]], dtype=torch.float32)
y = torch.tensor([2.1, 4.0, 5.8, 8.2, 10.1, 11.9, 14.1, 15.8, 18.2, 19.9], dtype=torch.float32)


class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        ones = torch.ones(X.shape[0], 1)
        X_b = torch.cat([ones, X], dim=1)
        theta = torch.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        return X @ self.weights + self.bias

    def mse(self, y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2)


if __name__ == "__main__":
    lr = LinearRegression()
    lr.fit(X, y)
    lr_predictions = lr.predict(X)
    print("Linear Regression Predictions:", lr_predictions)
    print("MSE:", lr.mse(lr_predictions, y).item())

    test_sample = torch.tensor([[11.0]], dtype=torch.float32)
    lr_test_pred = lr.predict(test_sample)
    print(f"Linear Regression Prediction for {test_sample.tolist()}: {lr_test_pred.item():.2f}")
