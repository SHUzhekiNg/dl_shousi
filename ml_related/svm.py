import numpy as np


class SVM:
    def __init__(self, n_features, lr=0.01, lambda_reg=0.01):
        self.weights = np.zeros(n_features)
        self.bias = np.zeros(1)
        self.lr = lr
        self.lambda_reg = lambda_reg

    def fit(self, X, y, epochs=1000):
        y_ = np.where(y <= 0, -1.0, 1.0)
        for _ in range(epochs):
            scores = X @ self.weights + self.bias
            margin = y_ * scores
            mask = (margin < 1).astype(np.float32)
            dw = self.lambda_reg * self.weights - (1.0 / X.shape[0]) * (X.T @ (mask * y_))
            db = -(1.0 / X.shape[0]) * np.sum(mask * y_)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        scores = X @ self.weights + self.bias
        return np.sign(scores)

    def hinge_loss(self, X, y):
        y_ = np.where(y <= 0, -1.0, 1.0)
        scores = X @ self.weights + self.bias
        return np.mean(np.maximum(0, 1 - y_ * scores)) + 0.5 * self.lambda_reg * np.sum(self.weights ** 2)


if __name__ == "__main__":
    np.random.seed(42)
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [6.0, 5.0], [7.0, 8.0], [8.0, 7.0]], dtype=np.float32)
    y = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    model = SVM(n_features=2, lr=0.01, lambda_reg=0.01)
    model.fit(X, y, epochs=1000)
    preds = model.predict(X)
    print("Predictions:", preds)
    print("Hinge loss:", model.hinge_loss(X, y))
