import torch
import torch.nn as nn

X = torch.tensor([[1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0], [3.5, 5.0],
                  [4.5, 5.0], [3.5, 4.5], [2.0, 2.5], [2.5, 3.0], [1.0, 2.0]], dtype=torch.float32)
y = torch.tensor([0, 0, 1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.long)

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        # 计算欧几里得距离
        return torch.sqrt(torch.sum((x1 - x2) ** 2))

    def predict_single(self, x):
        # 预测单个样本
        distances = torch.tensor([self.euclidean_distance(x, x_train) for x_train in self.X_train])
        _, indices = torch.topk(distances, self.k, largest=False)  # 找到 k 个最近邻
        k_nearest_labels = self.y_train[indices]
        return torch.mode(k_nearest_labels)[0].item()

    def predict(self, X):
        # 预测多个样本
        return torch.tensor([self.predict_single(x) for x in X], dtype=torch.long)



if __name__ == "__main__":
    # 训练 kNN
    knn = KNN(k=3)
    knn.fit(X, y)
    knn_predictions = knn.predict(X)
    print("kNN Predictions:", knn_predictions)
    print("kNN Accuracy:", torch.mean((knn_predictions == y).float()).item())


    test_sample = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
    knn_test_pred = knn.predict_single(test_sample[0])
    print(f"kNN Prediction: {knn_test_pred}")