import torch
import numpy as np

# 构造小型数据集
X = torch.tensor([[1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0], [3.5, 5.0],
                  [4.5, 5.0], [3.5, 4.5], [2.0, 2.5], [2.5, 3.0], [1.0, 2.0]], dtype=torch.float32)
y = torch.tensor([0, 0, 1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.long)

# 1. 决策树实现
class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def entropy(self, labels):
        # 计算熵: -Σ(p * log2(p))
        _, counts = torch.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return -torch.sum(probs * torch.log2(probs))

    def information_gain(self, X, y, feature_idx, threshold):
        # 计算信息增益
        parent_entropy = self.entropy(y)
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        if torch.sum(left_mask) == 0 or torch.sum(right_mask) == 0:
            return 0.0

        left_entropy = self.entropy(y[left_mask])
        right_entropy = self.entropy(y[right_mask])
        n_left, n_right = torch.sum(left_mask), torch.sum(right_mask)
        child_entropy = (n_left / len(y)) * left_entropy + (n_right / len(y)) * right_entropy
        return parent_entropy - child_entropy

    def find_best_split(self, X, y):
        # 寻找最佳分割点（特征和阈值）
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]

        for feature_idx in range(n_features):
            thresholds = torch.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self.information_gain(X, y, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def fit(self, X, y, depth=0):
        # 递归构建决策树
        if depth >= self.max_depth or len(torch.unique(y)) == 1:
            return {'leaf': True, 'class': torch.mode(y)[0].item()}

        feature_idx, threshold, gain = self.find_best_split(X, y)
        if feature_idx is None or gain == 0:
            return {'leaf': True, 'class': torch.mode(y)[0].item()}

        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        return {
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': self.fit(X[left_mask], y[left_mask], depth + 1),
            'right': self.fit(X[right_mask], y[right_mask], depth + 1),
            'leaf': False
        }

    def predict_single(self, x, node):
        # 预测单个样本
        if node['leaf']:
            return node['class']
        if x[node['feature_idx']] <= node['threshold']:
            return self.predict_single(x, node['left'])
        return self.predict_single(x, node['right'])

    def predict(self, X):
        # 预测多个样本
        return torch.tensor([self.predict_single(x, self.tree) for x in X], dtype=torch.long)

# 示例用法
if __name__ == "__main__":
    # 训练决策树
    dt = DecisionTree(max_depth=3)
    dt.tree = dt.fit(X, y)
    dt_predictions = dt.predict(X)
    print("Decision Tree Predictions:", dt_predictions)
    print("Decision Tree Accuracy:", torch.mean((dt_predictions == y).float()).item())

    # 测试一个新样本
    test_sample = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
    dt_test_pred = dt.predict_single(test_sample[0], dt.tree)
    print(f"\nTest sample {test_sample.tolist()}:")
    print(f"Decision Tree Prediction: {dt_test_pred}")