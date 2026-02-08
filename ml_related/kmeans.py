import torch

X = torch.tensor([[1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0], [3.5, 5.0],
                  [4.5, 5.0], [3.5, 4.5], [2.0, 2.5], [2.5, 3.0], [1.0, 2.0]], dtype=torch.float32)


class KMeans:
    def __init__(self, k=2, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        indices = torch.randperm(X.shape[0])[:self.k]
        self.centroids = X[indices].clone()

        for _ in range(self.max_iters):
            labels = self.predict(X)
            new_centroids = torch.stack([X[labels == i].mean(dim=0) for i in range(self.k)])
            if torch.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        return labels

    def predict(self, X):
        distances = torch.cdist(X, self.centroids)
        return torch.argmin(distances, dim=1)


if __name__ == "__main__":
    torch.manual_seed(42)
    km = KMeans(k=2, max_iters=100)
    labels = km.fit(X)
    print("KMeans Labels:", labels)
    print("Centroids:", km.centroids)

    test_sample = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
    km_test_pred = km.predict(test_sample)
    print(f"KMeans Prediction for {test_sample.tolist()}: cluster {km_test_pred.item()}")
