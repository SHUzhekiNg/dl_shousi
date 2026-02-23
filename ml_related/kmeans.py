import numpy as np

X = np.array([[1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0], [3.5, 5.0],
              [4.5, 5.0], [3.5, 4.5], [2.0, 2.5], [2.5, 3.0], [1.0, 2.0]], dtype=np.float32)


class KMeans:
    def __init__(self, k=2, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        indices = np.random.permutation(X.shape[0])[:self.k]
        self.centroids = X[indices].copy()

        for _ in range(self.max_iters):
            labels = self.predict(X)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        return labels

    def predict(self, X):
        # 计算每个点到每个质心的距离
        distances = np.sqrt(((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)


if __name__ == "__main__":
    np.random.seed(42)
    km = KMeans(k=2, max_iters=100)
    labels = km.fit(X)
    print("KMeans Labels:", labels)
    print("Centroids:", km.centroids)

    test_sample = np.array([[2.0, 3.0]], dtype=np.float32)
    km_test_pred = km.predict(test_sample)
    print(f"KMeans Prediction for {test_sample.tolist()}: cluster {km_test_pred[0]}")
