import numpy as np

X = np.array([[1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0], [3.5, 5.0],
              [4.5, 5.0], [3.5, 4.5], [2.0, 2.5], [2.5, 3.0], [1.0, 2.0]], dtype=np.float32)
y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0], dtype=np.int64)


class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.prior = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0)
            self.prior[c] = len(X_c) / len(X)

    def gaussian_prob(self, x, mean, var):
        return np.exp(-0.5 * ((x - mean) ** 2) / (var + 1e-8)) / np.sqrt(2 * np.pi * (var + 1e-8))

    def predict_single(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.prior[c])
            likelihood = np.sum(np.log(self.gaussian_prob(x, self.mean[c], self.var[c])))
            posteriors.append(prior + likelihood)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X], dtype=np.int64)


if __name__ == "__main__":
    nb = NaiveBayes()
    nb.fit(X, y)
    nb_predictions = nb.predict(X)
    print("Naive Bayes Predictions:", nb_predictions)
    print("Naive Bayes Accuracy:", np.mean(nb_predictions == y))

    test_sample = np.array([[2.0, 3.0]], dtype=np.float32)
    nb_test_pred = nb.predict_single(test_sample[0])
    print(f"Naive Bayes Prediction: {nb_test_pred}")
