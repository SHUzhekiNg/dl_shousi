import torch

X = torch.tensor([[1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0], [3.5, 5.0],
                  [4.5, 5.0], [3.5, 4.5], [2.0, 2.5], [2.5, 3.0], [1.0, 2.0]], dtype=torch.float32)
y = torch.tensor([0, 0, 1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.long)


class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.prior = {}

    def fit(self, X, y):
        self.classes = torch.unique(y)
        for c in self.classes:
            c = c.item()
            X_c = X[y == c]
            self.mean[c] = X_c.mean(dim=0)
            self.var[c] = X_c.var(dim=0)
            self.prior[c] = len(X_c) / len(X)

    def gaussian_prob(self, x, mean, var):
        return torch.exp(-0.5 * ((x - mean) ** 2) / (var + 1e-8)) / torch.sqrt(2 * torch.pi * (var + 1e-8))

    def predict_single(self, x):
        posteriors = []
        for c in self.classes:
            c = c.item()
            prior = torch.log(torch.tensor(self.prior[c]))
            likelihood = torch.sum(torch.log(self.gaussian_prob(x, self.mean[c], self.var[c])))
            posteriors.append(prior + likelihood)
        return self.classes[torch.argmax(torch.tensor(posteriors))].item()

    def predict(self, X):
        return torch.tensor([self.predict_single(x) for x in X], dtype=torch.long)


if __name__ == "__main__":
    nb = NaiveBayes()
    nb.fit(X, y)
    nb_predictions = nb.predict(X)
    print("Naive Bayes Predictions:", nb_predictions)
    print("Naive Bayes Accuracy:", torch.mean((nb_predictions == y).float()).item())

    test_sample = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
    nb_test_pred = nb.predict_single(test_sample[0])
    print(f"Naive Bayes Prediction: {nb_test_pred}")
