import torch


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = X.mean(dim=0)
        X_centered = X - self.mean
        cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        idx = torch.argsort(eigenvalues, descending=True)
        self.components = eigenvectors[:, idx[:self.n_components]]
        self.explained_variance = eigenvalues[idx[:self.n_components]]

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components

    def inverse_transform(self, Z):
        return Z @ self.components.T + self.mean


if __name__ == "__main__":
    torch.manual_seed(42)
    X = torch.randn(100, 5)
    pca = PCA(n_components=2)
    pca.fit(X)
    Z = pca.transform(X)
    X_reconstructed = pca.inverse_transform(Z)
    print("Original shape:", X.shape)
    print("Transformed shape:", Z.shape)
    print("Explained variance:", pca.explained_variance)
    print("Reconstruction error:", torch.mean((X - X_reconstructed) ** 2).item())
