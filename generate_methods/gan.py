import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_gan(real_data, latent_dim=16, hidden_dim=64, lr=2e-4, epochs=2000):
    data_dim = real_data.shape[1]
    G = Generator(latent_dim, hidden_dim, data_dim)
    D = Discriminator(data_dim, hidden_dim)
    optim_G = torch.optim.Adam(G.parameters(), lr=lr)
    optim_D = torch.optim.Adam(D.parameters(), lr=lr)
    bce = nn.BCELoss()

    for epoch in range(epochs):
        batch_size = real_data.shape[0]
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        z = torch.randn(batch_size, latent_dim)
        fake_data = G(z)
        d_real = D(real_data)
        d_fake = D(fake_data.detach())
        loss_D = bce(d_real, real_labels) + bce(d_fake, fake_labels)
        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()

        z = torch.randn(batch_size, latent_dim)
        fake_data = G(z)
        d_fake = D(fake_data)
        loss_G = bce(d_fake, real_labels)
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}, D_loss: {loss_D.item():.4f}, G_loss: {loss_G.item():.4f}")

    return G, D


if __name__ == "__main__":
    torch.manual_seed(42)
    real_data = torch.randn(256, 2) * 0.5 + torch.tensor([3.0, 3.0])
    G, D = train_gan(real_data, latent_dim=16, hidden_dim=64, epochs=2000)
    z = torch.randn(10, 16)
    generated = G(z)
    print("Generated samples:\n", generated.detach())
