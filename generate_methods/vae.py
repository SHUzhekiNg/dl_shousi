import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, latent_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 2 * latent_dim, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_channels=128, out_channels=3):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, 3, stride=1, padding=1),
        )

    def forward(self, z):
        return self.deconv(z)

class VAE(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_channels, in_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

if __name__ == "__main__":
    # Hyperparameters
    B, C, H, W = 4, 3, 32, 32
    latent_dim = 64
    
    # Model, Optimizer
    encoder = Encoder(in_channels=C, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, out_channels=C)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    
    # Dummy Dataloader
    dataloader = [torch.randn(B, C, H, W) for _ in range(10)]

    for x_batch in dataloader:  # x_batch: [B, C, H, W]
        h = encoder(x_batch)       # [B, 2*latent_dim, H', W']
        
        # 拆分均值和对数方差
        mu, logvar = h.chunk(2, dim=1)  # [B, latent_dim, H', W'] * 2

        # 重参数化采样
        std = torch.exp(0.5 * logvar)   # [B, latent_dim, H', W']
        eps = torch.randn_like(std)     # [B, latent_dim, H', W'], epsilon ~ N(0,1)
        z = mu + std * eps              # [B, latent_dim, H', W']

        x_recon = decoder(z)            # [B, C, H, W]
        
        # 设计你喜欢的recon loss～
        recon_loss = torch.mean((x_recon - x_batch)**2)
        # 把下面的负号集成在这里取反了。
        kl_loss = torch.sum(-0.5 * (1 + logvar - mu**2 - torch.exp(logvar))) / x_batch.size(0) 

        loss = recon_loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}")

