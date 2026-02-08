import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, latent_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            # (B, 128, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            # (B, 128, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(hidden_channels, latent_dim, 3, stride=1, padding=1),
            # (B, 64, H/4, W/4)
        )
    def forward(self, x):
        z_e = self.conv(x)  # shape: (B, latent_dim, H/4, W/4)
        return z_e

class Decoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_channels=128, out_channels=3):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_channels, 4, stride=2, padding=1),
            # (B, 128, H/2, W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            # (B, 128, H, W)
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, 3, stride=1, padding=1),
            # (B, 3, H, W)
        )
    def forward(self, z_q):
        x_recon = self.deconv(z_q)  # shape: (B, 3, H, W)
        return x_recon

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)  # (K, D)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        self.beta = commitment_cost

    def forward(self, z_e):
        # z_e: (B, D, H/4, W/4)
        B, D, H, W = z_e.shape
        z_flattened = z_e.permute(0, 2, 3, 1).contiguous()  # (B, H/4, W/4, D)
        z_flattened = z_flattened.view(-1, D)  # (B*H*W/16, D)

        # 计算到 codebook 的距离
        # self.embedding.weight: (K, D)
        # dist: (B*H*W/16, K)
        dist = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)  # (B*H*W, 1)
            + torch.sum(self.embedding.weight ** 2, dim=1)    # (K,)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())  # (B*H*W, K)
        )

        # 最近邻索引
        encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)  # (B*H*W/16, 1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z_e.device)  # (B*H*W/16, K)
        encodings.scatter_(1, encoding_indices, 1)  # one-hot (B*H*W/16, K)

        # 量化向量
        z_q = torch.matmul(encodings, self.embedding.weight)  # (B*H*W/16, D)
        z_q = z_q.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # (B, D, H/4, W/4)

        # 损失项
        commitment_loss = self.beta * F.mse_loss(z_e.detach(), z_q)  # scalar
        codebook_loss = F.mse_loss(z_e, z_q.detach())               # scalar

        # Straight-through trick
        z_q = z_e + (z_q - z_e).detach()  # (B, D, H/4, W/4)

        return z_q, codebook_loss + commitment_loss

class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.quantizer = VectorQuantizer()
        self.decoder = Decoder()

    def forward(self, x):
        z_e = self.encoder(x)                 # (B, D, H/4, W/4)
        z_q, vq_loss = self.quantizer(z_e)   # (B, D, H/4, W/4)
        x_recon = self.decoder(z_q)          # (B, 3, H, W)
        recon_loss = F.mse_loss(x_recon, x)  # scalar
        total_loss = recon_loss + vq_loss
        return x_recon, total_loss, recon_loss, vq_loss


# ======= 训练示例 =======
model = VQVAE().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

x_batch = torch.randn(16, 3, 64, 64).cuda()  # (B, C, H, W)

x_recon, total_loss, recon_loss, vq_loss = model(x_batch)
optimizer.zero_grad()
total_loss.backward()
optimizer.step()

print(f"recon_loss={recon_loss.item():.4f}, vq_loss={vq_loss.item():.4f}")
