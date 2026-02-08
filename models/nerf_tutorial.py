
"""
nerf_tutorial.py

A minimal, readable PyTorch implementation of NeRF (coarse + fine).
This is a template: to run training you must prepare a dataset of images, intrinsics,
and poses (camera-to-world matrices) and plug them into the DataLoader section.

Save this file and run:
    python nerf_tutorial.py

It will NOT start heavy training by default. See the bottom of the file for usage.
"""

import os
import math
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Utilities
# -------------------------
def cast_rays(origins, directions, depths):
    # origins: (R,3), directions: (R,3), depths: (N,) => returns (R,N,3)
    return origins[:, None, :] + depths[None, :, None] * directions[:, None, :]

def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return [to_device(a, device) for a in x]
    return x

# -------------------------
# Positional Encoding
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs: int, include_input: bool = True, log_sampling: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_sampling = log_sampling
        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., num_freqs - 1, num_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** (num_freqs - 1), num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., D)
        returns: (..., D * (include_input + 2 * num_freqs))
        """
        out = [x] if self.include_input else []
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        return torch.cat(out, dim=-1)


# -------------------------
# NeRF MLP
# -------------------------
class NeRF(nn.Module):
    """
    A simple NeRF MLP that outputs density (sigma) and feature which goes to color network.
    Uses skip connection at layer `skips`.
    """

    def __init__(self,
                 in_channels_pos: int,
                 in_channels_dir: int,
                 depth: int = 8,
                 width: int = 256,
                 skips: Tuple[int] = (4,)):
        super().__init__()
        self.depth = depth
        self.width = width
        self.skips = skips

        # positional -> hidden
        self.pts_linears = nn.ModuleList(
            [nn.Linear(in_channels_pos, width)] +
            [nn.Linear(width, width) if i not in self.skips else nn.Linear(width + in_channels_pos, width)
             for i in range(1, depth)]
        )

        # output sigma
        self.sigma_layer = nn.Linear(width, 1)

        # bottleneck feature
        self.feature_layer = nn.Linear(width, width)

        # directional branch
        self.dir_linears = nn.ModuleList([nn.Linear(in_channels_dir + width, width // 2)])
        self.color_layer = nn.Sequential(
            nn.Linear(width // 2, 3),
            nn.Sigmoid()
        )

        # initialize
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        """
        x: (..., in_channels_pos) encoded positions
        d: (..., in_channels_dir) encoded directions (must broadcast to x)
        returns: sigma: (...,1), rgb: (...,3)
        """
        h = x
        for i, l in enumerate(self.pts_linears):
            h = l(h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], dim=-1)
        sigma = self.sigma_layer(h)  # (...,1)
        feat = self.feature_layer(h)  # (..., width)

        # directional branch
        h_dir = torch.cat([feat, d], dim=-1)  # (..., width + dir)
        for l in self.dir_linears:
            h_dir = l(h_dir)
            h_dir = F.relu(h_dir)
        rgb = self.color_layer(h_dir)
        return sigma, rgb


# -------------------------
# Volume rendering & sampling routines
# -------------------------
def compute_alphas(sigmas, deltas):
    # sigmas: (R, N, 1) or (...), deltas: (R, N) or (...)
    # returns alphas same shape as sigmas
    return 1.0 - torch.exp(-sigmas.squeeze(-1) * deltas)

def volume_rendering(sigmas, rgbs, z_vals, rays_dir, white_bkgd=True):
    """
    sigmas: (R, N, 1)
    rgbs:   (R, N, 3)
    z_vals: (R, N) sample depths along a ray
    rays_dir: (R,3)
    returns: comp_rgb: (R,3), depth: (R), weights: (R,N)
    """
    device = sigmas.device
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (R, N-1)
    # last delta: infinity approximation
    delta_last = 1e10 * torch.ones_like(deltas[:, :1])
    deltas = torch.cat([deltas, delta_last], dim=-1)  # (R,N)

    alphas = compute_alphas(sigmas, deltas)  # (R,N)
    # T_i = cumprod(1-alpha_j) for j < i ; implement via exclusive cumprod
    transmittance = torch.cumprod(torch.cat([torch.ones((alphas.shape[0], 1), device=device), 1. - alphas + 1e-10], -1), -1)[:, :-1]
    weights = transmittance * alphas  # (R,N)

    comp_rgb = torch.sum(weights[:, :, None] * rgbs, dim=1)  # (R,3)
    depth = torch.sum(weights * z_vals, dim=1)  # (R,)
    if white_bkgd:
        acc_map = torch.sum(weights, dim=-1, keepdim=True)  # (R,1)
        comp_rgb = comp_rgb + (1.0 - acc_map) * 1.0  # assume white background
    return comp_rgb, depth, weights


def stratified_sampling(rng, near, far, n_samples, deterministic=False):
    """
    Stratified sampling along the ray between near and far.
    rng: torch.Generator or None
    returns: z_vals (n_rays, n_samples) in ascending order
    """
    t_vals = torch.linspace(0.0, 1.0, steps=n_samples, device=near.device)
    mids = 0.5 * (t_vals[:-1] + t_vals[1:])
    # convert to [near, far]
    z_vals = near[:, None] * (1. - t_vals[None, :]) + far[:, None] * (t_vals[None, :])
    if not deterministic:
        # perturb within bins
        mids = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, :1], mids], dim=-1)
        t_rand = torch.rand(z_vals.shape, device=z_vals.device, generator=rng)
        z_vals = lower + (upper - lower) * t_rand
    return z_vals


def sample_pdf(bins, weights, n_samples, deterministic=False, rng=None):
    """
    Hierarchical sampling: sample n_samples from PDF defined by bins & weights.
    bins: (R, N_bins) - bin edges (z midpoints)
    weights: (R, N_bins) - weights per bin (typically from coarse network)
    returns: samples: (R, n_samples)
    """
    # Add small value to weights to avoid nans
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (R, N_bins+1)

    if deterministic:
        u = torch.linspace(0., 1., steps=n_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=bins.device, generator=rng)

    # invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], -1)  # (R, n_samples, 2)

    cdf_g = torch.gather(cdf.unsqueeze(1).expand(-1, n_samples, -1), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(-1, n_samples, -1), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples


# -------------------------
# A minimal Dataset skeleton (user should replace this)
# -------------------------
class SimpleRayDataset(Dataset):
    """
    Placeholder dataset. User should replace this by a loader that returns:
      image: (H,W,3) tensor in [0,1]
      pose: (4,4) camera-to-world matrix, extrinsics.
      intrinsics: dict with 'fx','fy','cx','cy', and optionally 'height','width'
    For training, we usually sample random batches of rays per image.
    """
    def __init__(self, images, poses, intrinsics):
        super().__init__()
        self.images = images
        self.poses = poses
        self.intrinsics = intrinsics
        assert len(images) == len(poses)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "pose": self.poses[idx],
            "intrinsics": self.intrinsics
        }


# -------------------------
# Ray helpers: from intrinsics+pose to ray origins & dirs
# -------------------------
def get_rays(H, W, fx, fy, cx, cy, c2w):
    i, j = torch.meshgrid(torch.arange(W, device=c2w.device),
                          torch.arange(H, device=c2w.device), indexing='xy')
    # pixel coordinates to camera ray directions
    dirs = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], dim=-1)  # (W, H, 3)
    dirs = dirs.view(-1, 3)  # (H*W, 3) but note i,j swapped earlier
    # transform to world coordinates
    rays_d = (dirs @ c2w[:3, :3].T).view(-1, 3)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


# -------------------------
# Training helpers
# -------------------------
def render_rays(models, embeddings, rays_o, rays_d, near, far, N_samples, N_importance=0, rng=None, white_bkgd=True):
    """
    Render a batch of rays using coarse (and optionally fine) model.
    models: dict with 'coarse' and optionally 'fine' NeRF models
    embeddings: dict with 'pos' and 'dir' encoders
    rays_o, rays_d: (R,3)
    near, far: (R,) or scalars
    returns: results dict with 'rgb_coarse','depth_coarse', and if fine: 'rgb_fine','depth_fine'
    """
    device = rays_o.device
    R = rays_o.shape[0]

    # 1. stratified coarse samples
    z_vals_coarse = stratified_sampling(rng, near * torch.ones(R, device=device), far * torch.ones(R, device=device), N_samples)
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_coarse[:, :, None]  # (R, N, 3)

    # encode and forward coarse model
    pts_enc = embeddings['pos'](pts.reshape(-1, 3)).reshape(R, N_samples, -1)
    dirs_enc = embeddings['dir'](rays_d).unsqueeze(1).expand(R, N_samples, -1).reshape(-1, embeddings['dir_out'])
    sigma_coarse, rgb_coarse = models['coarse'](pts_enc.reshape(-1, pts_enc.shape[-1]), dirs_enc)
    sigma_coarse = sigma_coarse.view(R, N_samples, 1)
    rgb_coarse = rgb_coarse.view(R, N_samples, 3)

    comp_rgb_c, depth_c, weights_c = volume_rendering(sigma_coarse, rgb_coarse, z_vals_coarse, rays_d, white_bkgd=white_bkgd)

    results = {
        'rgb_coarse': comp_rgb_c,
        'depth_coarse': depth_c,
        'weights_coarse': weights_c,
        'z_vals_coarse': z_vals_coarse
    }

    # 2. hierarchical sampling (fine)
    if N_importance > 0:
        # use midpoints of z_vals_coarse as bins
        mids = 0.5 * (z_vals_coarse[:, :-1] + z_vals_coarse[:, 1:])
        new_z_samples = sample_pdf(mids, weights_c[:, 1:-1], N_importance, deterministic=False, rng=rng)
        z_vals_fine, _ = torch.sort(torch.cat([z_vals_coarse, new_z_samples], dim=-1), dim=-1)
        pts_fine = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_fine[:, :, None]

        pts_enc_f = embeddings['pos'](pts_fine.reshape(-1, 3)).reshape(R, z_vals_fine.shape[1], -1)
        dirs_enc_f = embeddings['dir'](rays_d).unsqueeze(1).expand(R, z_vals_fine.shape[1], -1).reshape(-1, embeddings['dir_out'])
        sigma_fine, rgb_fine = models['fine'](pts_enc_f.reshape(-1, pts_enc_f.shape[-1]), dirs_enc_f)
        sigma_fine = sigma_fine.view(R, z_vals_fine.shape[1], 1)
        rgb_fine = rgb_fine.view(R, z_vals_fine.shape[1], 3)

        comp_rgb_f, depth_f, weights_f = volume_rendering(sigma_fine, rgb_fine, z_vals_fine, rays_d, white_bkgd=white_bkgd)
        results.update({
            'rgb_fine': comp_rgb_f,
            'depth_fine': depth_f,
            'weights_fine': weights_f,
            'z_vals_fine': z_vals_fine
        })
    return results


# -------------------------
# Putting it together: a simple training loop skeleton
# -------------------------
def make_nerf_models(pos_encode_dim=10, dir_encode_dim=4, device='cuda'):
    pos_embed = PositionalEncoding(pos_encode_dim)
    dir_embed = PositionalEncoding(dir_encode_dim)
    # compute encoded sizes
    dummy_pos = torch.zeros(1, 3)
    dummy_dir = torch.zeros(1, 3)
    pos_dim = pos_embed(dummy_pos).shape[-1] # 3 * (include_input + 2 * pos_encode_dim)
    dir_dim = dir_embed(dummy_dir).shape[-1] # 3 * (include_input + 2 * dir_encode_dim)

    coarse = NeRF(in_channels_pos=pos_dim, in_channels_dir=dir_dim).to(device)
    fine = NeRF(in_channels_pos=pos_dim, in_channels_dir=dir_dim).to(device)
    embeddings = {
        'pos': pos_embed,
        'dir': dir_embed,
        'dir_out': dir_dim
    }
    models = {
        'coarse': coarse,
        'fine': fine
    }
    return models, embeddings


def train_epoch(models, embeddings, optimizer, dataset, device, batch_rays=1024,
                N_samples=64, N_importance=128, near=2.0, far=6.0):
    models['coarse'].train()
    models['fine'].train()

    # Simple per-image ray batching (user can replace with more advanced sampler)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    mse_loss = nn.MSELoss()
    for data in dataloader:
        image = data['image'][0].to(device)  # assume (H,W,3)
        pose = data['pose'][0].to(device)
        intr = data['intrinsics']

        H, W = image.shape[:2]
        fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']

        # compute all rays for the image (this can be large; for memory, sample random pixels instead)
        rays_o, rays_d = get_rays(H, W, fx, fy, cx, cy, pose)
        pixels = image.view(-1, 3)

        # randomly sample rays for this iteration
        idxs = torch.randperm(rays_o.shape[0], device=device)[:batch_rays]
        rays_o_b = rays_o[idxs]
        rays_d_b = rays_d[idxs]
        pixels_b = pixels[idxs]

        rng = torch.Generator(device=device)
        results = render_rays(models, embeddings, rays_o_b, rays_d_b, near, far, N_samples, N_importance, rng=rng)
        rgb_pred = results['rgb_fine'] if 'rgb_fine' in results else results['rgb_coarse']

        loss = mse_loss(rgb_pred, pixels_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return


# -------------------------
# Example usage (main)
# -------------------------
if __name__ == "__main__":
    print("This is a NeRF PyTorch template. Edit the dataset loader and run training.")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    # === Placeholder synthetic dataset ===
    # The user should replace this with a real dataset loader. Here we create a single
    # dummy image and identity pose so running the script won't crash but won't produce meaningful results.
    H, W = 64, 64
    dummy_img = torch.ones((H, W, 3), dtype=torch.float32) * 0.5  # gray image
    dummy_pose = torch.eye(4, dtype=torch.float32)
    intr = {'fx': 50.0, 'fy': 50.0, 'cx': W / 2.0, 'cy': H / 2.0, 'height': H, 'width': W}

    dataset = SimpleRayDataset([dummy_img], [dummy_pose], intr)
    models, embeddings = make_nerf_models(device=device)
    optimizer = torch.optim.Adam(list(models['coarse'].parameters()) + list(models['fine'].parameters()), lr=5e-4)

    # Run one training epoch (fast because dataset tiny)
    train_epoch(models, embeddings, optimizer, dataset, device, batch_rays=512, N_samples=32, N_importance=32, near=0.1, far=2.0)

    print("Done one training epoch on dummy data. Replace dataset with real images+poses to train properly.")
