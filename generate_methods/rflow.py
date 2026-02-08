import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp

class RFlow:
    def sample(self, model, shape, steps, cond):
        x = torch.randn_like(shape)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((shape[0],), i*dt, device=x.device)
            with torch.no_grad():
                x1_pred = model(x, t, cond)
            v = (x1_pred - x) / (1.0 - i*dt + 1e-6)
            x = x + v * dt
        return x
    
    def train_step(self, model, x0, x1, cond):
        t_raw = torch.rand(x0.shape[0], 1, device=x0.device)
        t = t_raw.view(x0.shape[0], 1, 1, 1)
        x_t = (1 - t) * x0 + t * x1
        x1_pred = model(x_t, t, cond)
        loss = F.mse_loss(x1_pred, x1)
        return loss
