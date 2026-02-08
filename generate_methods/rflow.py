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


class JiTFlowScheduler:
    def __init__(self): pass
    def sample(self, model: nn.Module, batch_size: int, input_dim: int, num_steps: int = 10, device: str = 'cpu') -> torch.Tensor:
        x = torch.randn(batch_size, input_dim, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size, 1), i * dt, device=device)
            with torch.no_grad():
                x1_pred = model(x, t)
            v = (x1_pred - x) / (1.0 - t + 1e-6) 
            x = x + v * dt
        return x
    
    def train_step(self, model: nn.Module, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        batch_size = x0.shape[0]
        device = x0.device
        t = torch.rand(batch_size, 1, device=device)
        x_t = (1 - t) * x0 + t * x1
        x1_pred = model(x_t, t)
        loss = torch.mean((x1_pred - x1) ** 2)
        return loss


if __name__ == "__main__":
    input_dim = 2
    model = VelocityNet(input_dim=input_dim)
    
    # --- 测试标准 Rectified Flow ---
    print("=== Rectified Flow ===")
    scheduler = RectifiedFlowScheduler()
    samples = scheduler.sample(model=model, batch_size=32, input_dim=input_dim, num_steps=20, device='cpu')
    print(f"生成样本形状: {samples.shape}")
    x0 = torch.randn(32, input_dim)
    x1 = torch.randn(32, input_dim)
    loss = scheduler.train_step(model, x0, x1)
    print(f"训练损失 (v-pred): {loss.item():.4f}")

    # --- 测试 JiT (x0-prediction) ---
    print("\n=== JiT (Just image Transformers) ===")
    jit_scheduler = JiTFlowScheduler()
    # 注意：同一个 VelocityNet 这里只是演示，实际 JiT 的网络输出维度也是 input_dim
    jit_samples = jit_scheduler.sample(model=model, batch_size=32, input_dim=input_dim, num_steps=20, device='cpu')
    print(f"生成样本形状: {jit_samples.shape}")
    jit_loss = jit_scheduler.train_step(model, x0, x1)
    print(f"训练损失 (x0-pred): {jit_loss.item():.4f}")