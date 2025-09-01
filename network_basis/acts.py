import torch
import torch.nn as nn
import numpy as np

# f(x) = max(0, x)
def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0, x, torch.zeros_like(x))

# f(x) = x if x > 0 else αx, 其中 α 是小正数
class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01, learnable_slope=False):
        super(LeakyReLU, self).__init__()
        if learnable_slope:
            self.negative_slope = nn.Parameter(torch.tensor(float(negative_slope)))
        else:
            self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, x * self.negative_slope)

# f(x) = 1 / (1 + e^(-x))
# 输出范围 (0,1)，适合二分类
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + np.exp(-x))

# f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
# 输出范围 (-1,1)，梯度较 Sigmoid 好
def tanh(x: torch.Tensor) -> torch.Tensor:
    exp_x = torch.exp(x)
    exp_neg_x = torch.exp(-x)
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

# 5. GELU (Gaussian Error Linear Unit)
# 公式: f(x) = x * Φ(x) ≈ x * sigmoid(1.702x) (近似实现)
# 特点: 平滑非线性，广泛用于 Transformer
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        # 自定义实现：使用 sigmoid 近似 GELU
        # Φ(x) ≈ sigmoid(1.702x)
        return x * (1 / (1 + torch.exp(-1.702 * x)))

# 6. SiLU (Sigmoid Linear Unit, 即 Swish)
# 公式: f(x) = x * sigmoid(βx)
# 特点: 平滑，性能优于 ReLU
class SiLU(nn.Module):
    def __init__(self, beta=1.0, learnable_beta=False):
        super(SiLU, self).__init__()
        if learnable_beta:
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.beta = beta

    def forward(self, x):
        # 自定义实现：x * sigmoid(βx)
        return x * (1 / (1 + torch.exp(-self.beta * x)))

# 7. SwiGLU (Swish-Gated Linear Unit)
# 公式: f(x) = (Swish(xW1 + b1)) * (xW2 + b2)
# 特点: 结合 Swish 和门控机制，适合高效 Transformer
class SwiGLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.w = nn.Linear(input_dim, output_dim)
        self.v = nn.Linear(input_dim, output_dim)

    def forward(self,x):
        w_out = self.w(x)
        v_out = self.v(x)

        gate = v_out * torch.sigmoid(v_out)
        
        return w_out * gate
    
# 8. Softmax
# 公式: f(x_i) = e^(x_i) / Σ(e^(x_j))
# 特点: 输出概率分布，适合多分类
class Softmax(nn.Module):
    def __init__(self, dim=-1):
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        # 自定义实现：Softmax(x) = e^(x - max(x)) / Σ(e^(x - max(x)))
        # 减去最大值以提高数值稳定性
        x_shifted = x - torch.max(x, dim=self.dim, keepdim=True)[0]
        exp_x = torch.exp(x_shifted)
        sum_exp_x = torch.sum(exp_x, dim=self.dim, keepdim=True)
        return exp_x / sum_exp_x



if __name__ == "__main__":
    # 测试输入
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    batch_x = torch.randn(2, 3, 128)  # 用于 SwiGLU 的批处理输入

    # 测试每个激活函数
    leaky_relu = LeakyReLU(negative_slope=0.01)
    gelu = GELU()
    silu = SiLU(beta=1.0, learnable_beta=True)
    swiglu = SwiGLU(input_dim=128, output_dim=256, beta=1.0, learnable_beta=True)

    print("ReLU output:", relu(x))
    print("Leaky ReLU output:", leaky_relu(x))
    print("Sigmoid output:", sigmoid(x))
    print("Tanh output:", tanh(x))
    print("GELU output:", gelu(x))
    print("SiLU output:", silu(x))
    print("SwiGLU output shape:", swiglu(batch_x).shape)

    # 检查 SwiGLU 的可学习参数
    print("\nSwiGLU learnable parameters:")
    for name, param in swiglu.named_parameters():
        print(f"{name}: {param.shape}")
    