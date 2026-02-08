import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPM:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].reshape(-1, 1, 1, 1)
        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
    
    @torch.no_grad()
    def p_sample(self, model, xt, t):
        noise_pred = model(xt, t)
        alpha_bar_t = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        alpha_t = self.alphas[t].reshape(-1, 1, 1, 1)
        beta_t = self.betas[t].reshape(-1, 1, 1, 1)

        mean = torch.rsqrt(alpha_t) * (xt - beta_t * torch.rsqrt(1 - alpha_bar_t) * noise_pred)
        noise = torch.randn_like(xt)
        mask = (t>0).float().reshape(-1, 1, 1, 1)
        return mean + mask * torch.sqrt(beta_t) * noise

    @torch.no_grad()
    def sample(self, model, shape):
        xt = torch.randn(shape).to("cuda")
        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, dtype=torch.long, device="cuda")
            xt = self.p_sample(model, xt, t_batch)
        return xt

    def train_loss(self, model, x0):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.T, (batch_size, ), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        noise_pred = model(xt, t)
        loss = F.mse_loss(noise_pred, noise)
        return loss

class DDIM(DDPM):
    def __init__(self, T=1000, beta_start=0.0001, beta_end=0.02):
        super().__init__(T, beta_start, beta_end)
        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)

    @torch.no_grad()
    def p_sample(self, model, xt, t, eta=0.0):
        eps_pred = model(xt, t)
        alpha_bar_t = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        alpha_bar_prev = self.alpha_bars_prev[t].reshape(-1, 1, 1, 1)
        
        pred_x0 = (xt - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
        
        pred_dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * eps_pred
        noise = torch.randn_like(xt) if eta > 0 else 0
        return torch.sqrt(alpha_bar_prev) * pred_x0 + pred_dir_xt + sigma_t * noise
