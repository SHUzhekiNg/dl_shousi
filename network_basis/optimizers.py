import torch
import math


class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] + p.grad
                p.data -= self.lr * self.velocities[i]
            else:
                p.data -= self.lr * p.grad


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)


class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            p.data -= self.lr * self.weight_decay * p.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(100, 1)
    y = 3 * x + 2 + torch.randn(100, 1) * 0.1

    for OptClass, name in [(SGD, "SGD"), (Adam, "Adam"), (AdamW, "AdamW")]:
        w = torch.randn(1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        if name == "SGD":
            opt = OptClass([w, b], lr=0.01, momentum=0.9)
        elif name == "Adam":
            opt = OptClass([w, b], lr=0.1)
        else:
            opt = OptClass([w, b], lr=0.1, weight_decay=0.01)

        for epoch in range(200):
            y_pred = x * w + b
            loss = ((y_pred - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"{name}: w={w.item():.4f}, b={b.item():.4f}, loss={loss.item():.6f}")
