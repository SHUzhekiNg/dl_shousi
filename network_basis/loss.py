import torch
import numpy as np

def cross_entropy_loss(prediction, target):
    log_pred = torch.log_softmax(prediction, dim=-1)
    loss = -torch.sum(log_pred * target, dim=-1)
    return loss.mean()

def binary_cross_entropy_loss(prediction, target):
    # **
    epsilon = 1e-7
    prediction = torch.clamp(prediction, epsilon, 1 - epsilon)

    return torch.mean(-target * torch.log(prediction) - (1 - target) * torch.log(1 - prediction))

def mse_loss(pred, target):
    return torch.mean((target - pred) ** 2)

def kl_loss(pred, target):
    log_preds = torch.log_softmax(pred, dim=-1)
    log_target = torch.log_softmax(target, dim=-1)
    return torch.sum(target * (log_preds - log_target), dim=-1).mean()


def hinge_loss(predictions, targets):
    return torch.mean(torch.clamp(1 - predictions * targets, min=0))

