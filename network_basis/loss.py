import torch
import numpy as np
import torch.nn.functional as F

def cross_entropy_loss(pred, target):
    log_pred = torch.log_softmax(pred, dim=-1)
    loss = -torch.sum(log_pred * target, dim=-1)
    return loss.mean()

def binary_cross_entropy_loss(pred, target):
    epsilon = 1e-7
    pred = torch.clamp(pred, epsilon, 1 - epsilon)
    return -torch.mean(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))

def mse_loss(pred, target):
    return torch.mean((target - pred) ** 2)

def kl_loss(pred, target):
    log_preds = torch.log_softmax(pred, dim=-1)
    log_target = torch.log_softmax(target, dim=-1)
    return torch.sum(target * (log_preds - log_target), dim=-1).mean()


def hinge_loss(pred, target):
    return torch.mean(torch.clamp(1 - pred * target, min=0))

# 这是一个高度简化的预期目标版本
def ppo_loss_with_gae_entropy(old_policy_logprobs, new_policy_logprobs, advantages, kl_penalty_coef, clip_epsilon, entropy_bonus_coef):
    """概念性 PPO 损失函数，带有 GAE 和熵奖励（简化版）。"""

    ratio = np.exp(new_policy_logprobs - old_policy_logprobs)  # 概率比

    # 剪切代理目标（限制策略变化）
    surrogate_objective = np.minimum(ratio * advantages, np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages)
    policy_loss = -np.mean(surrogate_objective)

    # KL 散度惩罚（保持接近旧策略）
    kl_divergence = np.mean(new_policy_logprobs - old_policy_logprobs)
    kl_penalty = kl_penalty_coef * kl_divergence

    # 熵奖励（鼓励探索）
    entropy = -np.mean(new_policy_logprobs)  # 简化版熵（概率越高 = 熵越低，取负值以最大化熵）
    entropy_bonus = entropy_bonus_coef * entropy

    total_loss = policy_loss + kl_penalty - entropy_bonus  # 减去熵奖励，因为我们希望*最大化*熵
    return total_loss

# 注意：这不是实际公式。
# 这是一个高度简化的预期目标版本
def dpo_loss(policy_logits_preferred, policy_logits_dispreferred, ref_logits_preferred, ref_logits_dispreferred, beta_kl):
    """概念性 DPO 损失函数（简化版——直接使用 logits）。"""

    # 1. 从 logits 中获取对数概率（当前和参考模型的首选和非首选响应）
    policy_logprob_preferred = F.log_softmax(policy_logits_preferred, dim=-1).gather(...)  # 提取首选响应中实际标记的对数概率
    policy_logprob_dispreferred = F.log_softmax(policy_logits_dispreferred, dim=-1).gather(...)  # 提取非首选响应中实际标记的对数概率
    ref_policy_logprob_preferred = F.log_softmax(ref_logits_preferred, dim=-1).gather(...)  # 同样适用于参考模型
    ref_policy_logprob_dispreferred = F.log_softmax(ref_logits_dispreferred, dim=-1).gather(...)

    # 2. 计算对数比率（使用对数概率——如前所述）
    log_ratio = policy_logprob_preferred - policy_logprob_dispreferred - (ref_policy_logprob_preferred - ref_policy_logprob_dispreferred)

    # 3. 偏好概率（Bradley-Terry 模型——隐式奖励信号）
    preference_prob = 1 / (1 + np.exp(-beta_kl * log_ratio))

    # 4. 二元交叉熵损失（直接优化策略）
    dpo_loss = -np.log(preference_prob + 1e-8)
    return dpo_loss

# 注意：这不是实际公式。
# 这是一个高度简化的预期目标版本
def grae_advantages(rewards):
    """概念性组相对优势估计（结果监督）。"""
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    normalized_rewards = (rewards - mean_reward) / (std_reward + 1e-8)
    advantages = normalized_rewards  # 对于结果监督，优势 = 归一化奖励
    return advantages


def grpo_loss(old_policy_logprobs_group, new_policy_logprobs_group, group_advantages, kl_penalty_coef, clip_epsilon):
    """概念性 GRPO 损失函数（对一组响应取平均）。"""
    group_loss = 0
    for i in range(len(group_advantages)):  # 遍历组内的每个响应
        advantage = group_advantages[i]
        new_policy_logprob = new_policy_logprobs_group[i]
        old_policy_logprob = old_policy_logprobs_group[i]

        ratio = np.exp(new_policy_logprob - old_policy_logprob)
        clipped_ratio = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        surrogate_objective = np.minimum(ratio * advantage, clipped_ratio * advantage)
        policy_loss = -surrogate_objective

        kl_divergence = new_policy_logprob - old_policy_logprob
        kl_penalty = kl_penalty_coef * kl_divergence
        group_loss += (policy_loss + kl_penalty)  # 累加组内每个响应的损失

    return group_loss / len(group_advantages)  # 对组内损失取平均
