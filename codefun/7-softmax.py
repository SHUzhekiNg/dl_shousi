import math
from typing import List, Tuple

class Solution:
    def softmax(self, z: List[float]) -> List[float]:
        """
        计算Softmax函数输出的概率分布向量
        
        :param z: 输入向量
        :return: 概率分布向量
        """
        # 1. 找到输入向量的最大值
        max_z = max(z)
        
        # 2. 计算指数，使用最大值以提高数值稳定性
        exp_values = [math.exp(x - max_z) for x in z]
        
        # 3. 计算指数和
        sum_exp = sum(exp_values)
        
        # 4. 计算Softmax输出
        return [x / sum_exp for x in exp_values]

    def softmax_cross_entropy(self, logits: List[float], labels: List[int]) -> Tuple[List[float], float]:
        # ---------- 1. 计算 Softmax 概率（数值稳定版） ----------
        # 减去最大值，防止 exp 溢出
        max_logit = max(logits)
        exp_shifted = [math.exp(x - max_logit) for x in logits]
        sum_exp = sum(exp_shifted)
        probs = [v / sum_exp for v in exp_shifted]

        # ---------- 2. 计算交叉熵损失 ----------
        # 损失: L = -sum(y_i * log(p_i))
        # 在 one-hot 情况下，只会取到 y_i=1 对应的那一项
        eps = 1e-15
        loss = 0.0
        for y, p in zip(labels, probs):
            if y == 1:
                # 防止 log(0)
                loss -= math.log(max(p, eps))

        return probs, loss
