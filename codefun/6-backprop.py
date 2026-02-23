import math
from typing import List, Tuple


def sigmoid(z: float) -> float:
    """数值稳定版 sigmoid，避免 exp 溢出。"""
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


class Solution:
    def twoLayerNN(
        self,
        lr: float,
        W1: List[List[float]],  # 2x3
        b1: List[float],  # 长度 3
        W2: List[float],  # 长度 3
        b2: float,  # 标量
        x: List[float],  # 长度 2
        y: int,  # 0 或 1
    ) -> Tuple[
        float,  # loss
        List[List[float]],  # 更新后的 W1（2x3）
        List[float],  # 更新后的 b1（3）
        List[float],  # 更新后的 W2（3）
        float,  # 更新后的 b2
    ]:
        # ---------- 1. Forward 前向传播 ----------
        # 1) 隐藏层线性变换 z1 = W1 x + b1
        z1 = []
        for j in range(3):
            z = x[0] * W1[0][j] + x[1] * W1[1][j] + b1[j]
            z1.append(z)

        # 2) ReLU 激活 h = ReLU(z1)
        h = [max(0.0, z) for z in z1]

        # 3) 输出层线性 z2 = W2^T h + b2
        z2 = 0.0
        for j in range(3):
            z2 += W2[j] * h[j]
        z2 += b2

        # 4) Sigmoid 输出 y_hat
        y_hat = sigmoid(z2)

        # 5) BCE 损失（clip 避免 log(0)）
        eps = 1e-12
        y_hat_clipped = min(max(y_hat, eps), 1.0 - eps)
        loss = -(y * math.log(y_hat_clipped) + (1 - y) * math.log(1 - y_hat_clipped))

        # ---------- 2. Backward 反向传播 ----------
        # 输出层：dL/dz2 = y_hat - y（BCE + Sigmoid 合并后的结果）
        dL_dz2 = y_hat - y

        # 输出层参数梯度
        # dL/dW2_j = dL/dz2 * h_j
        dW2 = [dL_dz2 * h_j for h_j in h]
        # dL/db2 = dL/dz2
        db2 = dL_dz2

        # 传回隐藏层：dL/dh_j = dL/dz2 * W2_j
        dL_dh = [dL_dz2 * W2_j for W2_j in W2]

        # ReLU 反向：z1_j <= 0 时梯度为 0
        dL_dz1 = []
        for j in range(3):
            if z1[j] > 0:
                dL_dz1.append(dL_dh[j])
            else:
                dL_dz1.append(0.0)

        # 隐藏层参数梯度
        # dL/dW1[i][j] = x_i * dL/dz1_j
        dW1 = [[0.0] * 3 for _ in range(2)]
        for i in range(2):
            for j in range(3):
                dW1[i][j] = x[i] * dL_dz1[j]

        # dL/db1_j = dL/dz1_j
        db1 = dL_dz1[:]  # 复制一份

        # ---------- 3. 参数更新（Gradient Descent） ----------
        # 更新 W1
        W1_new = [[0.0] * 3 for _ in range(2)]
        for i in range(2):
            for j in range(3):
                W1_new[i][j] = W1[i][j] - lr * dW1[i][j]

        # 更新 b1
        b1_new = [0.0] * 3
        for j in range(3):
            b1_new[j] = b1[j] - lr * db1[j]

        # 更新 W2
        W2_new = [0.0] * 3
        for j in range(3):
            W2_new[j] = W2[j] - lr * dW2[j]

        # 更新 b2
        b2_new = b2 - lr * db2

        # ---------- 4. 返回 loss 和更新后的参数 ----------
        return loss, W1_new, b1_new, W2_new, b2_new
