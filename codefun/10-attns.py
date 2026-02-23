import math
from typing import List


class Solution:
    def selfAttention(
        self,
        n: int,
        d: int,
        X: List[List[float]],
        Wq: List[List[float]],
        Wk: List[List[float]],
        Wv: List[List[float]],
    ) -> List[List[str]]:

        # 辅助函数：矩阵乘法
        def matmul(A, B):
            rows_A, cols_A = len(A), len(A[0])
            rows_B, cols_B = len(B), len(B[0])
            # 结果矩阵初始化
            C = [[0.0] * cols_B for _ in range(rows_A)]
            for i in range(rows_A):
                for k in range(cols_A):
                    val = A[i][k]
                    if val == 0:
                        continue
                    for j in range(cols_B):
                        C[i][j] += val * B[k][j]
            return C

        # 辅助函数：矩阵转置
        def transpose(A):
            return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

        # 1. 线性映射 Q, K, V
        Q = matmul(X, Wq)
        K = matmul(X, Wk)
        V = matmul(X, Wv)

        # 2. 缩放点积 S = (Q @ K.T) / sqrt(d)
        K_T = transpose(K)
        S = matmul(Q, K_T)

        scale = 1.0 / math.sqrt(d)
        for i in range(n):
            for j in range(n):
                S[i][j] *= scale

        # 3. Softmax
        A = []
        for i in range(n):
            row = S[i]
            max_val = max(row)  # 数值稳定性处理
            exps = [math.exp(x - max_val) for x in row]
            sum_exps = sum(exps)
            A.append([e / sum_exps for e in exps])

        # 4. 输出 O = A @ V
        O = matmul(A, V)

        # 格式化输出：保留4位小数，不足补0
        res = []
        for row in O:
            res.append([f"{x:.4f}" for x in row])

        return res
