import math
class Solution:
    def sigmoid(self, x: list[float]) -> list[float]:
        """
        计算输入向量的 Sigmoid 值。
        """
        result: list[float] = []
        for value in x:
            # 将输入转换为浮点数以保证计算稳定
            v = float(value)
            if v >= 0.0:
                # 数值稳定版本：对非负数使用 1 / (1 + exp(-x))
                z = math.exp(-v)
                result.append(1.0 / (1.0 + z))
            else:
                # 数值稳定版本：对负数使用 exp(x) / (1 + exp(x))
                z = math.exp(v)
                result.append(z / (1.0 + z))
        return result
