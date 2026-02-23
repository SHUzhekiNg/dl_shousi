import math
from typing import List

def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

class Solution:
    def logisticRegression(self, samples: List[List[float]], lr: float, epochs: int) -> List[float]:
            w1 = 0.0
            w2 = 0.0
            b = 0.0
            n = len(samples)

            for _ in range(epochs):
                dw1 = 0.0
                dw2 = 0.0
                db = 0.0

                for x1, x2, y in samples:
                    z = w1 * x1 + w2 * x2 + b
                    y_pred = sigmoid(z)
                    diff = y_pred - y

                    dw1 += diff * x1
                    dw2 += diff * x2
                    db  += diff

                dw1 /= n
                dw2 /= n
                db  /= n

                w1 -= lr * dw1
                w2 -= lr * dw2
                b  -= lr * db

            return [w1, w2, b]
