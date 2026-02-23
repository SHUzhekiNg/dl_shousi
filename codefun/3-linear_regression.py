from typing import List
class Solution:
    def linear_regression(self, points: List[List[float]]) -> List[float]:
        n = len(points)
        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points)
        mean_x = sum_x / n
        mean_y = sum_y / n

        num = 0.0  # Σ (x_i - x̄)(y_i - ȳ)
        den = 0.0  # Σ (x_i - x̄)^2
        for x, y in points:
            dx = x - mean_x
            dy = y - mean_y
            num += dx * dy
            den += dx * dx

        w = num / den
        b = mean_y - w * mean_x
        return [w, b]