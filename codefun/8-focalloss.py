import math
class Solution:
    def focal_loss(self, alpha: float, gamma: float, p: float, y: int) -> float:
        if y == 1:
            mod_factor = (1.0 - p) ** gamma
            loss = -alpha * mod_factor * math.log(p)
        else:
            mod_factor = p ** gamma
            loss = -alpha * mod_factor * math.log(1.0 - p)

        return loss


