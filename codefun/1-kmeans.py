import numpy as np
from typing import List, Tuple
class Solution:
    def kMeans(self, n: int, k: int, max_iter: int, nums: List[List[float]]) -> Tuple[List[List[float]], List[int]]:
        points = np.array(nums)
        centers = points[:k].copy()
        labels = np.full(n, -1)

        for _ in range(max_iter):
            distances = np.linalg.norm(points[:, np.newaxis] - centers, axis=2)**2
            new_labels = np.argmin(distances, axis=1)

            if np.array_equal(labels, new_labels):
                break
            
            labels = new_labels

            for j in range(k):
                mask = (labels == j)
                if np.any(mask):
                    centers[j] = points[mask].mean(axis=0)

        centers_str = [[f"{c[0]:.4f}", f"{c[1]:.4f}"] for c in centers]
        return centers_str, labels.tolist()