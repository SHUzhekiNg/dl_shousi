from typing import List
import numpy as np
class Solution:
    def knn_classification(self, X: List[List[float]], y: List[int], x_test: List[float], K: int) -> int:
        """实现KNN分类，根据最近的K个邻居进行多数投票得到最终预测标签"""
        dis = np.linalg.norm(np.array(X) - np.array(x_test), axis=1)
        indices = np.argsort(dis)[:K]
        labels = [y[i] for i in indices]
        y_pred = max(set(labels), key=labels.count)
        return y_pred