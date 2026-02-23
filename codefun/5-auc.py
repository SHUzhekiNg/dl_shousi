from typing import List

class Solution:
    def calculate_auc(self, FPR: List[float], TPR: List[float]) -> float:
        """计算曲线下面积 (AUC) 的函数"""
        n = len(FPR)  # 获取假阳性率列表的长度
        auc = 0.0  # 初始化AUC值
        
        # 遍历FPR和TPR，计算相邻点形成的梯形面积
        for i in range(1, n):
            width = FPR[i] - FPR[i - 1]  # 计算梯形的宽度
            height = (TPR[i] + TPR[i - 1]) / 2  # 计算梯形的高度
            auc += width * height  # 累加当前梯形的面积
            
        return auc  # 返回最终计算得到的AUC值
