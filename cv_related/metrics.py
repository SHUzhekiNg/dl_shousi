import torch
import numpy as np

def psnr(prediction, target, max_pixel_value=255.0):
    mean = torch.mean((prediction - target) ** 2)
    if mean == 0: return float("inf")
    return 20 * torch.log10(max_pixel_value, torch.sqrt(mean))

def cosine_sim(pred, target):
    dot_prod = np.dot(pred, target)
    mag1 = np.linalg.norm(pred)
    mag2 = np.linalg.norm(target)
    return dot_prod / (mag1 * mag2)

def ssim(predictions, targets, window_size=11, sigma=1.5):
    # Ensure predictions and targets are in the same shape
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape.")
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu_x = torch.nn.functional.avg_pool2d(predictions, window_size, 1, window_size // 2)
    mu_y = torch.nn.functional.avg_pool2d(targets, window_size, 1, window_size // 2)
    
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    
    sigma_x_sq = torch.nn.functional.avg_pool2d(predictions ** 2, window_size, 1, window_size // 2) - mu_x_sq
    sigma_y_sq = torch.nn.functional.avg_pool2d(targets ** 2, window_size, 1, window_size // 2) - mu_y_sq
    sigma_xy = torch.nn.functional.avg_pool2d(predictions * targets, window_size, 1, window_size // 2) - mu_x_mu_y
    
    ssim_map = ((2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    
    return ssim_map.mean().item()

def lpips(predictions, targets):
    # Ensure predictions and targets are in the same shape
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape.")
    
    # Compute LPIPS using a pre-trained model (e.g., VGG)
    # This is a placeholder for the actual LPIPS computation
    lpips_value = torch.nn.functional.mse_loss(predictions, targets)
    return lpips_value.item()

def IoU(boxA,boxB):#x1,y1,x2,y2
    xA=max(boxA[0],boxB[0])
    yA=max(boxA[1],boxB[1])
    xB=min(boxA[2],boxB[2])
    yB=min(boxA[3],boxB[3])
    interArea=max(0,xB-xA+1)*max(0,yB-yA+1)
    boxAArea=(boxA[2]-boxA[0]+1)*(boxA[3]-boxA[1]+1)
    boxBArea=(boxB[2]-boxB[0]+1)*(boxB[3]-boxB[1]+1)
    iou=interArea/(boxAArea+boxBArea-interArea)
    return iou
#
def calculate_iou(bbox1, bbox2):
    # 计算bbox的面积
    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
    # 换一种更高级的方式计算面积
    # area2 = np.prod(bbox2[:, 2:] - bbox2[:, :2], axis=1)
    
    # 计算交集的左上角坐标和右下角坐标
    lt = np.maximum(bbox1[:, None, :2], bbox2[:, :2]) # [m, n, 2]
    rb = np.minimum(bbox1[:, None, 2:], bbox2[:, 2:])
    
    # 计算交集面积
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:,:,0] * wh[:,:,1]
    
    # 计算并集面积
    union = area1[:, None] + area2 - inter
    
    return inter / union