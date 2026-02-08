import scipy
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


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    
    # 计算每个框的面积
    # 如果点坐标是像素值，这个边是要+1的，e.g. (ax2 - ax1 + 1)，四个括号都要
    # 在一般的现代目标检测中，使用浮点坐标系统，亚像素精度，所以直接是几何面积。
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    
    iou_x1 = np.maximum(ax1, bx1)
    iou_y1 = np.maximum(ay1, by1)
    iou_x2 = np.minimum(ax2, bx2)
    iou_y2 = np.minimum(ay2, by2)
    
    iou_w = iou_x2 - iou_x1
    iou_h = iou_y2 - iou_y1
    area_iou = iou_w * iou_h
    iou = area_iou / (area_a + area_b - area_iou)
    return iou
    
    
def batch_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1) # shape (N, 4), format [x1, y1, x2, y2]
    boxes2 = np.array(boxes2) # shape (M, 4), format [x1, y1, x2, y2]
    
    # 计算每个框的面积
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)
    
    # 扩展维度以支持广播: boxes1 (N,1,4), boxes2 (1,M,4)
    boxes1_expanded = boxes1[:, None, :]  # (N, 1, 4)
    boxes2_expanded = boxes2[None, :, :]  # (1, M, 4)

    lt = np.maximum(boxes1_expanded[..., :2], boxes2_expanded[..., :2])  # (N, M, 2)
    rb = np.minimum(boxes1_expanded[..., 2:], boxes2_expanded[..., 2:])  # (N, M, 2)
    
    # 计算交集的宽高，如果为负则置为0
    wh = np.clip(rb - lt, a_min=0, a_max=None)  # (N, M, 2)
    intersection = wh[..., 0] * wh[..., 1]  # (N, M)
    
    # 使用广播：area1 (N,1) + area2 (1,M) - intersection (N,M)
    union = area1[:, None] + area2[None, :] - intersection  # (N, M)
    
    # 计算IoU，避免除零
    # shape (N, M), iou_matrix[i,j] = IoU(boxes1[i], boxes2[j])
    iou = intersection / np.maximum(union, 1e-8)
    
    return iou


def nms(boxes, scores, iou_threshold):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1: break
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        
        order = order[inds + 1]
        
    return keep


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)*2.0)
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

