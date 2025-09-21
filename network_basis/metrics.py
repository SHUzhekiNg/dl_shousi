import torch


def recall(pred, targets, threshold=0.5):
    predicted_labels = (pred >= threshold).float()
    true_positives = (predicted_labels * targets).sum().item()
    false_negatives = ((1 - predicted_labels) * targets).sum().item()
    
    if true_positives + false_negatives == 0:
        return 0.0
    
    return true_positives / (true_positives + false_negatives)


def precision(pred, targets, threshold=0.5):
    predicted_labels = (pred >= threshold).float()
    true_positives = (predicted_labels * targets).sum().item()
    false_positives = (predicted_labels * (1 - targets)).sum().item()
    
    if true_positives + false_positives == 0:
        return 0.0
    
    return true_positives / (true_positives + false_positives)


def f1_score(pred, targets, threshold=0.5):
    prec = precision(pred, targets, threshold)
    rec = recall(pred, targets, threshold)

    if prec + rec == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)


def mapk(pred, targets, k=10):
    if pred.shape[0] != targets.shape[0]:
        raise ValueError("Predictions and targets must have the same number of samples.")
    
    average_precisions = []

    for i in range(pred.shape[0]):
        pred_scores = pred[i]
        true_labels = targets[i]
        
        _, indices = torch.topk(pred_scores, k)
        selected_labels = true_labels[indices]
        
        precision_at_k = selected_labels.sum().item() / k
        average_precisions.append(precision_at_k)
    
    return torch.mean(torch.tensor(average_precisions)).item()

