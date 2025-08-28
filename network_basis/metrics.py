import torch


def recall(predictions, targets, threshold=0.5):
    """
    Compute the recall metric.
    
    Args:
        predictions (torch.Tensor): Predicted probabilities.
        targets (torch.Tensor): Ground truth labels.
        threshold (float): Threshold to convert probabilities to binary predictions.
        
    Returns:
        float: Computed recall.
    """
    predicted_labels = (predictions >= threshold).float()
    true_positives = (predicted_labels * targets).sum().item()
    false_negatives = ((1 - predicted_labels) * targets).sum().item()
    
    if true_positives + false_negatives == 0:
        return 0.0
    
    return true_positives / (true_positives + false_negatives)


def precision(predictions, targets, threshold=0.5):
    """
    Compute the precision metric.
    
    Args:
        predictions (torch.Tensor): Predicted probabilities.
        targets (torch.Tensor): Ground truth labels.
        threshold (float): Threshold to convert probabilities to binary predictions.
        
    Returns:
        float: Computed precision.
    """
    predicted_labels = (predictions >= threshold).float()
    true_positives = (predicted_labels * targets).sum().item()
    false_positives = (predicted_labels * (1 - targets)).sum().item()
    
    if true_positives + false_positives == 0:
        return 0.0
    
    return true_positives / (true_positives + false_positives)


def f1_score(predictions, targets, threshold=0.5):
    """
    Compute the F1 score metric.
    
    Args:
        predictions (torch.Tensor): Predicted probabilities.
        targets (torch.Tensor): Ground truth labels.
        threshold (float): Threshold to convert probabilities to binary predictions.
        
    Returns:
        float: Computed F1 score.
    """
    prec = precision(predictions, targets, threshold)
    rec = recall(predictions, targets, threshold)
    
    if prec + rec == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)


def mapk(predictions, targets, k=10):
    """
    Compute the Mean Average Precision at k (mAP@k) metric.
    
    Args:
        predictions (torch.Tensor): Predicted scores for each item.
        targets (torch.Tensor): Ground truth binary relevance labels.
        k (int): Number of top items to consider.
        
    Returns:
        float: Computed mAP@k.
    """
    if predictions.shape[0] != targets.shape[0]:
        raise ValueError("Predictions and targets must have the same number of samples.")
    
    average_precisions = []
    
    for i in range(predictions.shape[0]):
        pred_scores = predictions[i]
        true_labels = targets[i]
        
        _, indices = torch.topk(pred_scores, k)
        selected_labels = true_labels[indices]
        
        precision_at_k = selected_labels.sum().item() / k
        average_precisions.append(precision_at_k)
    
    return torch.mean(torch.tensor(average_precisions)).item()



def cross_entropy_loss(predictions, targets):
    """
    Compute the Cross-Entropy Loss metric.
    
    Args:
        predictions (torch.Tensor): Predicted probabilities (logits).
        targets (torch.Tensor): Ground truth labels (one-hot encoded or class indices).
        
    Returns:
        float: Computed Cross-Entropy Loss.
    """
    if predictions.shape[0] != targets.shape[0]:
        raise ValueError("Predictions and targets must have the same number of samples.")
    
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(predictions, targets).item()
