import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import matplotlib.pyplot as plt

def calculate_optimal_threshold(y_true, y_pred_prob):
    """
    Calculate the optimal threshold that maximizes F1 score
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_prob)
    
    # Calculate F1 score for each threshold
    f1_scores = []
    for threshold in thresholds:
        predictions = (y_pred_prob >= threshold).astype(int)
        f1 = f1_score(y_true, predictions)
        f1_scores.append(f1)
    
    # Find threshold with maximum F1 score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    return best_threshold, best_f1

def plot_precision_recall_curve(y_true, y_pred_prob, save_path=None):
    """
    Plot precision-recall curve and calculate average precision
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_prob)
    ap = average_precision_score(y_true, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='.', label=f'AP = {ap:.3f}')
    
    # Mark the optimal threshold
    optimal_threshold, best_f1 = calculate_optimal_threshold(y_true, y_pred_prob)
    
    # Find the precision and recall at the optimal threshold
    optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    optimal_precision = precisions[optimal_idx]
    optimal_recall = recalls[optimal_idx]
    
    plt.scatter(optimal_recall, optimal_precision, c='red', s=100, 
                label=f'Optimal (F1={best_f1:.3f}, thresh={optimal_threshold:.2f})', 
                zorder=10)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
        
    return optimal_threshold, best_f1

def calculate_metrics_at_threshold(y_true, y_pred_prob, threshold=0.5):
    """
    Calculate various metrics at a given threshold
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # True positives, false positives, true negatives, false negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # False positive rate and false negative rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'fpr': fpr,
        'fnr': fnr
    }
    
    return metrics
