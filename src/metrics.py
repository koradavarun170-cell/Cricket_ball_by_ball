"""
Evaluation metrics and threshold optimization utilities.
"""

from typing import Tuple, Optional, Iterable
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def find_best_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: Optional[Iterable[float]] = None
) -> Tuple[float, float]:
    """
    Scans a range of probability thresholds to maximize F1 score.
    Returns: (best_threshold, best_f1)
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.01)

    best_threshold = 0.5
    best_f1 = 0.0

    for t in thresholds:
        pred = (probs >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = round(float(t), 4)

    return best_threshold, best_f1


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: Optional[np.ndarray] = None,
    title: str = "Model Evaluation"
) -> dict:
    """
    Computes and prints classification metrics:
    Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    roc_auc = None
    pr_auc = None
    if probs is not None:
        roc_auc = roc_auc_score(y_true, probs)
        pr_auc = average_precision_score(y_true, probs)

    print(f"\n{'='*20} {title} {'='*20}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC  : {roc_auc:.4f}")
    if pr_auc is not None:
        print(f"PR-AUC   : {pr_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
    }
