"""Метрики оценки качества моделей и оптимизация бизнес-порога отсечки."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score
)


def calculate_gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Расчет банковского коэффициента Джини: Gini = 2 * ROC_AUC - 1."""
    auc = roc_auc_score(y_true, y_prob)
    return 2.0 * auc - 1.0


def calculate_credit_score(y_prob: float | np.ndarray, min_score: int = 300, max_score: int = 850) -> int | np.ndarray:
    """Перевод вероятности дефолта в шкалу кредитного скоринга (по аналогии со шкалой FICO: 300 - 850).

    Чем выше балл, тем надежнее клиент.
    """
    prob = np.clip(y_prob, 0.0, 1.0)
    score = max_score - prob * (max_score - min_score)
    if isinstance(score, np.ndarray):
        return np.round(score).astype(int)
    return int(round(score))


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 8.0,
    steps: int = 200
) -> dict:
    """Поиск порога классификации, минимизирующего финансовые потери банка.

    cost_fp: потеря маржи от отказа платежеспособному заемщику (False Positive).
    cost_fn: прямой убыток от невозврата выданного кредита (False Negative).
    """
    thresholds = np.linspace(0.01, 0.99, steps)
    best_threshold = 0.5
    min_cost = float("inf")
    best_stats = {}

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        total_cost = fp * cost_fp + fn * cost_fn

        if total_cost < min_cost:
            min_cost = total_cost
            best_threshold = float(t)
            best_stats = {
                "threshold": round(best_threshold, 4),
                "total_cost": round(float(total_cost), 2),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "cost_fp": cost_fp,
                "cost_fn": cost_fn
            }

    return best_stats


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Расчет полного набора метрик классификации и калибровки."""
    y_pred = (y_prob >= threshold).astype(int)
    roc_auc = roc_auc_score(y_true, y_prob)
    gini = 2.0 * roc_auc - 1.0
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "roc_auc": round(float(roc_auc), 4),
        "gini": round(float(gini), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }
