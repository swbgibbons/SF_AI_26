"""Classification metrics (no scikit-learn dependency)."""

import numpy as np

from . import CLASSES


def confusion_matrix(y_true, y_pred, n=len(CLASSES)):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def classification_report(y_true, y_pred):
    """Per-class precision/recall/F1, accuracy, macro-F1. Rows of cm = truth, cols = prediction."""
    cm = confusion_matrix(y_true, y_pred)
    per_class = {}
    for i, name in enumerate(CLASSES):
        tp = cm[i, i]
        support = int(cm[i].sum())
        predicted = int(cm[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": round(float(precision), 4), "recall": round(float(recall), 4),
                           "f1": round(float(f1), 4), "support": support}
    present = [c for c in CLASSES if per_class[c]["support"]]
    return {
        "accuracy": round(float(np.trace(cm) / max(cm.sum(), 1)), 4),
        # Macro over classes that appear, so a missing class doesn't drag the score to 0.
        "macro_f1": round(float(np.mean([per_class[c]["f1"] for c in present])) if present else 0.0, 4),
        # The number that matters most: how many damaged roofs we catch.
        "damaged_recall": per_class["damaged"]["recall"],
        "per_class": per_class,
        "confusion_matrix": {"rows_true_cols_pred": CLASSES, "values": cm.tolist()},
    }


def format_confusion(cm):
    w = max(len(c) for c in CLASSES) + 2
    lines = ["true \\ pred".ljust(w) + "".join(c.rjust(w) for c in CLASSES)]
    for name, row in zip(CLASSES, cm):
        lines.append(name.ljust(w) + "".join(str(v).rjust(w) for v in row))
    return "\n".join(lines)
