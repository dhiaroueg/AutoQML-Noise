"""Métriques et sérialisation des évaluations."""

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(y_true, predictions) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_classifier(model, x_train, y_train, x_test, y_test):
    start = perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = perf_counter() - start
    predict_start = perf_counter()
    predictions = model.predict(x_test)
    predict_seconds = perf_counter() - predict_start
    metrics = classification_metrics(y_test, predictions)
    metrics["fit_seconds"] = fit_seconds
    metrics["predict_seconds"] = predict_seconds
    return metrics


def save_metrics(metrics, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        raise ValueError("Aucune ligne à écrire")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
