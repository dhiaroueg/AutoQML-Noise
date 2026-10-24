"""Métriques et sérialisation des évaluations."""

import json
from pathlib import Path
from time import perf_counter

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_classifier(model, x_train, y_train, x_test, y_test):
    start = perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = perf_counter() - start
    predictions = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "fit_seconds": fit_seconds,
    }


def save_metrics(metrics, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
