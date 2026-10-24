"""Baselines logistique et SVM sur les mêmes partitions stratifiées."""

import argparse
import json

from _common import ROOT, get_partitions, library_versions
from src.classical.baseline_models import make_classical_baseline, make_logistic_regression
from src.config import DATASET_NAME, SEED
from src.evaluation import evaluate_classifier, save_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Utilise un sous-échantillon de démonstration")
    args = parser.parse_args()
    parts = get_partitions(quick=args.quick)
    results = {}
    for name, model in (
        ("logistic_regression", make_logistic_regression(SEED)),
        ("svm_rbf", make_classical_baseline(SEED)),
    ):
        metrics = evaluate_classifier(model, parts.x_train, parts.y_train, parts.x_test, parts.y_test)
        metrics.update({"dataset": DATASET_NAME, "model": name, "seed": SEED,
                        "n_train": len(parts.y_train), "n_test": len(parts.y_test)})
        metrics.update(library_versions())
        results[name] = metrics
    output = ROOT / "results" / "metrics" / "classical_baselines.json"
    save_metrics(results, output)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
