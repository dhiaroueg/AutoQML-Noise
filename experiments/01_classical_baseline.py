"""Baseline SVM sur le jeu de données jouet Breast Cancer de scikit-learn."""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.classical.baseline_models import make_classical_baseline
from src.evaluation import evaluate_classifier, save_metrics


def main():
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )
    metrics = evaluate_classifier(make_classical_baseline(), x_train, y_train, x_test, y_test)
    metrics["dataset"] = "sklearn_breast_cancer"
    metrics["model"] = "SVM-RBF"
    save_metrics(metrics, "results/metrics/classical_baseline.json")
    print(metrics)


if __name__ == "__main__":
    main()
