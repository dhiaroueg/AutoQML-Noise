"""Baselines classiques simples à comparer au VQC."""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_classical_baseline(random_state: int = 42):
    """Construit une SVM RBF avec normalisation intégrée au pipeline."""
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=random_state))
