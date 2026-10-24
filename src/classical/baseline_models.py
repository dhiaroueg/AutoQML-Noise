"""Baselines classiques simples à comparer au VQC."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_classical_baseline(random_state: int = 42):
    """Construit une SVM RBF avec normalisation intégrée au pipeline."""
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=random_state))


def make_logistic_regression(random_state: int = 42):
    """Régression logistique, normalisée et entraînée uniquement sur train."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=random_state),
    )
