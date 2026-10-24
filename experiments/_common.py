"""Aides partagées par les scripts d'expérience, sans effet de bord à l'import."""

from pathlib import Path
from importlib.metadata import version
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    N_QUBITS,
    QUANTUM_TEST_LIMIT,
    QUANTUM_TRAIN_LIMIT,
    QUANTUM_VALIDATION_LIMIT,
    SEED,
)
from src.preprocessing import (  # noqa: E402
    QuantumPreprocessor,
    load_binary_dataset,
    split_dataset,
    subsample_partitions,
)


def get_partitions(seed: int = SEED, quick: bool = False):
    x, y, names = load_binary_dataset()
    partitions = split_dataset(x, y, seed=seed)
    partitions.feature_names = names
    limits = (24, 16, 20) if quick else (
        QUANTUM_TRAIN_LIMIT,
        QUANTUM_VALIDATION_LIMIT,
        QUANTUM_TEST_LIMIT,
    )
    return subsample_partitions(partitions, seed, limits)


def prepare_features(partitions, encoding: str, n_qubits: int = N_QUBITS):
    processor = QuantumPreprocessor(n_qubits, encoding)
    x_train = processor.fit_transform(partitions.x_train)
    return (
        processor,
        x_train,
        processor.transform(partitions.x_validation),
        processor.transform(partitions.x_test),
    )


def library_versions():
    """Versions enregistrées avec chaque résultat pour faciliter sa reproduction."""
    import platform

    return {
        "python_version": platform.python_version(),
        "pennylane_version": version("PennyLane"),
        "scikit_learn_version": version("scikit-learn"),
        "numpy_version": version("numpy"),
    }

