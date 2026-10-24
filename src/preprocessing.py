"""Chargement, partitionnement et prétraitement sans fuite d'information."""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

from .config import N_QUBITS, SEED, TEST_SIZE, VALIDATION_SIZE


@dataclass
class DataPartitions:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]


def load_binary_dataset() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Charge le dataset public Breast Cancer Wisconsin (Diagnostic) de sklearn."""
    data = load_breast_cancer()
    return data.data.astype(float), data.target.astype(int), tuple(data.feature_names)


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = SEED,
    test_size: float = TEST_SIZE,
    validation_size: float = VALIDATION_SIZE,
) -> DataPartitions:
    """Crée des partitions stratifiées et identiques pour les modèles comparés."""
    x_train_validation, x_test, y_train_validation, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    relative_validation = validation_size / (1.0 - test_size)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_validation,
        y_train_validation,
        test_size=relative_validation,
        random_state=seed,
        stratify=y_train_validation,
    )
    return DataPartitions(x_train, y_train, x_validation, y_validation, x_test, y_test, tuple())


class QuantumPreprocessor:
    """Scaler et PCA appris sur train seulement, adaptés à l'encodage choisi."""

    def __init__(self, n_qubits: int, encoding: str):
        if n_qubits < 1:
            raise ValueError("n_qubits doit être >= 1")
        if encoding not in {"angle", "amplitude"}:
            raise ValueError("Encodage pris en charge : angle ou amplitude")
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.n_features = n_qubits if encoding == "angle" else 2**n_qubits
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_features, random_state=SEED)
        self.angle_scaler = MinMaxScaler(feature_range=(0.0, 2.0 * np.pi))

    def fit(self, x: np.ndarray) -> "QuantumPreprocessor":
        if x.shape[1] < self.n_features:
            raise ValueError(
                f"L'encodage {self.encoding} avec {self.n_qubits} qubits demande "
                f"{self.n_features} variables, mais le jeu en contient {x.shape[1]}"
            )
        reduced = self.pca.fit_transform(self.scaler.fit_transform(x))
        if self.encoding == "angle":
            self.angle_scaler.fit(reduced)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        reduced = self.pca.transform(self.scaler.transform(x))
        if self.encoding == "angle":
            return self.angle_scaler.transform(reduced)
        return reduced

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def subsample_partitions(partitions: DataPartitions, seed: int, limits: tuple[int, int, int]):
    """Sous-échantillonne chaque partition de façon stratifiée pour les circuits."""
    def choose(x, y, limit):
        if len(y) <= limit:
            return x, y
        selected_x, _, selected_y, _ = train_test_split(
            x, y, train_size=limit, random_state=seed, stratify=y
        )
        return selected_x, selected_y

    train = choose(partitions.x_train, partitions.y_train, limits[0])
    validation = choose(partitions.x_validation, partitions.y_validation, limits[1])
    test = choose(partitions.x_test, partitions.y_test, limits[2])
    return DataPartitions(*train, *validation, *test, partitions.feature_names)

