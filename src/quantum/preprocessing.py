"""Prétraitement des données vers des vecteurs compatibles avec un circuit."""

import pennylane as qml
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class AnglePreprocessor:
    """Réduction à n_qubits dimensions et mise à l'échelle pour l'encodage."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.pca = PCA(n_components=n_qubits, random_state=42)
        self.scaler = MinMaxScaler(feature_range=(0.0, 2.0 * 3.141592653589793))

    def fit(self, x):
        reduced = self.pca.fit_transform(x)
        self.scaler.fit(reduced)
        return self

    def transform(self, x):
        return self.scaler.transform(self.pca.transform(x))

    def fit_transform(self, x):
        return self.fit(x).transform(x)


def angle_embedding(features, wires):
    """Encodage angle; features doivent déjà être ramenées dans [0, 2π]."""
    qml.AngleEmbedding(features, wires=wires, rotation="Y")
