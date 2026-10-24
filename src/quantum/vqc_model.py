"""Classifieur VQC minimal, prêt à être étendu avec un optimiseur entraînable."""

import pennylane as qml

from .circuit import build_qnode


class VQCModel:
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, noise: bool = False, seed: int = 42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.noise = noise
        self.seed = seed
        self.qnode = build_qnode(n_qubits, n_layers, noise=noise)
        self.weights = qml.numpy.array(
            0.01 * qml.numpy.random.randn(n_layers, n_qubits, 3), requires_grad=True
        )

    def decision_function(self, features):
        return self.qnode(features, self.weights)

    def predict(self, features):
        return [1 if float(self.decision_function(row)) >= 0 else 0 for row in features]
