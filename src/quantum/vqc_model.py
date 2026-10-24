"""Classifieur VQC entraînable avec l'optimiseur Adam de PennyLane."""

from time import perf_counter

import numpy as np
import pennylane as qml

from ..config import CircuitConfig, SEED, TRAINING_STEPS
from .circuit import build_qnode, estimated_gate_count


class VQCModel:
    """Classifieur binaire dont la sortie Pauli-Z est seuillée à zéro."""

    def __init__(
        self,
        n_qubits: int = 4,
        config: CircuitConfig | None = None,
        seed: int = SEED,
        steps: int = TRAINING_STEPS,
        learning_rate: float = 0.08,
    ):
        self.n_qubits = n_qubits
        self.config = config or CircuitConfig()
        self.seed = seed
        self.steps = steps
        self.learning_rate = learning_rate
        self.n_layers = self.config.n_layers
        self.n_parameters = 3 * n_qubits * self.n_layers
        self.gate_count = estimated_gate_count(
            n_qubits, self.n_layers, self.config.structure, self.config.encoding
        )
        self.qnode = build_qnode(n_qubits, self.n_layers, self.config)
        self.weights = None
        self.fit_seconds = 0.0
        self.predict_seconds = 0.0

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "VQCModel":
        if len(features) != len(labels) or len(labels) == 0:
            raise ValueError("features et labels doivent avoir la même longueur non nulle")
        expected_features = self.n_qubits if self.config.encoding == "angle" else 2**self.n_qubits
        if features.shape[1] != expected_features:
            raise ValueError(f"Cet encodage attend {expected_features} caractéristiques")
        unique_labels = set(np.unique(labels).tolist())
        if not unique_labels.issubset({0, 1}) or len(unique_labels) != 2:
            raise ValueError("Le VQC attend les deux classes binaires 0 et 1")

        rng = np.random.default_rng(self.seed)
        initial = 0.1 * rng.standard_normal((self.n_layers, self.n_qubits, 3))
        weights = qml.numpy.array(initial, requires_grad=True)
        x = qml.numpy.array(features, requires_grad=False)
        y = qml.numpy.array(1.0 - 2.0 * np.asarray(labels, dtype=float), requires_grad=False)
        optimizer = qml.AdamOptimizer(stepsize=self.learning_rate)

        def loss_fn(current_weights):
            predictions = qml.math.stack([self.qnode(row, current_weights) for row in x])
            return qml.math.mean((predictions - y) ** 2)

        start = perf_counter()
        for _ in range(self.steps):
            weights = optimizer.step(loss_fn, weights)
        self.fit_seconds = perf_counter() - start
        self.weights = weights
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Le modèle doit être entraîné avant predict")
        return np.asarray([float(self.qnode(row, self.weights)) for row in features])

    def predict(self, features: np.ndarray) -> np.ndarray:
        # Z=+1 mappe vers la classe 0, Z=-1 vers la classe 1.
        start = perf_counter()
        predictions = (self.decision_function(features) < 0.0).astype(int)
        self.predict_seconds = perf_counter() - start
        return predictions

    def set_evaluation_noise(self, probability: float) -> None:
        """Change le canal du QNode pour mesurer un modèle entraîné sous bruit donné."""
        from dataclasses import replace

        self.qnode = build_qnode(
            self.n_qubits,
            self.n_layers,
            replace(self.config, noise_probability=probability),
        )
