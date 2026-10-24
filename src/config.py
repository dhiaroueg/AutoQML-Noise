"""Paramètres communs du protocole expérimental."""

from dataclasses import dataclass


SEED = 42
DATASET_NAME = "breast_cancer_wisconsin_diagnostic"
N_QUBITS = 4
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25  # fraction du train initial
QUANTUM_TRAIN_LIMIT = 96
QUANTUM_VALIDATION_LIMIT = 48
QUANTUM_TEST_LIMIT = 64
TRAINING_STEPS = 12
NOISE_LEVELS = (0.0, 0.01, 0.03)


@dataclass(frozen=True)
class CircuitConfig:
    n_layers: int = 2
    encoding: str = "angle"
    structure: str = "linear"
    noise_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError("n_layers doit être >= 1")
        if self.encoding not in {"angle", "amplitude"}:
            raise ValueError("encoding doit être 'angle' ou 'amplitude'")
        if self.structure not in {"linear", "ring"}:
            raise ValueError("structure doit être 'linear' ou 'ring'")
        if not 0.0 <= self.noise_probability <= 1.0:
            raise ValueError("noise_probability doit être entre 0 et 1")

