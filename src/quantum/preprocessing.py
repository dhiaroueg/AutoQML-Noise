"""Compatibilité pour l'ancien chemin de prétraitement quantique."""

from ..preprocessing import QuantumPreprocessor


class AnglePreprocessor(QuantumPreprocessor):
    def __init__(self, n_qubits: int):
        super().__init__(n_qubits=n_qubits, encoding="angle")
