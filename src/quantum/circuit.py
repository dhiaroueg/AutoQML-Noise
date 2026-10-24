"""Ansatz variationnel et mesure pour les classifieurs VQC."""

from collections.abc import Sequence

import pennylane as qml

from ..config import CircuitConfig
from .noise_model import apply_depolarizing_noise


def build_qnode(n_qubits: int, n_layers: int, config: CircuitConfig, shots=None):
    """Construit un QNode idéal ou bruité avec un ansatz linéaire/en anneau."""
    if n_qubits < 2:
        raise ValueError("Le circuit demande au moins deux qubits")
    if n_layers < 1:
        raise ValueError("n_layers doit être >= 1")
    device_name = "default.mixed" if config.noise_probability > 0 else "default.qubit"
    device = qml.device(device_name, wires=n_qubits, shots=shots)
    diff_method = "parameter-shift" if config.noise_probability > 0 else "backprop"

    @qml.qnode(device, interface="autograd", diff_method=diff_method)
    def circuit(features, weights):
        wires = tuple(range(n_qubits))
        if config.encoding == "angle":
            qml.AngleEmbedding(features, wires=wires, rotation="Y")
        else:
            qml.AmplitudeEmbedding(features, wires=wires, normalize=True, pad_with=0.0)

        for layer in weights:
            for wire, angles in zip(wires, layer):
                qml.Rot(*angles, wires=wire)
            pairs = list(zip(wires[:-1], wires[1:]))
            if config.structure == "ring":
                pairs.append((wires[-1], wires[0]))
            for left, right in pairs:
                qml.CNOT(wires=[left, right])
            if config.noise_probability > 0:
                apply_depolarizing_noise(config.noise_probability, wires)
        return qml.expval(qml.PauliZ(0))

    return circuit


def estimated_gate_count(n_qubits: int, n_layers: int, structure: str, encoding: str) -> int:
    """Compte les portes explicites de l'ansatz et de l'encodage angle.

    L'état de base amplitude est compté comme une opération de préparation; sa
    décomposition en portes dépend du compilateur et n'est pas estimée ici.
    """
    embedding_gates = n_qubits if encoding == "angle" else 1
    entanglers_per_layer = n_qubits if structure == "ring" else n_qubits - 1
    return embedding_gates + n_layers * (3 * n_qubits + entanglers_per_layer)
