"""Ansatz variationnel réutilisable pour les expériences de classification."""

import pennylane as qml


def variational_layer(weights, wires):
    for wire, angles in zip(wires, weights):
        qml.Rot(*angles, wires=wire)
    for left, right in zip(wires[:-1], wires[1:]):
        qml.CNOT(wires=[left, right])


def build_qnode(n_qubits: int, n_layers: int, noise: bool = False, shots=None):
    """Construit un QNode; noise active des canaux dépolarisants après l'ansatz."""
    device = qml.device("default.mixed" if noise else "default.qubit", wires=n_qubits, shots=shots)

    @qml.qnode(device, interface="autograd", diff_method="backprop")
    def circuit(features, weights):
        qml.AngleEmbedding(features, wires=range(n_qubits), rotation="Y")
        for layer in weights:
            variational_layer(layer, list(range(n_qubits)))
            if noise:
                for wire in range(n_qubits):
                    qml.DepolarizingChannel(0.01, wires=wire)
        return qml.expval(qml.PauliZ(0))

    return circuit
