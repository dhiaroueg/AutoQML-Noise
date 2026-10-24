"""Canaux de bruit configurables pour les circuits PennyLane."""

import pennylane as qml


def apply_depolarizing_noise(probability: float, wires) -> None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability doit être comprise entre 0 et 1")
    for wire in wires:
        qml.DepolarizingChannel(probability, wires=wire)
