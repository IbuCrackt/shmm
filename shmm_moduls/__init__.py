"""rescrf - Residual-CRF-Bibliothek (siehe Bachelorarbeit, Kapitel 2-4).

Öffentliche Schnittstelle: Modellaufbau (model), Datengenerierung
(data, hierarchical_data, util) und Trainingslogik (training).
"""
from .model import get_model
from .data import load_dataset, create_data, create_random_data
from .hierarchical_data import create_hierarchical_data

__all__ = [
    "get_model",
    "load_dataset",
    "create_data",
    "create_random_data",
    "create_hierarchical_data",
]
