"""Real-robot interfaces for the reduced Adaptive-HRC laboratory task.

The package deliberately keeps Stretch SDK imports out of the core project
runtime.  ``app`` runs the learner and operator UI in the normal project venv;
``bridge`` runs as a second local process in the Stretch hardware environment.
"""

from .config import LabConfig, load_lab_config
from .domain import PhysicalTaskDomain, PhysicalObservation
from .hardware import DryRunExecutor, HttpStretchExecutor
from .session import LiveHrcSession

__all__ = [
    "DryRunExecutor",
    "HttpStretchExecutor",
    "LabConfig",
    "LiveHrcSession",
    "PhysicalObservation",
    "PhysicalTaskDomain",
    "load_lab_config",
]
