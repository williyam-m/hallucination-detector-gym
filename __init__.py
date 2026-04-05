"""
Hallucination Detector Gym — Root Package.

OpenEnv scaffold convention: the project root directory acts as the top-level
Python package. This file re-exports the core library so that OpenEnv tooling
(push, build, client) can discover models and environment classes.
"""

from hallucination_detector_gym.models import (
    HallucinationAction,
    HallucinationObservation,
    HallucinationState,
)

try:
    from client import HallucinationDetectorEnv
except (ImportError, ModuleNotFoundError):
    HallucinationDetectorEnv = None  # type: ignore[assignment,misc]

__all__ = [
    "HallucinationAction",
    "HallucinationObservation",
    "HallucinationState",
    "HallucinationDetectorEnv",
]
