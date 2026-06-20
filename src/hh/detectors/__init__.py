"""Detectors. Importing this package registers the built-ins."""

from .base import Detector, available_detectors, load_detector, register

__all__ = ["Detector", "available_detectors", "load_detector", "register"]
