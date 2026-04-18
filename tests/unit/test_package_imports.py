"""Unit tests for package and public API imports."""

from __future__ import annotations

import importlib

from pycellmodeller.api.simulation import Simulation


def test_package_import() -> None:
    """Top-level package should import cleanly."""
    package = importlib.import_module("pycellmodeller")
    assert package.__title__ == "pycellmodeller"


def test_public_api_simulation_import_path() -> None:
    """Public API Simulation import path should be stable."""
    assert Simulation.__name__ == "Simulation"
