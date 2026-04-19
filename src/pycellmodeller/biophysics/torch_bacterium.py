"""v1 bacterium growth and division mechanics implemented with torch tensors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class TorchBacterium:
    """Simple rod-shaped bacterium growth/division model for v1 simulations.

    The model tracks only scalar rod length dynamics and symmetric division.
    """

    growth_rate: float = 0.02
    division_length: float = 4.0
    min_length: float = 0.1
    partition_noise_std: float = 0.0
    division_displacement: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        if self.growth_rate < 0.0:
            msg = "growth_rate must be non-negative"
            raise ValueError(msg)
        if self.division_length <= 0.0:
            msg = "division_length must be strictly positive"
            raise ValueError(msg)
        if self.min_length <= 0.0:
            msg = "min_length must be strictly positive"
            raise ValueError(msg)
        if self.min_length >= self.division_length:
            msg = "min_length must be smaller than division_length"
            raise ValueError(msg)
        if self.partition_noise_std < 0.0:
            msg = "partition_noise_std must be non-negative"
            raise ValueError(msg)
        if self.division_displacement < 0.0:
            msg = "division_displacement must be non-negative"
            raise ValueError(msg)
        if self.seed < 0:
            msg = "seed must be non-negative"
            raise ValueError(msg)

    def grow(self, lengths: torch.Tensor, dt: float) -> torch.Tensor:
        """Return updated lengths after one growth step.

        Uses a simple exponential-like update: L(t+dt) = L(t) * (1 + growth_rate * dt).
        """
        if dt <= 0.0:
            msg = "dt must be strictly positive"
            raise ValueError(msg)
        if not lengths.dtype.is_floating_point:
            msg = "lengths tensor must use a floating point dtype"
            raise TypeError(msg)

        growth_factor = 1.0 + (self.growth_rate * dt)
        grown = lengths * growth_factor
        return torch.clamp(grown, min=self.min_length)

    def should_divide(self, lengths: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask indicating which cells are ready to divide."""
        return lengths >= self.division_length

    def compute_volume(self, lengths: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        """Compute rod volume using a simple cylinder approximation."""
        if lengths.shape != radii.shape:
            msg = "lengths and radii must have the same shape"
            raise ValueError(msg)
        if not lengths.dtype.is_floating_point or not radii.dtype.is_floating_point:
            msg = "lengths and radii must use floating point dtypes"
            raise TypeError(msg)
        return math.pi * torch.square(radii) * torch.clamp(lengths, min=self.min_length)

    def update_cell_growth(self, cell: dict[str, object], dt: float) -> None:
        """Update per-cell length and volume in-place using per-cell growth_rate."""
        if dt <= 0.0:
            msg = "dt must be strictly positive"
            raise ValueError(msg)

        length = float(cell.get("length", 1.0))
        growth_rate = float(cell.get("growth_rate", self.growth_rate))
        if growth_rate < 0.0:
            growth_rate = 0.0
        grown = max(self.min_length, length * (1.0 + (growth_rate * dt)))
        cell["length"] = grown

        radius = float(cell.get("radius", 0.5))
        cell["volume"] = math.pi * radius * radius * grown

    def division_ready(self, cell: dict[str, object]) -> bool:
        """Return True if a cell should divide this step."""
        divide_flag = bool(cell.get("divide", False))
        volume = float(cell.get("volume", 0.0))
        target_volume = float(cell.get("target_volume", 1.0))
        return divide_flag or (volume > target_volume)

    def divide_cells(self, cells: list[dict[str, object]], dt: float) -> list[dict[str, object]]:
        """Advance one step for dict-based cells and apply deterministic division.

        This routine is CPU-oriented and intentionally simple. It updates growth,
        recomputes volume, evaluates division, and emits daughter entries with:
        - incremented generation,
        - copied biological fields,
        - deterministic spatial offsets along the cell direction.
        """
        updated: list[dict[str, object]] = []
        for cell in cells:
            parent = dict(cell)
            self.update_cell_growth(parent, dt)

            if not self.division_ready(parent):
                parent["divide"] = False
                updated.append(parent)
                continue

            parent_pos = torch.as_tensor(parent.get("position", (0.0, 0.0)), dtype=torch.float32).clone()
            direction = torch.as_tensor(parent.get("direction", (1.0, 0.0)), dtype=torch.float32).clone()
            if direction.numel() != parent_pos.numel():
                direction = torch.zeros_like(parent_pos)
                direction[0] = 1.0

            norm = torch.linalg.vector_norm(direction)
            if float(norm.item()) == 0.0:
                direction.zero_()
                direction[0] = 1.0
            else:
                direction /= norm

            offset = direction * self.division_displacement
            daughter_length = max(self.min_length, float(parent["length"]) * 0.5)
            daughter_generation = int(parent.get("generation", 0)) + 1

            daughter_a = dict(parent)
            daughter_b = dict(parent)
            daughter_a["length"] = daughter_length
            daughter_b["length"] = daughter_length
            daughter_a["generation"] = daughter_generation
            daughter_b["generation"] = daughter_generation
            daughter_a["divide"] = False
            daughter_b["divide"] = False
            daughter_a["position"] = parent_pos - offset
            daughter_b["position"] = parent_pos + offset

            radius = float(parent.get("radius", 0.5))
            daughter_volume = math.pi * radius * radius * daughter_length
            daughter_a["volume"] = daughter_volume
            daughter_b["volume"] = daughter_volume

            updated.append(daughter_a)
            updated.append(daughter_b)

        return updated

    def divide(
        self,
        lengths: torch.Tensor,
        divide_mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute daughter lengths for cells selected by ``divide_mask``.

        Returns two tensors ``(daughter_a, daughter_b)`` with the same shape as
        ``lengths``. Entries for non-dividing cells are copied unchanged.
        """
        if lengths.shape != divide_mask.shape:
            msg = "lengths and divide_mask must have the same shape"
            raise ValueError(msg)

        daughter_a = lengths.clone()
        daughter_b = lengths.clone()

        if not torch.any(divide_mask):
            return daughter_a, daughter_b

        half = lengths[divide_mask] * 0.5
        if self.partition_noise_std > 0.0:
            noise = torch.randn(
                half.shape,
                dtype=half.dtype,
                device=half.device,
                generator=generator,
            )
            noise = noise * self.partition_noise_std
            frac = torch.clamp(0.5 + noise, min=0.1, max=0.9)
            a = lengths[divide_mask] * frac
            b = lengths[divide_mask] - a
        else:
            a = half
            b = half

        daughter_a[divide_mask] = torch.clamp(a, min=self.min_length)
        daughter_b[divide_mask] = torch.clamp(b, min=self.min_length)
        return daughter_a, daughter_b
