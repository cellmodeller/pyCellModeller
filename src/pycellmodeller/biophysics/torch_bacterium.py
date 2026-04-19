"""v1 bacterium growth and division mechanics implemented with torch tensors."""

from __future__ import annotations

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
