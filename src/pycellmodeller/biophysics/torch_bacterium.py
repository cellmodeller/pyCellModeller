"""v1 bacterium growth, division, and contact mechanics implemented with torch tensors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class TorchBacterium:
    """Simple 2D rod-shaped bacterium growth/division/contact model for v1 simulations."""

    growth_rate: float = 0.02
    division_length: float = 4.0
    min_length: float = 0.1
    partition_noise_std: float = 0.0
    contact_iterations: int = 6
    contact_gap: float = 1e-3
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
        if self.contact_iterations < 0:
            msg = "contact_iterations must be non-negative"
            raise ValueError(msg)
        if self.contact_gap < 0.0:
            msg = "contact_gap must be non-negative"
            raise ValueError(msg)
        if self.seed < 0:
            msg = "seed must be non-negative"
            raise ValueError(msg)

    def normalize_directions(self, directions: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Return normalized 2D directions with +x fallback for degenerate vectors."""
        if directions.ndim != 2 or directions.shape[1] != 2:
            msg = "directions must have shape (n, 2)"
            raise ValueError(msg)
        out = directions.clone()
        norms = torch.linalg.vector_norm(out, dim=1, keepdim=True)
        invalid = norms.squeeze(1) <= eps
        safe_norms = torch.where(invalid.unsqueeze(1), torch.ones_like(norms), norms)
        out = out / safe_norms
        if torch.any(invalid):
            out[invalid] = torch.tensor([1.0, 0.0], dtype=out.dtype, device=out.device)
        return out

    def grow(self, lengths: torch.Tensor, dt: float) -> torch.Tensor:
        """Return updated lengths after one growth step."""
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

    def supports_longitudinal_division(self, lengths: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        """Return a mask of cells that can divide into two non-negative cylindrical daughters."""
        if lengths.shape != radii.shape:
            msg = "lengths and radii must have the same shape"
            raise ValueError(msg)
        minimum_parent_length = (4.0 / 3.0) * radii
        return lengths >= minimum_parent_length

    def compute_volume(self, lengths: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        """Compute 2D spherocylinder-equivalent 3D volume."""
        if lengths.shape != radii.shape:
            msg = "lengths and radii must have the same shape"
            raise ValueError(msg)
        if not lengths.dtype.is_floating_point or not radii.dtype.is_floating_point:
            msg = "lengths and radii must use floating point dtypes"
            raise TypeError(msg)
        cylinder = math.pi * torch.square(radii) * lengths
        caps = (4.0 / 3.0) * math.pi * torch.pow(radii, 3)
        return cylinder + caps

    def _length_from_volume(self, volumes: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        cap_volume = (4.0 / 3.0) * math.pi * torch.pow(radii, 3)
        cylinder_factor = math.pi * torch.square(radii)
        return (volumes - cap_volume) / cylinder_factor

    def divide(
        self,
        lengths: torch.Tensor,
        radii: torch.Tensor,
        divide_mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute daughter cylindrical lengths with exact per-parent volume conservation."""
        if lengths.shape != divide_mask.shape or lengths.shape != radii.shape:
            msg = "lengths, radii, and divide_mask must have the same shape"
            raise ValueError(msg)

        daughter_a = lengths.clone()
        daughter_b = lengths.clone()

        if not torch.any(divide_mask):
            return daughter_a, daughter_b

        parent_lengths = lengths[divide_mask]
        parent_radii = radii[divide_mask]
        parent_volume = self.compute_volume(parent_lengths, parent_radii)

        if self.partition_noise_std > 0.0:
            noise = torch.randn(
                parent_volume.shape,
                dtype=parent_volume.dtype,
                device=parent_volume.device,
                generator=generator,
            )
            frac = torch.clamp(0.5 + (noise * self.partition_noise_std), min=0.25, max=0.75)
        else:
            frac = torch.full_like(parent_volume, 0.5)

        daughter_a_volume = parent_volume * frac
        daughter_b_volume = parent_volume - daughter_a_volume

        cap_volume = (4.0 / 3.0) * math.pi * torch.pow(parent_radii, 3)
        daughter_a_volume = torch.clamp(daughter_a_volume, min=cap_volume)
        daughter_b_volume = torch.clamp(daughter_b_volume, min=cap_volume)

        a = self._length_from_volume(daughter_a_volume, parent_radii)
        b = self._length_from_volume(daughter_b_volume, parent_radii)

        daughter_a[divide_mask] = a
        daughter_b[divide_mask] = b
        return daughter_a, daughter_b

    def placement_offsets(
        self,
        lengths_a: torch.Tensor,
        radii_a: torch.Tensor,
        lengths_b: torch.Tensor,
        radii_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Geometry-based center offsets that guarantee a small inter-daughter gap."""
        offset_a = (0.5 * lengths_a) + radii_a + (0.5 * self.contact_gap)
        offset_b = (0.5 * lengths_b) + radii_b + (0.5 * self.contact_gap)
        return offset_a, offset_b

    def place_daughters(
        self,
        parent_position: torch.Tensor,
        parent_direction: torch.Tensor,
        daughter_a_length: float,
        daughter_b_length: float,
        parent_radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Place daughters along parent orientation with non-overlapping geometry offsets."""
        direction = self.normalize_directions(parent_direction.reshape(1, 2)).reshape(2)
        dtype = parent_position.dtype
        device = parent_position.device
        length_a_t = torch.tensor([daughter_a_length], dtype=dtype, device=device)
        length_b_t = torch.tensor([daughter_b_length], dtype=dtype, device=device)
        radius_t = torch.tensor([parent_radius], dtype=dtype, device=device)
        offset_a, offset_b = self.placement_offsets(length_a_t, radius_t, length_b_t, radius_t)

        pos_a = parent_position - (offset_a[0] * direction)
        pos_b = parent_position + (offset_b[0] * direction)
        return pos_a, pos_b

    def _closest_points_on_segments(
        self,
        p1: torch.Tensor,
        q1: torch.Tensor,
        p2: torch.Tensor,
        q2: torch.Tensor,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d1 = q1 - p1
        d2 = q2 - p2
        r = p1 - p2
        a = torch.dot(d1, d1)
        e = torch.dot(d2, d2)
        f = torch.dot(d2, r)

        if float(a.item()) <= eps and float(e.item()) <= eps:
            return p1, p2
        if float(a.item()) <= eps:
            s = torch.tensor(0.0, dtype=p1.dtype, device=p1.device)
            t = torch.clamp(f / e, 0.0, 1.0)
        else:
            c = torch.dot(d1, r)
            if float(e.item()) <= eps:
                t = torch.tensor(0.0, dtype=p1.dtype, device=p1.device)
                s = torch.clamp(-c / a, 0.0, 1.0)
            else:
                b = torch.dot(d1, d2)
                denom = a * e - b * b
                if float(denom.item()) > eps:
                    s = torch.clamp((b * f - c * e) / denom, 0.0, 1.0)
                else:
                    s = torch.tensor(0.0, dtype=p1.dtype, device=p1.device)
                t = (b * s + f) / e
                if float(t.item()) < 0.0:
                    t = torch.tensor(0.0, dtype=p1.dtype, device=p1.device)
                    s = torch.clamp(-c / a, 0.0, 1.0)
                elif float(t.item()) > 1.0:
                    t = torch.tensor(1.0, dtype=p1.dtype, device=p1.device)
                    s = torch.clamp((b - c) / a, 0.0, 1.0)

        c1 = p1 + d1 * s
        c2 = p2 + d2 * t
        return c1, c2

    def resolve_contacts(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
        lengths: torch.Tensor,
        radii: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve capsule overlaps by overdamped positional relaxation."""
        n_cells = int(positions.shape[0])
        if n_cells <= 1 or self.contact_iterations == 0:
            return positions

        dirs = self.normalize_directions(directions)
        half = 0.5 * lengths
        out = positions.clone()

        for _ in range(self.contact_iterations):
            for i in range(n_cells):
                ui = dirs[i]
                pi0 = out[i] - (half[i] * ui)
                pi1 = out[i] + (half[i] * ui)
                for j in range(i + 1, n_cells):
                    uj = dirs[j]
                    pj0 = out[j] - (half[j] * uj)
                    pj1 = out[j] + (half[j] * uj)
                    ci, cj = self._closest_points_on_segments(pi0, pi1, pj0, pj1)
                    delta = ci - cj
                    dist = torch.linalg.vector_norm(delta)
                    min_sep = radii[i] + radii[j] + self.contact_gap
                    overlap = min_sep - dist
                    if float(overlap.item()) <= 0.0:
                        continue
                    if float(dist.item()) > 1e-12:
                        normal = delta / dist
                    else:
                        center_delta = out[i] - out[j]
                        center_dist = torch.linalg.vector_norm(center_delta)
                        if float(center_dist.item()) > 1e-12:
                            normal = center_delta / center_dist
                        else:
                            normal = torch.tensor([1.0, 0.0], dtype=out.dtype, device=out.device)
                    correction = 0.5 * overlap * normal
                    out[i] = out[i] + correction
                    out[j] = out[j] - correction

        return out
