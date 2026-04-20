"""Unit tests for TorchBacterium rod geometry and division mechanics."""

from __future__ import annotations

import math

import torch

from pycellmodeller.biophysics import TorchBacterium


def test_direction_normalization_with_zero_vector_fallback() -> None:
    model = TorchBacterium()
    directions = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float32)

    normalized = model.normalize_directions(directions)

    assert torch.allclose(normalized[0], torch.tensor([0.6, 0.8]))
    assert torch.allclose(normalized[1], torch.tensor([1.0, 0.0]))


def test_spherocylinder_volume_formula() -> None:
    model = TorchBacterium()
    lengths = torch.tensor([2.0], dtype=torch.float64)
    radii = torch.tensor([0.5], dtype=torch.float64)

    volume = float(model.compute_volume(lengths, radii)[0].item())
    expected = (math.pi * (0.5**2) * 2.0) + ((4.0 / 3.0) * math.pi * (0.5**3))

    assert volume == expected


def test_exact_mass_conservation_with_partition_noise() -> None:
    model = TorchBacterium(partition_noise_std=0.05)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(8)

    lengths = torch.tensor([4.0, 5.0], dtype=torch.float32)
    radii = torch.tensor([0.5, 0.4], dtype=torch.float32)
    divide_mask = torch.tensor([True, True])

    daughter_a, daughter_b = model.divide(lengths, radii, divide_mask, generator=generator)

    parent_volume = model.compute_volume(lengths, radii)
    daughters_volume = model.compute_volume(daughter_a, radii) + model.compute_volume(daughter_b, radii)
    assert torch.allclose(parent_volume, daughters_volume, atol=1e-6, rtol=1e-6)


def test_non_overlapping_daughter_placement() -> None:
    model = TorchBacterium(contact_gap=1e-3)
    parent_position = torch.tensor([2.0, -1.0], dtype=torch.float32)
    parent_direction = torch.tensor([2.0, 0.0], dtype=torch.float32)

    pos_a, pos_b = model.place_daughters(
        parent_position=parent_position,
        parent_direction=parent_direction,
        daughter_a_length=1.6,
        daughter_b_length=1.4,
        parent_radius=0.5,
    )

    center_distance = float(torch.linalg.vector_norm(pos_b - pos_a).item())
    minimum_distance = (1.6 + 1.4) * 0.5 + (2.0 * 0.5) + model.contact_gap
    assert center_distance >= minimum_distance - 1e-6


def test_longitudinal_division_requires_sufficient_parent_length() -> None:
    model = TorchBacterium()
    lengths = torch.tensor([0.5, 0.8], dtype=torch.float32)
    radii = torch.tensor([0.5, 0.5], dtype=torch.float32)

    supported = model.supports_longitudinal_division(lengths, radii)

    assert torch.equal(supported, torch.tensor([False, True]))


def test_divide_produces_non_negative_daughter_lengths_with_noise() -> None:
    model = TorchBacterium(partition_noise_std=0.2)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(3)
    lengths = torch.tensor([0.8], dtype=torch.float32)
    radii = torch.tensor([0.5], dtype=torch.float32)
    divide_mask = torch.tensor([True])

    daughter_a, daughter_b = model.divide(lengths, radii, divide_mask, generator=generator)

    assert float(daughter_a[0].item()) >= 0.0
    assert float(daughter_b[0].item()) >= 0.0
