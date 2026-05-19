from __future__ import annotations

import torch

from ep08.deepinv_wrapper import MultiFramePhysics
from ep08.forward import ForwardOperator


def test_multiframe_physics_matches_ep08_forward_operator() -> None:
    x_hr = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 10.0
    shifts = torch.tensor([[0.0, 0.0], [0.25, 0.5], [0.5, 0.25]], dtype=torch.float32)
    op = ForwardOperator(hr_shape=(8, 8), lr_shape=(4, 4), shifts=shifts, psf_sigma=0.0)
    physics = MultiFramePhysics(op, frame_indices=[0, 2])

    y = physics.A(x_hr[None, None])

    expected = torch.stack([op(x_hr, 0), op(x_hr, 2)], dim=0).unsqueeze(0)
    assert y.shape == (1, 2, 4, 4)
    assert torch.allclose(y, expected)


def test_multiframe_physics_adjoint_shape_and_finiteness() -> None:
    shifts = torch.tensor([[0.0, 0.0], [0.25, 0.5]], dtype=torch.float32)
    op = ForwardOperator(hr_shape=(8, 8), lr_shape=(4, 4), shifts=shifts, psf_sigma=0.0)
    physics = MultiFramePhysics(op)
    y = torch.ones((1, 2, 4, 4), dtype=torch.float32)

    x_back = physics.A_adjoint(y)

    assert x_back.shape == (1, 1, 8, 8)
    assert torch.isfinite(x_back).all()
