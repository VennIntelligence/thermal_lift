from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

try:
    from deepinv.physics import Physics
except ImportError as exc:  # pragma: no cover - exercised only when optional dependency is absent.
    raise ImportError("deepinv is required for ep08.deepinv_wrapper") from exc


class MultiFramePhysics(Physics):
    """DeepInverse Physics wrapper for EP08 multi-frame microscan observations.

    DeepInverse DIP emits a batch of HR images as ``(B, 1, H_hr, W_hr)``. EP08's
    forward model predicts one LR frame at a time from a 2D HR image. This
    wrapper bridges the two conventions and returns stacked observations with
    shape ``(B, K, H_lr, W_lr)`` for the selected frame indices.
    """

    def __init__(
        self,
        forward_operator: torch.nn.Module,
        frame_indices: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.forward_operator = forward_operator
        if frame_indices is None:
            if not hasattr(forward_operator, "shifts"):
                raise ValueError("frame_indices must be provided when forward_operator has no shifts buffer")
            frame_indices = range(int(forward_operator.shifts.shape[0]))
        indices = torch.as_tensor(list(frame_indices), dtype=torch.long)
        if indices.ndim != 1 or indices.numel() == 0:
            raise ValueError("frame_indices must be a non-empty 1D sequence")
        self.register_buffer("frame_indices", indices)

    @property
    def selected_indices(self) -> list[int]:
        return [int(v) for v in self.frame_indices.detach().cpu().tolist()]

    def _normalize_hr_batch(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[1] != 1:
                raise ValueError(f"expected one HR channel, got shape {tuple(x.shape)}")
            return x[:, 0]
        if x.ndim == 3:
            return x
        if x.ndim == 2:
            return x.unsqueeze(0)
        raise ValueError(f"expected HR image as HxW, BxHxW, or Bx1xHxW, got {tuple(x.shape)}")

    def A(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        batch = self._normalize_hr_batch(x)
        outputs: list[torch.Tensor] = []
        indices = self.selected_indices
        for image in batch:
            frames = [self.forward_operator(image, index) for index in indices]
            outputs.append(torch.stack(frames, dim=0))
        return torch.stack(outputs, dim=0)

    def A_adjoint(self, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        if y.ndim == 3:
            y_batch = y.unsqueeze(0)
        elif y.ndim == 4:
            y_batch = y
        else:
            raise ValueError(f"expected LR observations as KxHxW or BxKxHxW, got {tuple(y.shape)}")
        if y_batch.shape[1] != self.frame_indices.numel():
            raise ValueError(
                f"expected {int(self.frame_indices.numel())} stacked frames, got {int(y_batch.shape[1])}"
            )

        images: list[torch.Tensor] = []
        indices = self.selected_indices
        for sample in y_batch:
            accum = None
            for frame, index in zip(sample, indices, strict=True):
                back = self.forward_operator.adjoint(frame, index)
                accum = back if accum is None else accum + back
            images.append(accum / float(len(indices)))
        return torch.stack(images, dim=0).unsqueeze(1)


__all__ = ["MultiFramePhysics"]
