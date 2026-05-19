from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from ep08.utils import (
    as_frame_tensor,
    coordinate_grid,
    prepare_highpass_observations,
    render_model_image,
    resolve_device,
    set_seed,
    warmup_cosine_lr_lambda,
)


@dataclass(slots=True)
class TrainConfig:
    max_iter: int = 200
    lr: float = 1.0e-4
    warmup_steps: int = 20
    min_lr_factor: float = 0.05
    batch_k: int = 8
    grad_clip_norm: float | None = 1.0
    val_interval: int = 25
    early_stop_patience: int = 50
    early_stop_min_delta: float = 1.0e-6
    early_stop_min_steps: int = 0
    seed: int = 42


@dataclass(slots=True)
class TrainResult:
    image: torch.Tensor
    history: list[dict[str, float]]
    best_loss: float
    best_step: int


class INRTrainer:
    """Reusable training loop for EP08 highpass-domain SR models."""

    def __init__(
        self,
        model: nn.Module,
        observations: torch.Tensor | np.ndarray,
        shifts: torch.Tensor | np.ndarray,
        *,
        hr_shape: tuple[int, int] | None = None,
        forward_operator: nn.Module | None = None,
        psf_sigma_lr_px: float = 1.0,
        highpass_sigma_bg_lr_px: float = 5.0,
        scale: int = 2,
        device: str | torch.device = "cpu",
        config: TrainConfig | None = None,
        observations_are_highpass: bool = True,
        train_indices: torch.Tensor | np.ndarray | list[int] | None = None,
        val_indices: torch.Tensor | np.ndarray | list[int] | None = None,
        coord_aspect_mode: str = "preserve",
    ) -> None:
        self.device = resolve_device(device)
        self.config = config or TrainConfig()
        set_seed(self.config.seed)
        self.model = model.to(self.device)
        self.observations = as_frame_tensor(observations, device=self.device)
        if not observations_are_highpass:
            self.observations = prepare_highpass_observations(
                self.observations,
                sigma_bg_lr_px=highpass_sigma_bg_lr_px,
            )
        self.shifts = torch.as_tensor(shifts, dtype=torch.float32, device=self.device)
        if self.shifts.ndim != 2 or self.shifts.shape[1] != 2:
            raise ValueError(f"expected shifts as Nx2, got {tuple(self.shifts.shape)}")
        if self.shifts.shape[0] != self.observations.shape[0]:
            raise ValueError("shifts and observations must have the same frame count")

        lr_h, lr_w = int(self.observations.shape[-2]), int(self.observations.shape[-1])
        self.hr_shape = hr_shape or (lr_h * scale, lr_w * scale)
        self.psf_sigma_lr_px = psf_sigma_lr_px
        self.highpass_sigma_bg_lr_px = highpass_sigma_bg_lr_px
        self.scale = scale
        self.coord_aspect_mode = str(coord_aspect_mode)
        self.forward_operator = forward_operator or self._build_forward_operator()
        self.forward_operator = self.forward_operator.to(self.device)
        self.train_indices = self._normalize_indices(train_indices, default_all=True)
        self.val_indices = self._normalize_indices(val_indices, default_all=False)
        if bool(getattr(self.model, "expects_coords", True)):
            self._coords_grid = coordinate_grid(
                *self.hr_shape,
                device=self.device,
                flatten=True,
                aspect_mode=self.coord_aspect_mode,
            )
        else:
            self._coords_grid = None

    def _normalize_indices(
        self,
        indices: torch.Tensor | np.ndarray | list[int] | None,
        *,
        default_all: bool,
    ) -> torch.Tensor:
        count = int(self.observations.shape[0])
        if indices is None:
            values = torch.arange(count, device=self.device) if default_all else torch.empty(0, dtype=torch.long, device=self.device)
        else:
            values = torch.as_tensor(indices, dtype=torch.long, device=self.device).reshape(-1)
        if values.numel() == 0 and default_all:
            raise ValueError("train_indices must not be empty")
        if values.numel() and (int(values.min()) < 0 or int(values.max()) >= count):
            raise ValueError("frame indices out of range")
        return values

    def _render_image(self) -> torch.Tensor:
        if self._coords_grid is not None:
            pred = self.model(self._coords_grid).reshape(self.hr_shape[0], self.hr_shape[1], -1)
            if pred.shape[-1] != 1:
                raise ValueError(f"expected one output channel, got {pred.shape[-1]}")
            return pred[..., 0]
        return render_model_image(self.model, self.hr_shape, device=self.device)

    def _sample_train_indices(self, generator: torch.Generator) -> torch.Tensor:
        count = int(self.train_indices.numel())
        k = min(int(self.config.batch_k), count)
        order = torch.randperm(count, generator=generator, device=self.device)[:k]
        return self.train_indices.index_select(0, order)

    def _build_forward_operator(self) -> nn.Module:
        try:
            from ep08.forward import ForwardOperator
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ep08.forward.ForwardOperator is required unless forward_operator is injected"
            ) from exc
        return ForwardOperator(
            hr_shape=self.hr_shape,
            lr_shape=tuple(self.observations.shape[-2:]),
            shifts=self.shifts,
            psf_sigma=self.psf_sigma_lr_px,
            scale=self.scale,
        )

    def fit(self) -> TrainResult:
        cfg = self.config
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: warmup_cosine_lr_lambda(
                step,
                warmup_steps=cfg.warmup_steps,
                total_steps=cfg.max_iter,
                min_factor=cfg.min_lr_factor,
            ),
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(cfg.seed)
        history: list[dict[str, float]] = []
        best_loss = float("inf")
        best_step = 0
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0

        for step in range(1, cfg.max_iter + 1):
            self.model.train()
            indices = self._sample_train_indices(generator)
            optimizer.zero_grad(set_to_none=True)
            x_hr = self._render_image()
            pred = self._predict_batch(x_hr, indices)
            target = self.observations.index_select(0, indices)
            loss = torch.mean((pred - target).square())
            loss.backward()
            if cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            loss_value = float(loss.detach().cpu())
            record: dict[str, float] = {
                "step": float(step),
                "train_loss": loss_value,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            if cfg.val_interval > 0 and (step == 1 or step % cfg.val_interval == 0):
                if self.val_indices.numel():
                    record["holdout_loss"] = self.evaluate_loss(self.val_indices)
                record["train_set_loss"] = self.evaluate_loss(self.train_indices)
            history.append(record)

            monitor = record.get("holdout_loss", record.get("train_set_loss", loss_value))
            if monitor + cfg.early_stop_min_delta < best_loss:
                best_loss = monitor
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if (
                cfg.early_stop_patience > 0
                and step >= cfg.early_stop_min_steps
                and stale >= cfg.early_stop_patience
            ):
                break

        if best_state is not None:
            self.model.load_state_dict({key: value.to(self.device) for key, value in best_state.items()})
        image = self._render_image().detach().cpu()
        return TrainResult(image=image, history=history, best_loss=best_loss, best_step=best_step)

    def _predict_batch(self, x_hr: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        preds = [self.forward_operator(x_hr, int(idx)) for idx in indices.detach().cpu().tolist()]
        return torch.stack(preds, dim=0)

    @torch.no_grad()
    def evaluate_loss(self, indices: torch.Tensor | None = None) -> float:
        self.model.eval()
        if indices is None:
            indices = torch.arange(self.observations.shape[0], device=self.device)
        else:
            indices = indices.to(self.device)
        x_hr = self._render_image()
        pred = self._predict_batch(x_hr, indices)
        target = self.observations.index_select(0, indices)
        return float(torch.mean((pred - target).square()).cpu())


def train_inr(
    model: nn.Module,
    observations: torch.Tensor | np.ndarray,
    shifts: torch.Tensor | np.ndarray,
    **kwargs: Any,
) -> TrainResult:
    return INRTrainer(model, observations, shifts, **kwargs).fit()
