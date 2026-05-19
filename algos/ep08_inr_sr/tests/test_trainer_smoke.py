from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ep08.models import DeepDecoder, Siren
from ep08.trainer import INRTrainer, TrainConfig


class TinyForwardOperator(nn.Module):
    def forward(self, x_hr: torch.Tensor, index: int) -> torch.Tensor:
        del index
        lr = F.avg_pool2d(x_hr[None, None], kernel_size=2, stride=2)[0, 0]
        return lr


def test_trainer_smoke_loss_backward() -> None:
    observations = torch.zeros(4, 8, 8)
    shifts = torch.zeros(4, 2)
    model = Siren(hidden_features=8, hidden_layers=1)
    trainer = INRTrainer(
        model,
        observations,
        shifts,
        hr_shape=(16, 16),
        forward_operator=TinyForwardOperator(),
        config=TrainConfig(max_iter=2, batch_k=2, warmup_steps=1, val_interval=1),
    )

    result = trainer.fit()

    assert result.image.shape == (16, 16)
    assert len(result.history) == 2
    assert result.best_loss >= 0.0


def test_trainer_supports_deep_decoder_smoke() -> None:
    observations = torch.zeros(3, 8, 8)
    shifts = torch.zeros(3, 2)
    model = DeepDecoder(latent_channels=4, hidden_channels=(8,), latent_spatial=(8, 8))
    trainer = INRTrainer(
        model,
        observations,
        shifts,
        hr_shape=(16, 16),
        forward_operator=TinyForwardOperator(),
        config=TrainConfig(max_iter=1, batch_k=2, warmup_steps=1, val_interval=1),
    )

    result = trainer.fit()

    assert result.image.shape == (16, 16)
    assert len(result.history) == 1


def test_cli_modules_import() -> None:
    for script in ("train_siren.py", "train_wire.py", "train_deep_decoder.py", "eval_all.py", "validate_p0.py"):
        path = ROOT / "scripts" / script
        spec = importlib.util.spec_from_file_location(script.removesuffix(".py"), path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "parse_args")


def test_train_cli_accepts_real_data_args(monkeypatch) -> None:
    path = ROOT / "scripts" / "train_siren.py"
    spec = importlib.util.spec_from_file_location("train_siren_cli_args", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_siren.py",
            "--data-mode",
            "real",
            "--alignment-method",
            "ncc_init",
            "--workers",
            "2",
            "--frame-audit-path",
            "audit.csv",
            "--data-dir",
            "frames",
        ],
    )

    args = module.parse_args()

    assert args.data_mode == "real"
    assert args.alignment_method == "ncc_init"
    assert args.workers == 2
    assert args.frame_audit_path == Path("audit.csv")
    assert args.data_dir == Path("frames")
