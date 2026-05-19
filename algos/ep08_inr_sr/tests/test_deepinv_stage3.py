from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT = ROOT / "scripts" / "train_deepinv_dip.py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train_deepinv_dip_stage3", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_spatial_shape_accepts_rectangular_and_square_forms() -> None:
    module = _load_train_module()

    assert module.parse_spatial_shape("30,40") == (30, 40)
    assert module.parse_spatial_shape("30x40") == (30, 40)
    assert module.parse_spatial_shape([30, 40]) == (30, 40)
    assert module.parse_spatial_shape((30, 40)) == (30, 40)
    assert module.parse_spatial_shape(30) == (30, 30)


@pytest.mark.parametrize(
    "value",
    [0, -1, True, "", "30,", "30,0", "30x-1", "30,40,50", "30.0,40", [30], [30, 0]],
)
def test_parse_spatial_shape_rejects_invalid_shapes(value) -> None:
    module = _load_train_module()

    with pytest.raises(ValueError):
        module.parse_spatial_shape(value)


def test_apply_cli_accepts_full_patch_rectangular_in_spatial_and_batch_k(monkeypatch) -> None:
    module = _load_train_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_deepinv_dip.py",
            "--patch-shape",
            "full",
            "--in-spatial",
            "30x40",
            "--batch-k",
            "8",
        ],
    )

    args = module.parse_args()
    cfg = module._apply_cli(module._default_config(), args)

    assert cfg["data"]["patch_shape"] is None
    assert cfg["deepinv"]["in_spatial"] == [30, 40]
    assert cfg["train"]["batch_k"] == 8


def test_rectangular_convdecoder_render_shape_smoke() -> None:
    pytest.importorskip("deepinv")
    torch = pytest.importorskip("torch")
    module = _load_train_module()
    cfg = module._default_config()
    cfg["deepinv"].update({"channels": 4, "layers": 2, "in_spatial": [3, 5], "verbose": False})

    backbone, z = module._build_convdecoder(cfg, (12, 20), torch.device("cpu"))
    image = module._render_backbone(backbone, z, (12, 20))

    assert tuple(image.shape) == (12, 20)
