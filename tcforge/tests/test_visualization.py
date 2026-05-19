from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tcforge.visualization as visualization


def test_plot_image_validates_rank_before_matplotlib_import() -> None:
    with pytest.raises(ValueError, match="2D"):
        visualization.plot_image(np.zeros((1, 2, 3), dtype=np.float32))


def test_plot_and_save_figure_when_matplotlib_is_available(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    fig, ax = visualization.plot_image(np.ones((4, 5), dtype=np.float32), colorbar=False)
    assert ax.figure is fig
    out = visualization.save_figure(fig, tmp_path / "preview.png")

    assert out.exists()
    assert out.stat().st_size > 0
