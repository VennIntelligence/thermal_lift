from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ep08.stage1 import parse_training_args, run_stage1_training


def parse_args():
    return parse_training_args(
        "wire",
        "Train a WIRE EP08 Stage 1 model on highpass-domain observations.",
    )


def main() -> None:
    run_stage1_training("wire", parse_args())


if __name__ == "__main__":
    main()
