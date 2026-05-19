from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect lightweight EP08 smoke summaries.")
    parser.add_argument("--data-mode", choices=["synthetic", "real"], default="synthetic", help="accepted for CLI symmetry")
    parser.add_argument("--device", default="cpu", help="accepted for CLI symmetry; evaluation reads saved summaries")
    parser.add_argument("--max-iter", type=int, default=10, help="accepted for CLI symmetry")
    parser.add_argument("--n-frames", type=int, default=None, help="accepted for CLI symmetry")
    parser.add_argument("--patch-size", type=int, default=256, help="LR patch size; accepted for CLI symmetry")
    parser.add_argument("--batch-k", type=int, default=8, help="accepted for CLI symmetry")
    parser.add_argument("--alignment-method", default="contour_refined", help="accepted for CLI symmetry")
    parser.add_argument("--workers", type=int, default=1, help="accepted for CLI symmetry")
    parser.add_argument("--frame-audit-path", type=Path, default=None, help="accepted for CLI symmetry")
    parser.add_argument("--data-dir", type=Path, default=None, help="accepted for CLI symmetry")
    parser.add_argument("--output-dir", default="output/ep08_inr_sr")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    rows = []
    for name in ("siren", "wire", "deep_decoder"):
        summary_path = root / name / "summary.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append({"model": name, **payload})
        else:
            rows.append({"model": name, "missing": True})
    out_path = root / "eval_all_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
