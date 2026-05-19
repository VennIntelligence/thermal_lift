#!/usr/bin/env python3
"""Backward-compatible entry point for the EP08 TCForge benchmark.

The original Stage 2 sanity script only covered Deep Decoder and
DeepInverse-DIP. Stage 3 prep now uses ``run_tcforge_benchmark.py`` as the
authoritative HR-ground-truth check for all four methods.
"""

from __future__ import annotations

from run_tcforge_benchmark import main


if __name__ == "__main__":
    main()
