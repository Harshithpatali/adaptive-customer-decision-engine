"""Offline research/training entry point.

This module is never imported by the production API. Production loads only frozen artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/hillstrom_email_marketing.csv")
    parser.add_argument("--output", default="models/production")
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    print(f"Research training source: {args.data}")
    print("Production model selection is frozen separately; use the dedicated research scripts for search.")


if __name__ == "__main__":
    main()
