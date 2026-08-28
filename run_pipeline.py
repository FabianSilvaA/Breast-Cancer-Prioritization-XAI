from __future__ import annotations
import argparse
from src.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the breast-cancer stratification, XAI, and prioritization pipeline."
    )
    parser.add_argument("--data", required=True, help="Path to authorized patient-level CSV.")
    parser.add_argument("--output", default="outputs", help="Output directory.")
    args = parser.parse_args()
    run_pipeline(args.data, args.output)

if __name__ == "__main__":
    main()
