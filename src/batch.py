"""Batch-classify feedback items and write labeled results to CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .classify import classify_items
    from .fetch import load_feedback
except ImportError:
    from classify import classify_items
    from fetch import load_feedback


def process_batch(
    items: list[dict],
    *,
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    """Classify each item and return a DataFrame of labeled feedback."""
    labeled = classify_items(items, provider=provider, api_key=api_key, model=model)
    return pd.DataFrame(labeled)


def write_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist labeled results to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch classify feedback items.")
    parser.add_argument(
        "--source",
        choices=["local", "github"],
        default="local",
        help="Input source: local JSON file or GitHub issues.",
    )
    parser.add_argument(
        "--input",
        default="examples/sample_issues.json",
        help="Path to local JSON input when --source=local.",
    )
    parser.add_argument("--owner", help="GitHub repository owner when --source=github.")
    parser.add_argument("--repo", help="GitHub repository name when --source=github.")
    parser.add_argument("--state", default="open", choices=["open", "closed", "all"])
    parser.add_argument("--per-page", type=int, default=30)
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of items to classify after loading.",
    )
    parser.add_argument(
        "--output",
        default="examples/sample_predictions.csv",
        help="CSV path for labeled output.",
    )
    parser.add_argument("--provider", default=None, help="LLM provider override.")
    parser.add_argument("--model", default=None, help="Model name override.")
    args = parser.parse_args()

    items = load_feedback(
        source=args.source,
        path=args.input,
        owner=args.owner,
        repo=args.repo,
        state=args.state,
        per_page=args.per_page,
    )
    if args.limit is not None:
        items = items[: args.limit]

    df = process_batch(items, provider=args.provider, model=args.model)
    output = write_csv(df, args.output)
    print(f"Wrote {len(df)} labeled rows to {output}")


if __name__ == "__main__":
    main()
