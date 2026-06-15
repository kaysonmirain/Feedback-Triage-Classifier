"""Fetch feedback items from GitHub issues or a local JSON file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

GITHUB_API = "https://api.github.com"


def load_local_json(path: str | Path) -> list[dict[str, Any]]:
    """Load feedback items from a local JSON file.

    Expected shape: a list of objects with at least ``id`` and ``text`` keys.
    Optional ``title`` and ``label`` (ground truth) are preserved when present.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with file_path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of feedback items.")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {index} is not an object.")
        text = item.get("text") or item.get("body") or item.get("title")
        if not text:
            raise ValueError(f"Item at index {index} is missing text content.")
        normalized.append(
            {
                "id": item.get("id", index + 1),
                "text": str(text).strip(),
                "title": item.get("title"),
                "label": item.get("label"),
                "source": item.get("source", "local"),
            }
        )
    return normalized


def fetch_github_issues(
    owner: str,
    repo: str,
    *,
    state: str = "open",
    per_page: int = 30,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch issues from a public GitHub repository via the REST API."""
    import requests

    url = f"{GITHUB_API}/repos/{owner}/{repo}/issues"
    headers = {"Accept": "application/vnd.github+json"}
    auth_token = token or os.environ.get("GITHUB_TOKEN")
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    params = {"state": state, "per_page": per_page}
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()

    issues = response.json()
    items: list[dict[str, Any]] = []
    for issue in issues:
        if "pull_request" in issue:
            continue
        title = issue.get("title", "").strip()
        body = (issue.get("body") or "").strip()
        text = f"{title}\n\n{body}".strip() if body else title
        items.append(
            {
                "id": issue.get("number"),
                "text": text,
                "title": title,
                "label": None,
                "source": f"github:{owner}/{repo}",
            }
        )
    return items


def _parse_repo_slug(slug: str) -> tuple[str, str]:
    if "/" not in slug:
        raise ValueError("Repository must be in 'owner/repo' format.")
    owner, repo = slug.split("/", 1)
    if not owner or not repo:
        raise ValueError("Repository must include both owner and repo.")
    return owner, repo


def load_feedback(
    *,
    source: str = "local",
    path: str | Path | None = None,
    owner: str | None = None,
    repo: str | None = None,
    state: str = "open",
    per_page: int = 30,
) -> list[dict[str, Any]]:
    """Load feedback from the configured source."""
    if source == "local":
        if not path:
            raise ValueError("path is required when source='local'.")
        return load_local_json(path)

    if source == "github":
        if not owner or not repo:
            raise ValueError("owner and repo are required when source='github'.")
        return fetch_github_issues(owner, repo, state=state, per_page=per_page)

    raise ValueError(f"Unsupported source: {source}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load feedback from a local JSON file or GitHub issues."
    )
    parser.add_argument(
        "--source",
        choices=["local", "github"],
        default="local",
        help="Input source.",
    )
    parser.add_argument(
        "--input",
        default="examples/sample_issues.json",
        help="Path to a local JSON file when --source=local.",
    )
    parser.add_argument(
        "--repo",
        help="GitHub repository in owner/repo format when --source=github.",
    )
    parser.add_argument("--owner", help="GitHub repository owner.")
    parser.add_argument("--name", help="GitHub repository name.")
    parser.add_argument("--state", default="open", choices=["open", "closed", "all"])
    parser.add_argument("--per-page", type=int, default=30)
    parser.add_argument(
        "--output",
        help="Optional JSON output path. If omitted, prints a compact preview.",
    )
    args = parser.parse_args()

    owner = args.owner
    repo = args.name
    if args.repo:
        owner, repo = _parse_repo_slug(args.repo)

    items = load_feedback(
        source=args.source,
        path=args.input,
        owner=owner,
        repo=repo,
        state=args.state,
        per_page=args.per_page,
    )

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items, indent=2), encoding="utf-8")
        print(f"Wrote {len(items)} feedback items to {path}")
        return

    print(json.dumps(items[:3], indent=2))
    if len(items) > 3:
        print(f"... {len(items) - 3} more items")


if __name__ == "__main__":
    main()
