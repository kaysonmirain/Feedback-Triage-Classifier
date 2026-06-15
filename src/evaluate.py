"""Evaluate classifier accuracy and generate result visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from .classify import CATEGORIES
except ImportError:
    from classify import CATEGORIES


def _ordered_categories(values: pd.Series) -> list[str]:
    present = [category for category in CATEGORIES if category in set(values)]
    extras = sorted(set(values) - set(CATEGORIES))
    return present + extras


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute overall accuracy and per-class counts for labeled rows."""
    labeled = df.dropna(subset=["label"]).copy()
    if labeled.empty:
        raise ValueError(
            "No labeled rows found. Each evaluation item needs a 'label' column."
        )

    labeled["label"] = labeled["label"].astype(str)
    labeled["category"] = labeled["category"].astype(str)

    matches = labeled["label"].eq(labeled["category"])
    accuracy = matches.mean()
    per_class = (
        labeled.groupby("label", sort=False)
        .size()
        .reindex(CATEGORIES, fill_value=0)
        .astype(int)
        .to_dict()
    )
    prediction_counts = (
        labeled.groupby("category", sort=False)
        .size()
        .reindex(CATEGORIES, fill_value=0)
        .astype(int)
        .to_dict()
    )

    return {
        "accuracy": float(accuracy),
        "support": len(labeled),
        "correct": int(matches.sum()),
        "per_class_counts": per_class,
        "prediction_counts": prediction_counts,
    }


def plot_category_distribution(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a bar chart of predicted category distribution."""
    counts = df["category"].value_counts()
    ordered = [category for category in CATEGORIES if category in counts.index]
    ordered += [category for category in counts.index if category not in ordered]
    plot_data = counts.reindex(ordered).fillna(0).astype(int)

    sns.set_theme(style="whitegrid", font_scale=1.0)
    palette = ["#2F6B8F", "#77A642", "#C7654A", "#7D5BA6", "#6C757D"]
    plot_df = plot_data.rename_axis("Category").reset_index(name="Count")
    plt.figure(figsize=(11, 6.5))
    axis = sns.barplot(
        data=plot_df,
        x="Category",
        y="Count",
        hue="Category",
        palette=palette[: len(plot_data)],
        legend=False,
    )
    for container in axis.containers:
        axis.bar_label(container, padding=4, fontsize=10)
    plt.title("Predicted Category Distribution", fontsize=16, weight="bold", pad=14)
    plt.xlabel("")
    plt.ylabel("Count")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, max(plot_data.values) + 1)
    plt.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_confusion_matrix(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a confusion-matrix heatmap comparing predictions to hand labels."""
    labeled = df.dropna(subset=["label"]).copy()
    if labeled.empty:
        raise ValueError("Cannot plot confusion matrix without labeled rows.")

    labels = _ordered_categories(labeled["label"])
    predictions = _ordered_categories(labeled["category"])
    matrix_labels = list(dict.fromkeys(labels + predictions))

    matrix = pd.crosstab(
        labeled["label"],
        labeled["category"],
        rownames=["Hand-Labeled Category"],
        colnames=["Predicted Category"],
    ).reindex(index=matrix_labels, columns=matrix_labels, fill_value=0)

    sns.set_theme(style="white", font_scale=0.95)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"label": "Count"},
    )
    plt.title(
        "Confusion Matrix: Hand Labels vs Predictions",
        fontsize=16,
        weight="bold",
        pad=14,
    )
    plt.xlabel("Predicted Category")
    plt.ylabel("Hand-Labeled Category")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def evaluate_predictions(
    predictions_csv: str | Path,
    *,
    distribution_plot: str | Path = "results/category_distribution.png",
    confusion_plot: str | Path = "results/confusion_matrix.png",
) -> dict:
    """Load predictions, compute metrics, and write evaluation visuals."""
    df = pd.read_csv(predictions_csv)
    metrics = compute_metrics(df)
    distribution_path = plot_category_distribution(df, distribution_plot)
    confusion_path = plot_confusion_matrix(df, confusion_plot)

    metrics["distribution_plot"] = str(distribution_path)
    metrics["confusion_plot"] = str(confusion_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate labeled predictions.")
    parser.add_argument(
        "--predictions",
        default="examples/sample_predictions.csv",
        help="CSV file containing label and category columns.",
    )
    parser.add_argument(
        "--distribution-plot",
        default="results/category_distribution.png",
        help="Output path for the category distribution bar chart.",
    )
    parser.add_argument(
        "--confusion-plot",
        default="results/confusion_matrix.png",
        help="Output path for the confusion matrix heatmap.",
    )
    args = parser.parse_args()

    metrics = evaluate_predictions(
        args.predictions,
        distribution_plot=args.distribution_plot,
        confusion_plot=args.confusion_plot,
    )

    print(
        f"Accuracy: {metrics['accuracy']:.2%} "
        f"({metrics['correct']}/{metrics['support']} labeled items)"
    )
    print("Hand-labeled counts:")
    for category, count in metrics["per_class_counts"].items():
        print(f"  {category}: {count}")
    print(f"Saved distribution chart to {metrics['distribution_plot']}")
    print(f"Saved confusion matrix to {metrics['confusion_plot']}")


if __name__ == "__main__":
    main()
