"""
Visualisation helpers.
Returns base64-encoded PNG strings so FastAPI can serve them
without writing to disk.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def get_correlation_matrix_image(df: pd.DataFrame) -> str | None:
    num_cols = df.select_dtypes(include="number")
    if num_cols.shape[1] < 2:
        return None

    corr = num_cols.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.8), max(5, len(corr) * 0.7)))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
    return _fig_to_base64(fig)


def get_distribution_images(df: pd.DataFrame, max_categories: int = 15) -> dict[str, str]:
    """
    Returns a dict with up to two keys:
      'numerical'   -> base64 PNG of histograms
      'categorical' -> base64 PNG of bar plots
    """
    images = {}

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if num_cols:
        n = len(num_cols)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()
        for i, col in enumerate(num_cols):
            axes[i].hist(df[col].dropna(), bins=30, color="#4C72B0", edgecolor="white", linewidth=0.5)
            axes[i].set_title(col, fontsize=11, fontweight="bold")
            axes[i].set_ylabel("Count")
            axes[i].spines[["top", "right"]].set_visible(False)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Numerical Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        images["numerical"] = _fig_to_base64(fig)

    if cat_cols:
        n = len(cat_cols)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()
        for i, col in enumerate(cat_cols):
            counts = df[col].value_counts().head(max_categories)
            axes[i].barh(counts.index.astype(str)[::-1], counts.values[::-1], color="#55A868")
            axes[i].set_title(col, fontsize=11, fontweight="bold")
            axes[i].set_xlabel("Count")
            axes[i].spines[["top", "right"]].set_visible(False)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Categorical Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        images["categorical"] = _fig_to_base64(fig)

    return images