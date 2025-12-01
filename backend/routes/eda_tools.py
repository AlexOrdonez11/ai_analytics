# routes/eda_tools.py

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import io
import matplotlib.pyplot as plt
import pandas as pd

from db import get_dataset_df, save_project_plot
from .gcs_utils import new_image_path, upload_image_and_get_url

# ---------- Args schemas ----------

class DistArgs(BaseModel):
    project_id: str
    dataset_id: str
    column: str
    bins: int = 30
    max_rows: int = 5000
    gcs_prefix: str = "plots/dist"

class ScatterArgs(BaseModel):
    project_id: str
    dataset_id: str
    x: str
    y: str
    hue: Optional[str] = None
    max_rows: int = 5000
    gcs_prefix: str = "plots/scatter"

class TsArgs(BaseModel):
    project_id: str
    dataset_id: str
    time_col: str
    y: str
    group: Optional[str] = None
    max_rows: int = 5000
    gcs_prefix: str = "plots/ts"

class CorrArgs(BaseModel):
    project_id: str
    dataset_id: str
    max_rows: int = 5000
    top_k: int = 15
    gcs_prefix: str = "plots/corr"

# ---------- Helpers ----------

def _save_fig_to_gcs(fig, prefix: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buf.seek(0)
    path = new_image_path(prefix, "png")
    return upload_image_and_get_url(buf.read(), path, content_type="image/png")

# ---------- Tools ----------

def distribution_tool(args: DistArgs) -> Dict[str, Any]:
    df = get_dataset_df(args.project_id, args.dataset_id, max_rows=args.max_rows)
    if df.empty:
        raise ValueError("Dataset returned no rows.")
    if args.column not in df.columns:
        raise ValueError(f"Column '{args.column}' not found in dataset.")

    s = pd.to_numeric(df[args.column], errors="coerce").dropna()
    if s.empty:
        raise ValueError(f"Column '{args.column}' is not numeric or has no valid values.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(s, bins=args.bins)
    ax.set_title(f"Distribution of {args.column}")
    ax.set_xlabel(args.column)
    ax.set_ylabel("Frequency")

    url = _save_fig_to_gcs(fig, args.gcs_prefix)
    meta = {
        "type": "distribution",
        "column": args.column,
        "rows_used": int(len(s)),
    }
    saved = save_project_plot(args.project_id, args.dataset_id, url, "distribution", meta)

    # return what the agent/frontend need
    out = dict(saved)
    out["url"] = url
    return out

def scatter_tool(args: ScatterArgs) -> Dict[str, Any]:
    df = get_dataset_df(args.project_id, args.dataset_id, max_rows=args.max_rows)
    if df.empty:
        raise ValueError("Dataset returned no rows.")
    for col in [args.x, args.y]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in dataset.")
    x = pd.to_numeric(df[args.x], errors="coerce")
    y = pd.to_numeric(df[args.y], errors="coerce")
    mask = x.notna() & y.notna()

    fig, ax = plt.subplots(figsize=(7, 4))
    if args.hue and args.hue in df.columns:
        for k, g in df[mask].groupby(args.hue):
            ax.scatter(g[args.x], g[args.y], label=str(k), s=12, alpha=0.8)
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.scatter(x[mask], y[mask], s=12, alpha=0.8)

    ax.set_title(f"{args.y} vs {args.x}")
    ax.set_xlabel(args.x)
    ax.set_ylabel(args.y)

    url = _save_fig_to_gcs(fig, args.gcs_prefix)
    meta = {
        "type": "scatter",
        "x": args.x,
        "y": args.y,
        "hue": args.hue,
        "rows_used": int(mask.sum()),
    }
    saved = save_project_plot(args.project_id, args.dataset_id, url, "scatter", meta)
    out = dict(saved)
    out["url"] = url
    return out

def timeseries_tool(args: TsArgs) -> Dict[str, Any]:
    df = get_dataset_df(args.project_id, args.dataset_id, max_rows=args.max_rows)
    if df.empty:
        raise ValueError("Dataset returned no rows.")
    if args.time_col not in df.columns or args.y not in df.columns:
        raise ValueError("time_col or y not in dataset.")

    df = df.copy()
    df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")
    df = df.dropna(subset=[args.time_col, args.y])

    fig, ax = plt.subplots(figsize=(8, 4))
    if args.group and args.group in df.columns:
        for k, g in df.groupby(args.group):
            ax.plot(g[args.time_col], g[args.y], label=str(k), linewidth=1.4)
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.plot(df[args.time_col], df[args.y], linewidth=1.4)

    ax.set_title(f"{args.y} over time")
    ax.set_xlabel(args.time_col)
    ax.set_ylabel(args.y)

    url = _save_fig_to_gcs(fig, args.gcs_prefix)
    meta = {
        "type": "timeseries",
        "time_col": args.time_col,
        "y": args.y,
        "group": args.group,
        "rows_used": int(len(df)),
    }
    saved = save_project_plot(args.project_id, args.dataset_id, url, "timeseries", meta)
    out = dict(saved)
    out["url"] = url
    return out

def correlation_tool(args: CorrArgs) -> Dict[str, Any]:
    df = get_dataset_df(args.project_id, args.dataset_id, max_rows=args.max_rows)
    num = df.select_dtypes("number")
    if num.empty:
        raise ValueError("No numeric columns available for correlation.")
    if num.shape[1] > args.top_k:
        counts = num.notna().sum().sort_values(ascending=False)
        keep = counts.index[:args.top_k]
        num = num[keep]

    corr = num.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=7)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation heatmap")

    url = _save_fig_to_gcs(fig, args.gcs_prefix)
    meta = {
        "type": "correlation_heatmap",
        "columns": list(corr.columns),
    }
    saved = save_project_plot(args.project_id, args.dataset_id, url, "correlation_heatmap", meta)
    out = dict(saved)
    out["url"] = url
    return out

# ---------- Tools list ----------
from langchain.tools import StructuredTool

EDA_TOOLS = [
    StructuredTool.from_function(
        name="eda_distribution",
        description="Plot a histogram distribution for a numeric column from a CSV dataset in GCS and persist the plot.",
        func=distribution_tool,
        args_schema=DistArgs,
    ),
    StructuredTool.from_function(
        name="eda_scatter",
        description="Create a scatterplot (optionally colored by a category) from a CSV dataset in GCS and persist the plot.",
        func=scatter_tool,
        args_schema=ScatterArgs,
    ),
    StructuredTool.from_function(
        name="eda_timeseries",
        description="Plot a time series from a CSV dataset in GCS and persist the plot.",
        func=timeseries_tool,
        args_schema=TsArgs,
    ),
    StructuredTool.from_function(
        name="eda_correlation",
        description="Produce a correlation heatmap from a CSV dataset in GCS and persist the plot.",
        func=correlation_tool,
        args_schema=CorrArgs,
    ),
]
