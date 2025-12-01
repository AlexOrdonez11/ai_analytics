# routes/eda_tools.py

from typing import Optional, Dict, Any, List
from pydantic import BaseModel
import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from langchain_core.tools import tool  # <-- modern tools API

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

class SummaryArgs(BaseModel):
    project_id: str
    dataset_id: str
    max_rows: int = 5000
    top_k_categories: int = 5 # for categorical tops

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

def smart_parse_time(df: pd.DataFrame, time_col: str) -> pd.Series:
    """
    Try hard to convert a time column to pandas datetime:
    - If it's already datetime-like, just return it.
    - Try generic to_datetime with inference.
    - If mostly NaT, try a list of common explicit formats.
    - If still failing, raise a helpful error with sample raw values.
    """
    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not in dataset. Available: {df.columns.tolist()}")

    s_raw = df[time_col]

    # 1) already datetime?
    if np.issubdtype(s_raw.dtype, np.datetime64):
        return pd.to_datetime(s_raw)

    raw_str = s_raw.astype(str).str.strip()

    # 2) generic parse with inference
    parsed = pd.to_datetime(raw_str, errors="coerce", infer_datetime_format=True)
    non_null_ratio = parsed.notna().mean()

    # If at least half of the values parsed ok, accept it
    if non_null_ratio >= 0.5:
        return parsed

    # 3) Try a list of explicit formats
    candidate_formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%Y-%m",
        "%b-%y",   # e.g. "Mar-04"
        "%b-%Y",   # e.g. "Mar-2004"
        "%Y",      # just year
    ]

    for fmt in candidate_formats:
        try:
            parsed_fmt = pd.to_datetime(raw_str, format=fmt, errors="coerce")
        except Exception:
            continue

        ratio = parsed_fmt.notna().mean()
        if ratio >= 0.8:  # require most rows to parse for an explicit format
            print(f"smart_parse_time: using explicit format '{fmt}' for column '{time_col}'", flush=True)
            return parsed_fmt

    # 4) As a last resort, give a clear error
    raise ValueError(
        f"Could not parse time_col '{time_col}' to datetime. "
        f"Sample raw values: {raw_str.head(8).tolist()}. "
        f"Consider specifying a format like '%b-%y' for values such as 'Mar-04'."
    )


# ---------- Core plotting functions (unchanged) ----------

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
    print("Initial data sample:")
    print(df[[args.time_col, args.y]].head(5))
    df[args.time_col] = smart_parse_time(df, args.time_col)
    print("After datetime conversion, sample data:")
    print(df[[args.time_col, args.y]].head(5))
    df = df.dropna(subset=[args.time_col, args.y])
    print("After dropping NA, rows left:", len(df))
    print(df[[args.time_col, args.y]].head(5))
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

def summary_tool(args: SummaryArgs) -> Dict[str, Any]:
    df = get_dataset_df(args.project_id, args.dataset_id, max_rows=args.max_rows)
    if df.empty:
        raise ValueError("Dataset returned no rows.")

    rows, cols = df.shape

    # Column-level basics
    col_info = []
    for c in df.columns:
        series = df[c]
        non_null = int(series.notna().sum())
        missing = int(series.isna().sum())
        nunique = int(series.nunique(dropna=True))
        sample_vals = series.dropna().astype(str).unique()[:args.top_k_categories].tolist()
        col_info.append(
            {
                "name": c,
                "dtype": str(series.dtype),
                "non_null": non_null,
                "missing": missing,
                "nunique": nunique,
                "sample_values": sample_vals,
            }
        )

    # Numeric describe
    num = df.select_dtypes("number")
    num_stats: Dict[str, Any] = {}
    if not num.empty:
        desc = num.describe().to_dict()  # {stat: {col: value}}
        # flip to {col: {stat: value}}
        for stat, colmap in desc.items():
            for col, val in colmap.items():
                num_stats.setdefault(col, {})[stat] = float(val)

    # Categorical tops
    cat_cols = df.select_dtypes(exclude="number").columns
    cat_tops: Dict[str, Any] = {}
    for c in cat_cols:
        vc = df[c].astype(str).value_counts().head(args.top_k_categories)
        cat_tops[c] = vc.to_dict()

    return {
        "type": "summary_stats",
        "shape": {"rows": rows, "cols": cols},
        "columns": col_info,
        "numeric_stats": num_stats,          # per-column describe
        "categorical_top_values": cat_tops,  # top categories per column
    }

# ---------- LangChain tools (wrappers using @tool) ----------

@tool
def eda_distribution(
    project_id: str,
    dataset_id: str,
    column: str,
    bins: int = 30,
    max_rows: int = 5000,
    gcs_prefix: str = "plots/dist",
) -> Dict[str, Any]:
    """
    Plot a histogram distribution for a numeric column from a CSV dataset in GCS
    and persist the plot. Returns a dict with at least 'url'.
    """
    args = DistArgs(
        project_id=project_id,
        dataset_id=dataset_id,
        column=column,
        bins=bins,
        max_rows=max_rows,
        gcs_prefix=gcs_prefix,
    )
    print("Running distribution tool with args:", args)
    return distribution_tool(args)


@tool
def eda_scatter(
    project_id: str,
    dataset_id: str,
    x: str,
    y: str,
    hue: Optional[str] = None,
    max_rows: int = 5000,
    gcs_prefix: str = "plots/scatter",
) -> Dict[str, Any]:
    """
    Create a scatterplot (optionally colored by a category) from a CSV dataset in GCS
    and persist the plot. Returns a dict with at least 'url'.
    """
    args = ScatterArgs(
        project_id=project_id,
        dataset_id=dataset_id,
        x=x,
        y=y,
        hue=hue,
        max_rows=max_rows,
        gcs_prefix=gcs_prefix,
    )
    print("Running scatter tool with args:", args)
    return scatter_tool(args)


@tool
def eda_timeseries(
    project_id: str,
    dataset_id: str,
    time_col: str,
    y: str,
    group: Optional[str] = None,
    max_rows: int = 5000,
    gcs_prefix: str = "plots/ts",
) -> Dict[str, Any]:
    """
    Plot a time series from a CSV dataset in GCS and persist the plot.
    Returns a dict with at least 'url'.
    """
    args = TsArgs(
        project_id=project_id,
        dataset_id=dataset_id,
        time_col=time_col,
        y=y,
        group=group,
        max_rows=max_rows,
        gcs_prefix=gcs_prefix,
    )
    print("Running timeseries tool with args:", args)
    return timeseries_tool(args)


@tool
def eda_correlation(
    project_id: str,
    dataset_id: str,
    max_rows: int = 5000,
    top_k: int = 15,
    gcs_prefix: str = "plots/corr",
) -> Dict[str, Any]:
    """
    Produce a correlation heatmap from a CSV dataset in GCS and persist the plot.
    Returns a dict with at least 'url'.
    """
    args = CorrArgs(
        project_id=project_id,
        dataset_id=dataset_id,
        max_rows=max_rows,
        top_k=top_k,
        gcs_prefix=gcs_prefix,
    )
    print("Running correlation tool with args:", args)
    return correlation_tool(args)

@tool
def eda_summary(
    project_id: str,
    dataset_id: str,
    max_rows: int = 5000,
    top_k_categories: int = 5,
) -> Dict[str, Any]:
    """
    Compute summary statistics for the dataset:
    shape, per-column dtypes, missingness, unique counts,
    numeric describe stats, and top categories for categorical columns.
    Returns a JSON dict (no plots).
    """
    args = SummaryArgs(
        project_id=project_id,
        dataset_id=dataset_id,
        max_rows=max_rows,
        top_k_categories=top_k_categories,
    )
    return summary_tool(args)


# ---------- Tools list ----------

EDA_TOOLS = [
    eda_distribution,
    eda_scatter,
    eda_timeseries,
    eda_correlation,
    eda_summary
]