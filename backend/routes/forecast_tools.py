# routes/forecast_tools.py

from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from datetime import timedelta
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from langchain_core.tools import tool  # same style as your EDA tools

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

from db import get_dataset_df, save_project_plot
from .gcs_utils import (
    new_image_path,
    upload_image_and_get_url,
    new_file_path,
    upload_bytes_and_get_url,
)


# ---------- Helpers ----------

def _save_fig_to_gcs(fig, prefix: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buf.seek(0)
    path = new_image_path(prefix, "png")
    return upload_image_and_get_url(buf.read(), path, content_type="image/png")


def _basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    nonzero = y_true != 0
    if np.any(nonzero):
        mape = float(np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100)
    else:
        mape = float("nan")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def _prepare_series(
    project_id: str,
    dataset_id: str,
    time_col: str,
    y_col: str,
    max_rows: int,
) -> pd.Series:
    df = get_dataset_df(project_id, dataset_id, max_rows=max_rows)
    if df.empty:
        raise ValueError("Dataset returned no rows.")
    if time_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Columns '{time_col}' or '{y_col}' not found in dataset.")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, y_col])
    print("After dropping NA, rows left:", len(df))
    print(df[[time_col, y_col]].head(5))
    df = df.sort_values(time_col)
    clean = df[y_col].astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]

    s = pd.to_numeric(clean, errors="coerce").dropna()

    s.index = df[time_col]
    s = s.dropna()

    if s.empty:
        raise ValueError(f"Target column '{y_col}' has no numeric values after cleaning.")
    return s


def _train_test_split_series(s: pd.Series, test_size: int) -> Dict[str, pd.Series]:
    if len(s) <= test_size:
        raise ValueError(f"Series too short ({len(s)}) for test_size={test_size}.")
    train = s.iloc[:-test_size]
    test = s.iloc[-test_size:]
    return {"train": train, "test": test}


def _plot_forecast(
    train: pd.Series,
    test: pd.Series,
    forecast: pd.Series,
    title: str,
    y_label: str,
    prefix: str,
) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    train.plot(ax=ax, label="Train", linewidth=1.2)
    test.plot(ax=ax, label="Test", linewidth=1.2)
    forecast.plot(ax=ax, label="Forecast", linestyle="--", linewidth=1.4)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.legend()
    url = _save_fig_to_gcs(fig, prefix)
    return url


def _future_index_from_series(s: pd.Series, horizon: int) -> pd.DatetimeIndex:
    last_idx = s.index[-1]
    if len(s.index) > 1:
        inferred = pd.infer_freq(s.index)
        if inferred is not None:
            freq = inferred
        else:
            freq = s.index[1] - s.index[0]
    else:
        # fallback: 1 day if only one point
        freq = "D"

    if isinstance(freq, timedelta):
        future_index = pd.date_range(last_idx + freq, periods=horizon, freq=freq)
    else:
        future_index = pd.date_range(last_idx, periods=horizon + 1, freq=freq)[1:]
    return future_index


# ---------- Arg schemas ----------

class BaseForecastArgs(BaseModel):
    project_id: str
    dataset_id: str
    time_col: str
    y_col: str
    horizon: int = 10
    test_size: int = 20
    max_rows: int = 5000
    gcs_prefix: str = "plots/forecast"


class HoltArgs(BaseForecastArgs):
    seasonal: Optional[str] = None   # None, "add", or "mul"
    seasonal_periods: Optional[int] = None


class ArimaArgs(BaseForecastArgs):
    p: int = 1
    d: int = 1
    q: int = 1


class RegrArgs(BaseForecastArgs):
    exog_cols: Optional[List[str]] = None  # optional extra features from the same dataset


# ---------- Core model implementations ----------

def _forecast_holt_impl(args: HoltArgs) -> Dict[str, Any]:
    s = _prepare_series(args.project_id, args.dataset_id, args.time_col, args.y_col, args.max_rows)
    split = _train_test_split_series(s, args.test_size)
    train, test = split["train"], split["test"]

    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal=args.seasonal,
        seasonal_periods=args.seasonal_periods if args.seasonal else None,
    ).fit(optimized=True)

    test_forecast = model.forecast(len(test))
    future_forecast = model.forecast(args.horizon)

    metrics = _basic_metrics(test.values, test_forecast.values)

    future_index = _future_index_from_series(s, args.horizon)
    future_forecast.index = future_index

    combined_forecast = pd.concat([test_forecast, future_forecast])
    url = _plot_forecast(
        train,
        test,
        combined_forecast,
        f"Holt/Holt-Winters forecast for {args.y_col}",
        args.y_col,
        args.gcs_prefix,
    )

    combined_forecast = pd.concat([test_forecast, future_forecast])
    url = _plot_forecast(
        train,
        test,
        combined_forecast,
        f"Holt/Holt-Winters forecast for {args.y_col}",
        args.y_col,
        args.gcs_prefix,
    )

    # ---------- NEW: CSV report ---------- #
    report_df = pd.DataFrame(
        {
            "timestamp": future_forecast.index,
            "forecast": future_forecast.values,
        }
    )
    csv_buf = io.StringIO()
    report_df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    report_path = new_file_path("reports/forecast_holt", ext="csv")
    report_url = upload_bytes_and_get_url(
        csv_bytes,
        report_path,
        content_type="text/csv",
    )
    # ------------------------------------- #

    meta = {
        "model": "holt_winters",
        "y_col": args.y_col,
        "time_col": args.time_col,
        "seasonal": args.seasonal,
        "seasonal_periods": args.seasonal_periods,
        "horizon": args.horizon,
        "test_size": args.test_size,
        "metrics": metrics,
        "report_path": report_path,
    }
    saved = save_project_plot(args.project_id, args.dataset_id, url, "forecast_holt", meta)
    out = dict(saved)
    out["url"] = url
    out["metrics"] = metrics
    out["horizon_forecast"] = {str(k): float(v) for k, v in future_forecast.items()}
    out["report_url"] = report_url
    return out


def _forecast_arima_impl(args: ArimaArgs) -> Dict[str, Any]:
    s = _prepare_series(args.project_id, args.dataset_id, args.time_col, args.y_col, args.max_rows)
    split = _train_test_split_series(s, args.test_size)
    train, test = split["train"], split["test"]

    model = ARIMA(train, order=(args.p, args.d, args.q)).fit()

    test_forecast = model.forecast(len(test))
    future_forecast = model.forecast(args.horizon)

    metrics = _basic_metrics(test.values, test_forecast.values)

    future_index = _future_index_from_series(s, args.horizon)
    future_forecast.index = future_index

    combined_forecast = pd.concat([test_forecast, future_forecast])
    url = _plot_forecast(
        train,
        test,
        combined_forecast,
        f"ARIMA({args.p},{args.d},{args.q}) forecast for {args.y_col}",
        args.y_col,
        args.gcs_prefix,
    )

    # ---------- NEW: CSV report with future predictions ---------- #
    report_df = pd.DataFrame(
        {
            "timestamp": future_forecast.index,
            "forecast": future_forecast.values,
        }
    )
    csv_buf = io.StringIO()
    report_df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    report_path = new_file_path("reports/forecast_arima", ext="csv")
    report_url = upload_bytes_and_get_url(
        csv_bytes,
        report_path,
        content_type="text/csv",
    )
    # ------------------------------------------------------------- #

    meta = {
        "model": "arima",
        "order": [args.p, args.d, args.q],
        "y_col": args.y_col,
        "time_col": args.time_col,
        "horizon": args.horizon,
        "test_size": args.test_size,
        "metrics": metrics,
        "report_path": report_path,   # keep path in Mongo
    }
    saved = save_project_plot(args.project_id, args.dataset_id, url, "forecast_arima", meta)
    out = dict(saved)
    out["url"] = url
    out["metrics"] = metrics
    out["horizon_forecast"] = {str(k): float(v) for k, v in future_forecast.items()}
    out["report_url"] = report_url   # 🔥 public CSV URL for frontend download
    return out


def _forecast_regression_impl(args: RegrArgs) -> Dict[str, Any]:
    df = get_dataset_df(args.project_id, args.dataset_id, max_rows=args.max_rows)
    if df.empty:
        raise ValueError("Dataset returned no rows.")
    if args.time_col not in df.columns or args.y_col not in df.columns:
        raise ValueError("time_col or y_col not in dataset.")

    df = df.copy()
    df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")
    df = df.dropna(subset=[args.time_col, args.y_col])
    df = df.sort_values(args.time_col)

    df["_t"] = np.arange(len(df))  # simple time index
    X_cols = ["_t"]

    if args.exog_cols:
        for col in args.exog_cols:
            if col in df.columns:
                X_cols.append(col)

    X = df[X_cols]
    y = pd.to_numeric(df[args.y_col], errors="coerce")
    mask = y.notna()
    X, y = X[mask], y[mask]

    if len(y) <= args.test_size:
        raise ValueError(f"Not enough points ({len(y)}) for test_size={args.test_size}.")

    X_train, X_test = X.iloc[:-args.test_size], X.iloc[-args.test_size:]
    y_train, y_test = y.iloc[:-args.test_size], y.iloc[-args.test_size:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    # Build future design matrix
    last_t = X["_t"].max()
    future_t = np.arange(last_t + 1, last_t + 1 + args.horizon)
    future_df = pd.DataFrame({"_t": future_t})

    # For exogenous vars, assume last observed value into the future (simple but works)
    if args.exog_cols:
        for col in args.exog_cols:
            if col in df.columns:
                future_df[col] = df[col].iloc[-1]

    y_future = model.predict(future_df[X_cols])

    # Build indices for plotting
    idx = df[args.time_col]
    s_full = pd.Series(y.values, index=idx)

    test_index = y_test.index
    test_index_ts = df.loc[test_index, args.time_col]

    # infer frequency for future index
    if len(idx) > 1:
        inferred = pd.infer_freq(idx)
        if inferred is not None:
            freq = inferred
        else:
            freq = idx.iloc[1] - idx.iloc[0]
    else:
        freq = "D"

    last_ts = idx.iloc[-1]
    if isinstance(freq, timedelta):
        future_index = pd.date_range(last_ts + freq, periods=args.horizon, freq=freq)
    else:
        future_index = pd.date_range(last_ts, periods=args.horizon + 1, freq=freq)[1:]

    train_series = s_full.iloc[:-args.test_size]
    test_series = pd.Series(y_test.values, index=test_index_ts)
    test_forecast_series = pd.Series(y_test_pred, index=test_index_ts)
    future_forecast_series = pd.Series(y_future, index=future_index)

    metrics = _basic_metrics(y_test.values, y_test_pred)

    combined_forecast = pd.concat([test_forecast_series, future_forecast_series])
    url = _plot_forecast(
        train_series,
        test_series,
        combined_forecast,
        f"Regression forecast for {args.y_col}",
        args.y_col,
        args.gcs_prefix,
    )
    # ---------- NEW: CSV report with future predictions ---------- #
    report_df = pd.DataFrame(
        {
            "timestamp": future_forecast_series.index,
            "forecast": future_forecast_series.values,
        }
    )
    csv_buf = io.StringIO()
    report_df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    report_path = new_file_path("reports/forecast_regression", ext="csv")
    report_url = upload_bytes_and_get_url(
        csv_bytes,
        report_path,
        content_type="text/csv",
    )
    # ---------------------------------------------------- #

    meta = {
        "model": "time_regression",
        "y_col": args.y_col,
        "time_col": args.time_col,
        "horizon": args.horizon,
        "test_size": args.test_size,
        "exog_cols": args.exog_cols or [],
        "metrics": metrics,
        "report_path": report_path,
    }
    saved = save_project_plot(args.project_id, args.dataset_id, url, "forecast_regression", meta)
    out = dict(saved)
    out["url"] = url
    out["metrics"] = metrics
    out["horizon_forecast"] = {str(k): float(v) for k, v in future_forecast_series.items()}
    out["report_url"] = report_url
    return out


# ---------- LangChain tools (modern @tool API) ----------

@tool
def forecast_holt(
    project_id: str,
    dataset_id: str,
    time_col: str,
    y_col: str,
    horizon: int = 10,
    test_size: int = 20,
    max_rows: int = 5000,
    gcs_prefix: str = "plots/forecast",
    seasonal: Optional[str] = None,
    seasonal_periods: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fit a Holt/Holt-Winters model on a time series from the dataset.
    Returns metrics, horizon forecasts, and a GCS plot URL (also stored in Mongo).
    """
    args = HoltArgs(
        project_id=project_id,
        dataset_id=dataset_id,
        time_col=time_col,
        y_col=y_col,
        horizon=horizon,
        test_size=test_size,
        max_rows=max_rows,
        gcs_prefix=gcs_prefix,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
    )
    return _forecast_holt_impl(args)


@tool
def forecast_arima(
    project_id: str,
    dataset_id: str,
    time_col: str,
    y_col: str,
    horizon: int = 10,
    test_size: int = 20,
    max_rows: int = 5000,
    gcs_prefix: str = "plots/forecast",
    p: int = 1,
    d: int = 1,
    q: int = 1,
) -> Dict[str, Any]:
    """
    Fit an ARIMA(p,d,q) model on a time series from the dataset.
    Returns metrics, horizon forecasts, and a GCS plot URL (also stored in Mongo).
    """
    args = ArimaArgs(
        project_id=project_id,
        dataset_id=dataset_id,
        time_col=time_col,
        y_col=y_col,
        horizon=horizon,
        test_size=test_size,
        max_rows=max_rows,
        gcs_prefix=gcs_prefix,
        p=p,
        d=d,
        q=q,
    )
    return _forecast_arima_impl(args)


@tool
def forecast_regression(
    project_id: str,
    dataset_id: str,
    time_col: str,
    y_col: str,
    horizon: int = 10,
    test_size: int = 20,
    max_rows: int = 5000,
    gcs_prefix: str = "plots/forecast",
    exog_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Fit a time-based regression (optionally with exogenous variables) on a series.
    Returns metrics, horizon forecasts, and a GCS plot URL (also stored in Mongo).
    """
    args = RegrArgs(
        project_id=project_id,
        dataset_id=dataset_id,
        time_col=time_col,
        y_col=y_col,
        horizon=horizon,
        test_size=test_size,
        max_rows=max_rows,
        gcs_prefix=gcs_prefix,
        exog_cols=exog_cols,
    )
    return _forecast_regression_impl(args)


# ---------- Tools list ----------

FORECAST_TOOLS = [
    forecast_holt,
    forecast_arima,
    forecast_regression,
]
