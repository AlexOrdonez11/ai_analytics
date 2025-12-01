from bson import ObjectId
from pymongo import MongoClient
from dotenv import load_dotenv
from fastapi import HTTPException
import os
from typing import Optional, Dict, Any , List
from google.cloud import storage
import pandas as pd
import datetime
import io

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

# Connect to Mongo Atlas
try:
    client = MongoClient(MONGO_URI)
    print( "Connection Succeded")
except Exception as e:
    print("❌ ERROR:", str(e))
    raise HTTPException(status_code=500, detail="Internal Server Error")

db = client["AI_Analytics"]
users_collection = db["Users"]
project_collection = db["Projects"]
conversations_collection = db["Conversations"]
datasets_collection = db["datasets"]
charts_collection = db["charts"]


GCS_BUCKET = os.getenv("GCS_BUCKET")  # e.g. "analytics-ai-bucket"
_storage_client = storage.Client()

def _get_dataset_meta(project_id: str, dataset_id: Optional[str]) -> Dict[str, Any]:
    """
    Load dataset metadata from Mongo.
    Assumes a collection 'datasets' with fields like:
      { _id, project_id, name, gcs_uri, ... }
    where gcs_uri looks like: 'gs://analytics-ai-bucket/datasets/<dataset_id>/'
    """
    if not dataset_id:
        raise RuntimeError("dataset_id is required to locate GCS CSVs")

    db = client["AI_Analytics"]  # change if your DB name is different
    doc = db["datasets"].find_one({"project_id": project_id})
    if not doc:
        raise RuntimeError(f"Dataset metadata not found for project={project_id}, dataset={dataset_id}")
    return doc


def _gcs_prefix_from_uri(gcs_uri: str):
    """
    Parse 'gs://bucket/path/to/folder/' -> (bucket, 'path/to/folder/')
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    without = gcs_uri[5:]
    parts = without.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


def get_dataset_df(project_id: str, dataset_id: Optional[str], max_rows: int = 5000) -> pd.DataFrame:
    """
    Load up to max_rows rows from CSV files for this dataset from GCS.
    Works if gcs_uri is a single file OR a folder (CSV files may be one layer down).
    """
    meta = _get_dataset_meta(project_id, dataset_id)
    gcs_uri = meta.get("gcs_uri")
    if not gcs_uri:
        raise RuntimeError("Dataset metadata missing 'gcs_uri'")

    bucket_name, prefix = _gcs_prefix_from_uri(gcs_uri)
    bucket = _storage_client.bucket(bucket_name)

    # --- Case A: gcs_uri is a direct CSV file and exists ---
    if prefix.lower().endswith(".csv"):
        blob = bucket.blob(prefix)
        if blob.exists():
            data = blob.download_as_bytes()
            df = pd.read_csv(io.BytesIO(data), nrows=max_rows)
            df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
            return df
        # If it *looks* like a file but doesn't exist, fall back to parent folder
        if "/" in prefix:
            prefix = prefix.rsplit("/", 1)[0] + "/"

    # --- Case B: gcs_uri is a folder (or we fell back to its parent) ---
    # List *recursively* all CSVs under prefix (one layer down or deeper)
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    csv_blobs = []
    for b in bucket.list_blobs(prefix=prefix):
        if b.name.lower().endswith(".csv"):
            csv_blobs.append(b)

    # If still none, try one more fallback: use just the parent of current prefix
    if not csv_blobs and prefix and "/" in prefix.strip("/"):
        parent_prefix = prefix.strip("/").rsplit("/", 1)[0] + "/"
        for b in bucket.list_blobs(prefix=parent_prefix):
            if b.name.lower().endswith(".csv"):
                csv_blobs.append(b)

    if not csv_blobs:
        raise RuntimeError(f"No CSV files found under {gcs_uri} (searched recursively).")

    # Read CSVs until max_rows is reached (stable order)
    csv_blobs.sort(key=lambda b: b.name)
    frames: List[pd.DataFrame] = []
    rows_left = max_rows

    for b in csv_blobs:
        if rows_left <= 0:
            break
        data = b.download_as_bytes()
        df_part = pd.read_csv(io.BytesIO(data), nrows=rows_left)
        if df_part.empty:
            continue
        frames.append(df_part)
        rows_left -= len(df_part)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df

def save_project_plot(project_id: str, dataset_id: str, url: str, plot_type: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Push a plot record into projects.plots array and return the saved record.
    """
    coll = _get_projects_collection()
    pid = _parse_project_id(project_id)

    plot_id = str(ObjectId())
    record = {
        "id": plot_id,
        "url": url,
        "type": plot_type,
        "dataset_id": dataset_id,
        "meta": meta or {},
        "created_at": datetime.utcnow(),
    }

    coll.update_one(
        {"_id": pid},
        {"$push": {"plots": record}},
        upsert=False,  # assume project already exists
    )

    return record