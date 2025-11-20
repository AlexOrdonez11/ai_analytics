import httpx, os, uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import base64
import json as pyjson
from datetime import timedelta
from db import datasets_collection, project_collection, charts_collection


#----- Config -----
DATABRICKS_INSTANCE = os.getenv("DBX_INSTANCE")     
DATABRICKS_TOKEN    = os.getenv("DBX_TOKEN")
GCS_BUCKET          = os.getenv("GCS_BUCKET", "analytics-ai-bucket")
JOB_ID              = os.getenv("DBX_EDA_JOB_ID", "573363216889606")  

router = APIRouter()
storage_client = storage.Client()

#----- Scatter Plot Tool Functions -----
class ScatterRequest(BaseModel):
    project_id: str
    x_col: str
    y_col: str
    category_col: str | None = None

@router.post("/charts/scatter")
def create_scatter(req: ScatterRequest):

    ds= datasets_collection.find_one({"project_id": req.project_id}, sort=[("createdAt", -1)])
    if not ds:
        raise HTTPException(status_code=404, detail="No dataset found for project")
    
    ds_path = ds.get("gcs_uri")
    if not ds_path: 
        raise HTTPException(status_code=400, detail="Dataset missing GCS object path")

    # 1) record in Mongo
    doc_id = str(uuid.uuid4())
    charts_collection.insert_one({
        "_id": doc_id, "type": "scatter", "params": req.dict(),
        "status": "running", "image_gcs_uri": None, "image_url": None
    })

    # 2) run-now with params
    payload = {
        "job_id": JOB_ID,
        "notebook_params": {
            "var1": req.x_col, "var2": req.y_col, "cat": req.category_col, "doc_id": doc_id, "input_path": ds_path
        }
    }
    with httpx.Client(timeout=30.0) as client:
        r = client.post(
            f"{DATABRICKS_INSTANCE}/api/2.1/jobs/run-now",
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
            json=payload
        )
    if r.status_code >= 300:
        charts_collection.update_one({"_id": doc_id}, {"$set": {"status": "error", "error": r.text}})
        raise HTTPException(502, f"Databricks run failed: {r.text}")

    run_id = r.json()["run_id"]
    charts_collection.update_one({"_id": doc_id}, {"$set": {"run_id": run_id}})
    return {"id": doc_id, "run_id": run_id, "status": "running"}

@router.post("/charts/{doc_id}/collect")
def collect_output(doc_id: str):
    doc = charts_collection.find_one({"_id": doc_id})
    if not doc or doc.get("status") not in ("running",):
        return {"status": doc.get("status") if doc else "unknown"}

    run_id = doc["run_id"]

    # 1) fetch run output (exit_value contains our base64)
    with httpx.Client(timeout=30.0) as client:
        r = client.get(
            f"{DATABRICKS_INSTANCE}/api/2.1/jobs/runs/get-output",
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
            params={"run_id": run_id}
        )
    if r.status_code >= 300:
        raise HTTPException(502, f"get-output failed: {r.text}")

    out = r.json()
    state = (out.get("metadata") or {}).get("state", {})
    life_cycle = state.get("life_cycle_state")
    result_state = state.get("result_state")

    if life_cycle not in ("TERMINATED", "INTERNAL_ERROR"):
        return {"status": "running"}  # still going

    if result_state != "SUCCESS":
        charts_collection.update_one({"_id": doc_id}, {"$set": {"status": "error", "error": str(out)[:2000]}})
        return {"status": "error"}

    # 2) decode the base64 PNG from the exit_value
    payload = pyjson.loads(out.get("notebook_output", {}).get("result", "{}"))
    b64 = payload.get("image_base64")
    if not b64:
        charts_collection.update_one({"_id": doc_id}, {"$set": {"status": "error", "error": "no-image"}})
        return {"status": "error"}

    image_bytes = base64.b64decode(b64)

    # 3) upload to GCS from backend (you control creds here)
    object_key = f"plots/{doc_id}.png"
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(object_key)
    blob.upload_from_string(image_bytes, content_type="image/png")

    # (Optional) signed URL
    signed_url = blob.generate_signed_url(version="v4", expiration=timedelta(hours=24), method="GET")

    charts_collection.update_one({"_id": doc_id}, {
        "$set": {"status": "ready", "image_gcs_uri": f"gs://{GCS_BUCKET}/{object_key}", "image_url": signed_url}
    })
    return {"status": "ready", "image_url": signed_url}
