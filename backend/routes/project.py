from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId
from db import project_collection

router = APIRouter()

# ---------- Models ----------
class ProjectCreate(BaseModel):
    name: str = Field(min_length=1)
    description: Optional[str] = ""
    user_id: str = Field(min_length=1)

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class ProjectOut(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""
    user_id: str
    createdAt: datetime
    updatedAt: datetime

def _oid(id_str: str) -> ObjectId:
    try:
        return ObjectId(id_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")

def _normalize(doc: Dict[str, Any]) -> Dict[str, Any]:
    doc = dict(doc)
    doc["id"] = str(doc.pop("_id"))
    return doc

class PlotItem(BaseModel):
  id: str
  url: str
  type: str
  dataset_id: str
  meta: Dict[str, Any]
  created_at: str  # ISO string

class ProjectPlotsOut(BaseModel):
  project_id: str
  plots: List[PlotItem]

def _get_projects_collection():
    return project_collection

# ---------- Endpoints ----------

@router.get("/projects/{project_id}/plots", response_model=ProjectPlotsOut)
def list_project_plots(project_id: str):
    coll = _get_projects_collection()
    # handle ObjectId or string
    try:
        pid = ObjectId(project_id)
        proj = coll.find_one({"_id": pid})
        if not proj:
            proj = coll.find_one({"_id": project_id})
    except Exception:
        proj = coll.find_one({"_id": project_id})

    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    plots = proj.get("plots", []) or []
    out_plots: List[PlotItem] = []

    for p in plots:
        created_at = p.get("created_at")
        if isinstance(created_at, datetime):
            created_at = created_at.isoformat()
        out_plots.append(
            PlotItem(
                id=str(p.get("id") or p.get("_id")),
                url=p.get("url", ""),
                type=p.get("type", ""),
                dataset_id=str(p.get("dataset_id", "")),
                meta=p.get("meta") or {},
                created_at=created_at or "",
            )
        )

    return ProjectPlotsOut(project_id=project_id, plots=out_plots)

@router.post("/projects", response_model=ProjectOut)
def create_project(payload: ProjectCreate):
    # unique name per user
    if project_collection.find_one({"name": payload.name, "user_id": payload.user_id}):
        raise HTTPException(status_code=400, detail="Project name already exists")

    now = datetime.utcnow()
    new_project = {
        "name": payload.name,
        "description": payload.description or "",
        "user_id": payload.user_id,
        "createdAt": now,
        "updatedAt": now,
    }
    result = project_collection.insert_one(new_project)
    created = project_collection.find_one({"_id": result.inserted_id})
    return _normalize(created)

@router.get("/projects/{project_id}", response_model=ProjectOut)
def get_project(project_id: str):
    proj = project_collection.find_one({"_id": _oid(project_id)})
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    return _normalize(proj)

@router.put("/projects/{project_id}", response_model=ProjectOut)
def update_project(project_id: str, payload: ProjectUpdate):
    update = {k: v for k, v in payload.dict().items() if v is not None}
    if not update:
        raise HTTPException(status_code=400, detail="No changes provided")

    update["updatedAt"] = datetime.utcnow()
    res = project_collection.update_one({"_id": _oid(project_id)}, {"$set": update})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Project not found")

    proj = project_collection.find_one({"_id": _oid(project_id)})
    return _normalize(proj)

@router.delete("/projects/{project_id}")
def delete_project(project_id: str):
    res = project_collection.delete_one({"_id": _oid(project_id)})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"message": "Project deleted"}

# Keep your route shape for compatibility
@router.get("/projects/user/{user_id}", response_model=List[ProjectOut])
def get_user_projects(user_id: str):
    items = list(project_collection.find({"user_id": user_id}).sort("updatedAt", -1))
    return [_normalize(p) for p in items]