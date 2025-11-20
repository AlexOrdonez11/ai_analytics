from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/")
def root():
    return {"message": "API is up"}

from routes import user, project, conversation, datasets, analyst, tools

app.include_router(user.router)
app.include_router(project.router)
app.include_router(conversation.router)
app.include_router(datasets.router)
app.include_router(analyst.router)
app.include_router(tools.router) 