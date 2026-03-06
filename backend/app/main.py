from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .db import fetch_hazards
from .models_loader import get_models  # noqa: F401  # ensure models can preload
from .ws_infer import websocket_inference_handler


BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = BASE_DIR / "static"


app = FastAPI(title="Hazard Eye – Road Hazard Detection and Mapping")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """
    Redirect root to the main Hazard Eye home page.
    """
    return RedirectResponse(url="/static/index.html")


@app.get("/home", include_in_schema=False)
async def home_page() -> FileResponse:
    """
    Optional explicit route for the home page.
    """
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/hazards")
async def get_hazards():
    """
    Return all stored road hazards for display on the map.
    """
    hazards = await fetch_hazards()
    return {"hazards": hazards}


@app.websocket("/ws/infer")
async def ws_infer(websocket: WebSocket):
    """
    WebSocket endpoint for real-time hazard detection.
    """
    await websocket_inference_handler(websocket)

