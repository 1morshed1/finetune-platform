"""FastAPI application factory.

Wires the routers together, serves the dashboard, and exposes a couple of
meta-endpoints (`/health`, `/api/v1/config`). Import ``app`` for uvicorn:

    uvicorn ftp.app:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ftp import __version__
from ftp.config import get_settings
from ftp.db import get_engine

from .api.routes import datasets, endpoints, jobs, models

_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


def create_app() -> FastAPI:
    settings = get_settings()
    settings.ensure_dirs()
    get_engine()  # create tables on startup

    app = FastAPI(
        title="Fine-Tuning Platform",
        version=__version__,
        description="Upload data, validate, fine-tune, register, deploy, evaluate.",
    )

    # Permissive CORS: this is a local, single-user tool. Tighten for real deploys.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for module in (datasets, jobs, models, endpoints):
        app.include_router(module.router)

    @app.get("/health", tags=["meta"])
    async def health():
        return {"status": "ok", "version": __version__}

    @app.get("/api/v1/config", tags=["meta"])
    async def config():
        """What the UI needs to know about how this instance is configured."""
        return {
            "version": __version__,
            "trainer_backend": settings.trainer_backend,
            "deploy_backend": settings.deploy_backend,
            "eval_min_relative_improvement": settings.eval_min_relative_improvement,
        }

    # Serve the dashboard at "/" (mounted last so it doesn't shadow the API).
    if _FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="dashboard")

    return app


app = create_app()
