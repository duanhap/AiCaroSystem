from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.database import engine, Base
from app.routers.client import auth_router, game_router, history_router
from app.routers.admin import training_router, checkpoint_router, evaluation_router, stats_router
from app.routers.admin import auth_router as admin_auth_router
from app.middleware import admin_auth_middleware

# Import models để Base nhận diện
from app.models import user, game, game_step, checkpoint, training_log  # noqa

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Caro 7x7")

# Middleware auth cho admin
app.add_middleware(BaseHTTPMiddleware, dispatch=admin_auth_middleware)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Admin auth
app.include_router(admin_auth_router.router, prefix="/admin", tags=["admin-auth"])

# Client routes
app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
app.include_router(game_router.router, prefix="/game", tags=["game"])
app.include_router(history_router.router, prefix="/history", tags=["history"])

# Admin routes
app.include_router(training_router.router, prefix="/admin/training", tags=["admin-training"])
app.include_router(checkpoint_router.router, prefix="/admin/checkpoints", tags=["admin-checkpoints"])
app.include_router(evaluation_router.router, prefix="/admin/evaluation", tags=["admin-evaluation"])
app.include_router(stats_router.router, prefix="/admin/stats", tags=["admin-stats"])

@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/admin/login")
