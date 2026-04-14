from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.database import engine, Base
from app.routers.client import auth_router, game_router, history_router
from app.routers.admin import training_router, checkpoint_router, evaluation_router, stats_router

# Tạo bảng nếu chưa có
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Caro 7x7")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

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
    return {"message": "AI Caro 7x7 API"}
