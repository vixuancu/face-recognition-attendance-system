"""
FastAPI Application — Hệ thống điểm danh sinh viên CNTT.

Chạy:  uvicorn app.main:app --reload --port 8000
Docs:  http://localhost:8000/docs
Web:   http://localhost:8000/
"""

import os
import logging
import threading

import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.redis_client import get_redis, close_redis

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-12s | %(levelname)-5s | %(message)s",
)
logger = logging.getLogger("app")

# ── FastAPI app ──
app = FastAPI(
    title="Hệ thống điểm danh sinh viên CNTT",
    description="Face recognition attendance system — FastAPI + Redis + pgvector",
    version="2.0.0",
)

# ── Static files ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Include routers ──
from app.routers.students import router as students_router    # noqa: E402
from app.routers.courses import router as courses_router      # noqa: E402
from app.routers.attendance import router as attendance_router  # noqa: E402
from app.routers.pages import router as pages_router          # noqa: E402

app.include_router(students_router)
app.include_router(courses_router)
app.include_router(attendance_router)
app.include_router(pages_router)


def _warmup_deepface():
    """
    Warm-up DeepFace models trong background thread khi server khởi động.
    Lần đầu tiên gọi DeepFace.represent() sẽ load RetinaFace + Facenet512
    vào RAM (mất 3-8 giây). Làm sẵn ở đây để request điểm danh đầu tiên
    không bị chậm.
    """
    try:
        logger.info("[WARMUP] Đang load DeepFace models vào RAM...")
        from deepface import DeepFace
        from app.config import FACE_MODEL, DETECTOR_BACKEND

        # Tạo ảnh giả 224x224 (3 kênh) để kích hoạt model load
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        DeepFace.represent(
            img_path=dummy,
            model_name=FACE_MODEL,
            enforce_detection=False,
            detector_backend=DETECTOR_BACKEND,
        )
        logger.info("[WARMUP] ✓ DeepFace models đã sẵn sàng (RetinaFace + Facenet512)")
    except Exception as e:
        logger.warning(f"[WARMUP] Warm-up thất bại (không ảnh hưởng hoạt động): {e}")


# ── Startup / Shutdown ──
@app.on_event("startup")
async def startup_event():
    """Khởi tạo Redis pool khi server start."""
    r = await get_redis()
    try:
        await r.ping()
        logger.info("[✓] Redis connected")
    except Exception as e:
        logger.warning(f"[!] Redis connection failed: {e}")

    logger.info("=" * 50)
    logger.info("  HỆ THỐNG ĐIỂM DANH SINH VIÊN CNTT v2.0")
    logger.info("  FastAPI + Redis + pgvector")
    logger.info("  http://localhost:8000")
    logger.info("  Docs: http://localhost:8000/docs")
    logger.info("=" * 50)

    # Warm-up DeepFace trong background thread (không block server startup)
    threading.Thread(target=_warmup_deepface, daemon=True).start()


@app.on_event("shutdown")
async def shutdown_event():
    """Đóng Redis pool khi server shutdown."""
    await close_redis()
    logger.info("[i] Redis pool closed")


# ── Health check ──
@app.get("/health", tags=["System"])
async def health_check():
    """Kiểm tra trạng thái server."""
    redis_ok = False
    try:
        r = await get_redis()
        await r.ping()
        redis_ok = True
    except Exception:
        pass

    return {
        "status": "ok",
        "redis": "connected" if redis_ok else "disconnected",
    }
