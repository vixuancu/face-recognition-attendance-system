"""
FastAPI Application — Hệ thống điểm danh sinh viên CNTT.

Chạy:  uvicorn app.main:app --reload --port 8000
Docs:  http://localhost:8000/docs
Web:   http://localhost:8000/
"""

import logging
import threading
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.ram_cache import get_cache_status

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-12s | %(levelname)-5s | %(message)s",
)
logger = logging.getLogger("app")

# ── FastAPI app ──
app = FastAPI(
    title="Hệ thống điểm danh sinh viên CNTT",
    description="Face recognition attendance system — FastAPI + InsightFace + pgvector",
    version="3.0.0",
)

# ── Static files ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Include routers ──
from app.routers.students import router as students_router      # noqa: E402
from app.routers.courses import router as courses_router        # noqa: E402
from app.routers.attendance import router as attendance_router  # noqa: E402
from app.routers.pages import router as pages_router            # noqa: E402

app.include_router(students_router)
app.include_router(courses_router)
app.include_router(attendance_router)
app.include_router(pages_router)


def _warmup_insightface():
    """
    Warm-up InsightFace model khi server khởi động (background thread).
    Load buffalo_l (SCRFD + ArcFace) vào RAM lần đầu mất ~3-5s.
    Làm sẵn để request điểm danh đầu tiên không bị chậm.
    """
    try:
        logger.info("[WARMUP] Đang load InsightFace buffalo_l vào RAM...")
        from app.face_service import get_face_app
        import numpy as np
        face_app = get_face_app()
        # Chạy thử trên ảnh dummy để kích hoạt lazy-load model weights
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        face_app.get(dummy)
        logger.info("[WARMUP] ✓ InsightFace buffalo_l sẵn sàng (SCRFD + ArcFace)")
    except Exception as e:
        logger.warning(f"[WARMUP] Warm-up thất bại (không ảnh hưởng hoạt động): {e}")


# ── Startup ──
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 50)
    logger.info("  HỆ THỐNG ĐIỂM DANH SINH VIÊN CNTT v3.0")
    logger.info("  FastAPI + InsightFace + pgvector + RAM Cache")
    logger.info("  http://localhost:8000")
    logger.info("  Docs: http://localhost:8000/docs")
    logger.info("=" * 50)

    # Warm-up InsightFace trong background thread (không block server startup)
    threading.Thread(target=_warmup_insightface, daemon=True).start()


# ── Health check ──
@app.get("/health", tags=["System"])
def health_check():
    """Kiểm tra trạng thái server và RAM cache."""
    cache_status = get_cache_status()
    return {
        "status": "ok",
        "version": "3.0.0",
        "cache_engine": "RAM (Python dict)",
        "active_cached_courses": len(cache_status),
    }


# ── Cache status ──
@app.get("/api/cache/status", tags=["System"])
def cache_status_endpoint():
    """
    Xem nội dung RAM cache hiện tại.
    Trả về danh sách các lớp đang được cache, số SV, số đã điểm danh.
    """
    courses = get_cache_status()
    return {
        "total_courses_cached": len(courses),
        "active_courses": courses,
    }
