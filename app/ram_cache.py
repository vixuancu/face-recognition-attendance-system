"""
RAM Cache — Thay thế Redis bằng Python dict trong bộ nhớ.

Cấu trúc cache:
  _cache[course_id] = {
      "embeddings": np.ndarray (N, 512),
      "student_ids": list[int],      # mỗi embedding thuộc SV nào
      "photo_ids":   list[int],
      "metadata":    dict,           # {student_id: {"student_code": ..., "full_name": ...}}
      "attended":    set[int],       # student_ids đã điểm danh trong phiên
      "loaded_at":   str,            # ISO timestamp
      "n_students":  int,
  }

Không cần service ngoài, không cần connection pool.
Toàn bộ hàm là SYNC (không async) — dùng trực tiếp, không cần await.
"""

import logging
import time
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger("ram_cache")

# ── Store toàn cục ──────────────────────────────────────────
_cache: dict[int, dict] = {}


# ════════════════════════════════════════════════════════════
#  LOAD / GET EMBEDDINGS
# ════════════════════════════════════════════════════════════

def load_class_embeddings_to_cache(
    course_id: int,
    students_data: list[dict],
) -> int:
    """
    Load embeddings của 1 lớp vào RAM cache.

    Args:
        course_id: ID lớp tín chỉ
        students_data: List[{
            "student_id": int,
            "student_code": str,
            "full_name": str,
            "photos": [{"photo_id": int, "embedding": list[float]}, ...]
        }]

    Returns:
        Số lượng embeddings đã load.
    """
    t0 = time.time()

    embeddings = []
    student_ids = []
    photo_ids = []
    metadata: dict[int, dict] = {}

    for student in students_data:
        sid = student["student_id"]
        metadata[sid] = {
            "student_code": student["student_code"],
            "full_name":    student["full_name"],
        }
        for photo in student["photos"]:
            embeddings.append(photo["embedding"])
            student_ids.append(sid)
            photo_ids.append(photo["photo_id"])

    emb_matrix = (
        np.array(embeddings, dtype=np.float32)
        if embeddings
        else np.empty((0, 512), dtype=np.float32)
    )

    _cache[course_id] = {
        "embeddings":  emb_matrix,
        "student_ids": student_ids,
        "photo_ids":   photo_ids,
        "metadata":    metadata,
        "attended":    set(),
        "loaded_at":   datetime.now().isoformat(timespec="seconds"),
        "n_students":  len(students_data),
    }

    elapsed_ms = (time.time() - t0) * 1000
    count = len(embeddings)
    logger.info(
        f"[CACHE] Loaded course_id={course_id}: "
        f"{len(students_data)} students, {count} embeddings ({elapsed_ms:.0f}ms)"
    )
    return count


def get_cached_embeddings(course_id: int) -> Optional[dict]:
    """
    Lấy embeddings từ RAM cache.

    Returns:
        {
            "embeddings": np.ndarray (N, 512),
            "student_ids": list[int],
            "photo_ids":   list[int],
            "metadata":    {student_id: {"student_code": ..., "full_name": ...}},
        }
        hoặc None nếu chưa có cache.
    """
    entry = _cache.get(course_id)
    if entry is None:
        return None
    return {
        "embeddings":  entry["embeddings"],
        "student_ids": entry["student_ids"],
        "photo_ids":   entry["photo_ids"],
        "metadata":    entry["metadata"],
    }


# ════════════════════════════════════════════════════════════
#  ATTENDED SET
# ════════════════════════════════════════════════════════════

def mark_attended(course_id: int, student_id: int) -> None:
    """Đánh dấu SV đã điểm danh trong cache."""
    entry = _cache.get(course_id)
    if entry is None:
        return
    entry["attended"].add(student_id)
    total = len(entry["attended"])
    n = entry["n_students"]
    logger.info(
        f"[CACHE] mark_attended course_id={course_id} "
        f"student_id={student_id} ({total}/{n})"
    )


def is_attended(course_id: int, student_id: int) -> bool:
    """Kiểm tra SV đã điểm danh chưa."""
    entry = _cache.get(course_id)
    if entry is None:
        return False
    return student_id in entry["attended"]


def get_attended_set(course_id: int) -> set[int]:
    """Lấy set student_id đã điểm danh."""
    entry = _cache.get(course_id)
    if entry is None:
        return set()
    return set(entry["attended"])  # trả copy


# ════════════════════════════════════════════════════════════
#  CLEAR / STATUS
# ════════════════════════════════════════════════════════════

def clear_class_cache(course_id: int) -> None:
    """Xóa cache của 1 lớp."""
    if course_id in _cache:
        del _cache[course_id]
        logger.info(f"[CACHE] Cleared course_id={course_id}")


def get_cache_status() -> list[dict]:
    """
    Trả thông tin tất cả lớp đang được cache trong RAM.
    Dùng cho endpoint /api/cache/status.
    """
    result = []
    for course_id, entry in _cache.items():
        result.append({
            "course_id":    course_id,
            "n_students":   entry["n_students"],
            "n_embeddings": len(entry["student_ids"]),
            "n_attended":   len(entry["attended"]),
            "loaded_at":    entry["loaded_at"],
        })
    return result
