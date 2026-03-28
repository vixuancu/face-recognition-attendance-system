"""
Redis client — Cache embeddings theo lớp tín chỉ.

Cấu trúc Redis keys:
  class:{course_id}:embeddings   → Hash { "sid:{student_id}:pid:{photo_id}" → embedding_bytes }
  class:{course_id}:metadata     → Hash { student_id → JSON{name, code} }
  class:{course_id}:attended     → Set { student_id_1, student_id_2, ... }
  session:{session_id}:info      → Hash { course_id, session_name, ... }
"""

import json
import struct
from typing import Optional

import numpy as np
import redis.asyncio as aioredis

from app.config import REDIS_URL, REDIS_CACHE_TTL

# ── Connection pool (singleton) ──
_redis_pool: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """Lấy Redis connection (tạo pool nếu chưa có)."""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.from_url(
            REDIS_URL,
            decode_responses=False,  # Trả bytes cho embeddings
            max_connections=20,
        )
    return _redis_pool


async def close_redis():
    """Đóng Redis pool khi shutdown."""
    global _redis_pool
    if _redis_pool:
        await _redis_pool.aclose()
        _redis_pool = None


# ════════════════════════════════════════════════════════════
#  EMBEDDING SERIALIZATION — float32 array ↔ bytes
# ════════════════════════════════════════════════════════════

def embedding_to_bytes(embedding: list) -> bytes:
    """Chuyển embedding list[float] 512D → bytes (2048 bytes)."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def bytes_to_embedding(data: bytes) -> list:
    """Chuyển bytes → list[float]."""
    n = len(data) // 4  # float32 = 4 bytes
    return list(struct.unpack(f"{n}f", data))


# ════════════════════════════════════════════════════════════
#  CACHE OPERATIONS
# ════════════════════════════════════════════════════════════

async def load_class_embeddings_to_cache(
    course_id: int,
    students_data: list[dict],
) -> int:
    """
    Load embeddings của 1 lớp vào Redis cache.

    Args:
        course_id: ID lớp tín chỉ
        students_data: List[{
            "student_id": int,
            "student_code": str,
            "full_name": str,
            "photos": [{"photo_id": int, "embedding": list[float]}, ...]
        }]

    Returns:
        Số lượng embeddings đã cache.
    """
    r = await get_redis()
    emb_key = f"class:{course_id}:embeddings"
    meta_key = f"class:{course_id}:metadata"

    # Xóa cache cũ
    await r.delete(emb_key, meta_key)

    count = 0
    pipe = r.pipeline()

    for student in students_data:
        sid = student["student_id"]
        # Metadata
        pipe.hset(meta_key, str(sid), json.dumps({
            "student_code": student["student_code"],
            "full_name": student["full_name"],
        }))
        # Embeddings
        for photo in student["photos"]:
            field = f"sid:{sid}:pid:{photo['photo_id']}"
            pipe.hset(emb_key, field, embedding_to_bytes(photo["embedding"]))
            count += 1

    # Set TTL
    pipe.expire(emb_key, REDIS_CACHE_TTL)
    pipe.expire(meta_key, REDIS_CACHE_TTL)

    await pipe.execute()
    return count


async def get_cached_embeddings(course_id: int) -> Optional[dict]:
    """
    Lấy embeddings từ Redis cache.

    Returns:
        {
            "embeddings": np.ndarray (N x 512),
            "student_ids": list[int],   (N items, mỗi embedding thuộc SV nào)
            "photo_ids": list[int],
            "metadata": {student_id: {"student_code": ..., "full_name": ...}},
        }
        hoặc None nếu cache trống.
    """
    r = await get_redis()
    emb_key = f"class:{course_id}:embeddings"
    meta_key = f"class:{course_id}:metadata"

    # Lấy tất cả embeddings
    raw_embs = await r.hgetall(emb_key)
    if not raw_embs:
        return None

    # Lấy metadata
    raw_meta = await r.hgetall(meta_key)
    metadata = {}
    for sid_bytes, info_bytes in raw_meta.items():
        sid = int(sid_bytes.decode())
        metadata[sid] = json.loads(info_bytes.decode())

    # Parse embeddings
    embeddings = []
    student_ids = []
    photo_ids = []

    for field_bytes, emb_bytes in raw_embs.items():
        field = field_bytes.decode()  # "sid:1:pid:5"
        parts = field.split(":")
        sid = int(parts[1])
        pid = int(parts[3])
        emb = bytes_to_embedding(emb_bytes)

        embeddings.append(emb)
        student_ids.append(sid)
        photo_ids.append(pid)

    return {
        "embeddings": np.array(embeddings, dtype=np.float32),
        "student_ids": student_ids,
        "photo_ids": photo_ids,
        "metadata": metadata,
    }


async def mark_attended(course_id: int, student_id: int):
    """Đánh dấu SV đã điểm danh trong cache."""
    r = await get_redis()
    await r.sadd(f"class:{course_id}:attended", str(student_id))


async def is_attended(course_id: int, student_id: int) -> bool:
    """Kiểm tra SV đã điểm danh chưa (từ cache)."""
    r = await get_redis()
    return await r.sismember(f"class:{course_id}:attended", str(student_id))


async def get_attended_set(course_id: int) -> set[int]:
    """Lấy danh sách student_id đã điểm danh."""
    r = await get_redis()
    members = await r.smembers(f"class:{course_id}:attended")
    return {int(m.decode()) for m in members}


async def clear_class_cache(course_id: int):
    """Xóa toàn bộ cache của 1 lớp."""
    r = await get_redis()
    keys = [
        f"class:{course_id}:embeddings",
        f"class:{course_id}:metadata",
        f"class:{course_id}:attended",
    ]
    await r.delete(*keys)
