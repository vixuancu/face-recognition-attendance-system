"""
Router: Điểm danh — RAM cache + background DB write.

Flow:
  1. POST /start         → load embeddings SV của lớp lên RAM cache
  2. POST /recognize-fast → InsightFace embed crops từ MediaPipe → so khớp RAM cache → trả ngay
                           → background: ghi DB
  3. POST /stop          → clear RAM cache
"""

import time
import base64
import json
import logging
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.models import (
    Student, StudentPhoto, Course, CourseEnrollment,
    AttendanceSession, AttendanceRecord,
)
from app.schemas import (
    AttendanceStartRequest,
    AttendanceStartResponse,
    RecognizeResponse,
    FaceResult,
    AttendanceStatusResponse,
)
from app.face_service import extract_embeddings_from_crops
from app.ram_cache import (
    load_class_embeddings_to_cache,
    get_cached_embeddings,
    mark_attended,
    is_attended,
    get_attended_set,
    clear_class_cache,
)
from app.config import (
    COSINE_THRESHOLD,
    MATCH_MARGIN_MIN,
    STUDENT_AGG_TOP_N,
    QUALITY_THRESHOLD_PENALTY,
)

logger = logging.getLogger("attendance")

router = APIRouter(prefix="/api/attendance", tags=["Attendance"])

# Biến toàn cục cho phiên điểm danh hiện tại
# Trong production nên dùng Redis hoặc DB để lưu trạng thái
_active_sessions: dict = {}  # {session_id: {course_id, session_name, ...}}


# ════════════════════════════════════════════════════════════
#  BACKGROUND TASKS — Ghi DB bất đồng bộ
# ════════════════════════════════════════════════════════════

def _bg_save_attendance(session_id: int, student_id: int, confidence: float):
    """
    Background job: ghi attendance record vào PostgreSQL.
    Chạy trong thread riêng, không block response.
    """
    db = SessionLocal()
    try:
        # Kiểm tra đã tồn tại chưa
        exists = (
            db.query(AttendanceRecord)
            .filter_by(session_id=session_id, student_id=student_id)
            .first()
        )
        if exists:
            return

        record = AttendanceRecord(
            session_id=session_id,
            student_id=student_id,
            confidence=confidence,
        )
        db.add(record)
        db.commit()
        logger.info(f"[BG] Saved attendance: session={session_id} student={student_id} conf={confidence:.3f}")
    except Exception as e:
        logger.error(f"[BG] Failed to save attendance: {e}")
        db.rollback()
    finally:
        db.close()


# ════════════════════════════════════════════════════════════
#  MATCHING TRÊN CACHE (numpy vectorized)
# ════════════════════════════════════════════════════════════

def _match_from_cache(
    face_embedding: list,
    cache_data: dict,
    face_quality: float = 1.0,
) -> tuple[Optional[dict], float, dict]:
    """
    So khớp khuôn mặt với embeddings đang ở trên Redis cache.
    Dùng numpy vectorized cosine similarity (KHÔNG gọi DB).
    Logic V2: aggregate per-student → margin check → adaptive threshold.

    Args:
        face_embedding: Vector 512D từ DeepFace
        cache_data: Output của get_cached_embeddings()
        face_quality: Điểm chất lượng [0.0 → 1.0]

    Returns:
        (match_info, score, debug_info) hoặc (None, 0.0, debug_info)
    """
    emb_matrix = cache_data["embeddings"]       # (N, 512)
    student_ids = cache_data["student_ids"]      # List[int]
    metadata = cache_data["metadata"]            # {sid: {name, code}}

    if emb_matrix.size == 0:
        return None, 0.0, {"reason": "empty_cache"}

    # Vectorized cosine similarity
    query = np.array(face_embedding, dtype=np.float32)
    query_norm = query / (np.linalg.norm(query) + 1e-10)

    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_normed = emb_matrix / norms

    similarities = emb_normed @ query_norm  # (N,)

    # ── Aggregate per student ──
    students = {}
    for i, sim in enumerate(similarities):
        sid = student_ids[i]
        if sid not in students:
            meta = metadata.get(sid, {"student_code": "?", "full_name": "?"})
            students[sid] = {
                "student_id": sid,
                "student_code": meta["student_code"],
                "full_name": meta["full_name"],
                "scores": [],
            }
        students[sid]["scores"].append(float(sim))

    # Tính aggregated score
    for info in students.values():
        scores = sorted(info["scores"], reverse=True)
        top_n = scores[:min(STUDENT_AGG_TOP_N, len(scores))]
        info["agg_score"] = sum(top_n) / len(top_n)
        info["max_score"] = scores[0]
        info["n_photos_matched"] = len(scores)

    # Xếp hạng
    ranked = sorted(students.values(), key=lambda x: x["agg_score"], reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) >= 2 else None

    # ── Adaptive threshold ──
    penalty = (1.0 - face_quality) * QUALITY_THRESHOLD_PENALTY
    effective_threshold = COSINE_THRESHOLD + penalty

    # ── Margin check ──
    margin = best["agg_score"] - (second["agg_score"] if second else 0.0)

    debug_info = {
        "face_quality": face_quality,
        "effective_threshold": round(effective_threshold, 3),
        "best_student": best["full_name"],
        "best_agg_score": round(best["agg_score"], 4),
        "margin": round(margin, 4),
    }

    # ── Quyết định ──
    if best["agg_score"] < effective_threshold:
        debug_info["reason"] = "below_threshold"
        logger.warning(
            f"[MATCH] ✗ REJECT — best={best['full_name']} "
            f"agg={best['agg_score']:.4f} < threshold={effective_threshold:.3f} "
            f"max={best['max_score']:.4f} quality={face_quality:.2f} "
            f"n_photos={best['n_photos_matched']}"
        )
        return None, 0.0, debug_info

    if second and margin < MATCH_MARGIN_MIN:
        debug_info["reason"] = "margin_too_small"
        logger.warning(
            f"[MATCH] ✗ AMBIGUOUS — best={best['full_name']} "
            f"agg={best['agg_score']:.4f} vs second={second['full_name']} "
            f"agg={second['agg_score']:.4f} margin={margin:.4f}"
        )
        return None, 0.0, debug_info

    debug_info["reason"] = "matched"
    return {
        "student_id": best["student_id"],
        "student_code": best["student_code"],
        "full_name": best["full_name"],
    }, best["agg_score"], debug_info


# ════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ════════════════════════════════════════════════════════════

@router.post("/start", response_model=AttendanceStartResponse)
async def start_attendance(data: AttendanceStartRequest, db: Session = Depends(get_db)):
    """
    Bắt đầu phiên điểm danh cho 1 lớp tín chỉ.
    → Load embeddings sinh viên của lớp đó lên Redis cache.
    """
    # Kiểm tra lớp tồn tại
    course = db.query(Course).filter(Course.id == data.course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Không tìm thấy lớp tín chỉ!")

    # Lấy danh sách SV đăng ký lớp
    enrollments = (
        db.query(CourseEnrollment)
        .filter(CourseEnrollment.course_id == data.course_id)
        .all()
    )
    if not enrollments:
        raise HTTPException(status_code=400, detail="Lớp chưa có sinh viên nào!")

    student_ids = [e.student_id for e in enrollments]

    # Load embeddings từ DB
    students_data = []
    for sid in student_ids:
        student = db.query(Student).filter(Student.id == sid).first()
        if not student:
            continue

        photos = db.query(StudentPhoto).filter(StudentPhoto.student_id == sid).all()
        if not photos:
            continue

        students_data.append({
            "student_id": sid,
            "student_code": student.student_code,
            "full_name": student.full_name,
            "photos": [
                {"photo_id": p.id, "embedding": list(p.face_embedding)}
                for p in photos
            ],
        })

    # Load lên RAM cache
    cached_count = load_class_embeddings_to_cache(data.course_id, students_data)

    # Tạo attendance session trong DB
    session_name = data.session_name or f"{course.course_name} - {course.room}"
    subject = data.subject or course.course_name

    att_session = AttendanceSession(
        session_name=session_name,
        subject=subject,
        course_id=data.course_id,
    )
    db.add(att_session)
    db.commit()
    db.refresh(att_session)

    # Lưu trạng thái phiên
    _active_sessions[att_session.id] = {
        "course_id": data.course_id,
        "session_name": session_name,
        "course_code": course.course_code,
    }

    logger.info(
        f"[START] Session #{att_session.id} — {session_name} — "
        f"{len(students_data)} students, {cached_count} embeddings cached"
    )

    return AttendanceStartResponse(
        status="success",
        session_id=att_session.id,
        course_id=data.course_id,
        session_name=session_name,
        total_students=len(students_data),
        cached_embeddings=cached_count,
        message=f"Đã load {cached_count} embeddings của {len(students_data)} SV lên cache",
    )


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize_face(
    background_tasks: BackgroundTasks,
    session_id: int = Form(...),
    image: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
):
    """
    Nhận diện khuôn mặt từ frame camera.

    Flow:
      1. DeepFace extract embeddings từ frame
      2. So khớp trên Redis cache (numpy vectorized) — NHANH
      3. Trả response ngay cho client
      4. Background: ghi attendance record vào PostgreSQL
    """
    # Validate session
    if session_id not in _active_sessions:
        raise HTTPException(status_code=400, detail="Phiên điểm danh không hợp lệ hoặc đã kết thúc!")

    session_info = _active_sessions[session_id]
    course_id = session_info["course_id"]

    # ── Đọc ảnh ──
    img_bytes = None
    if image:
        img_bytes = await image.read()
    elif image_base64:
        b64 = image_base64
        if "," in b64:
            b64 = b64.split(",")[1]
        img_bytes = base64.b64decode(b64)

    if not img_bytes:
        raise HTTPException(status_code=400, detail="Không có ảnh!")

    # Convert bytes → numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Ảnh không hợp lệ!")

    t0 = time.time()

    # ── Extract embeddings từ frame (full pipeline — dùng InsightFace) ──
    from app.face_service import get_face_app
    import insightface  # noqa
    _app = get_face_app()
    insight_faces = _app.get(frame)
    if not insight_faces:
        return RecognizeResponse(
            status="no_face",
            total_faces=0,
            total_attended=len(get_attended_set(course_id)),
        )
    # Convert sang format (embedding, facial_area, quality)
    from app.face_service import _compute_face_quality, _adaptive_threshold
    faces = []
    for f in insight_faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        size = max(x2 - x1, y2 - y1)
        quality = _compute_face_quality(size)
        faces.append((f.embedding.tolist(), {"x": x1, "y": y1, "w": x2-x1, "h": y2-y1}, quality))

    # ── Lấy cache từ RAM ──
    cache_data = get_cached_embeddings(course_id)
    if not cache_data:
        raise HTTPException(status_code=500, detail="Cache trống! Hãy bắt đầu lại phiên điểm danh.")

    # ── So khớp từng khuôn mặt trên cache ──
    face_results = []
    new_attended = []

    for emb, facial_area, quality in faces:
        face_box = {
            "x": facial_area.get("x", 0),
            "y": facial_area.get("y", 0),
            "w": facial_area.get("w", 0),
            "h": facial_area.get("h", 0),
        }

        match, score, debug = _match_from_cache(emb, cache_data, face_quality=quality)

        if match:
            student_id = match["student_id"]
            already = is_attended(course_id, student_id)

            if not already:
                # Đánh dấu đã điểm danh trên RAM cache (ngay lập tức)
                mark_attended(course_id, student_id)

                # Background: ghi vào PostgreSQL
                background_tasks.add_task(
                    _bg_save_attendance, session_id, student_id, score
                )

                new_attended.append({
                    "full_name": match["full_name"],
                    "student_code": match["student_code"],
                    "confidence": round(score * 100, 1),
                })

                logger.info(
                    f"[RECOGNIZE] ✓ {match['full_name']} ({match['student_code']}) "
                    f"score={score:.3f} quality={quality:.2f}"
                )

            face_results.append(FaceResult(
                recognized=True,
                student_code=match["student_code"],
                full_name=match["full_name"],
                confidence=round(score * 100, 1),
                already_marked=already,
                face_box=face_box,
            ))
        else:
            face_results.append(FaceResult(
                recognized=False,
                face_box=face_box,
            ))

    elapsed = (time.time() - t0) * 1000
    attended_set = get_attended_set(course_id)

    logger.info(f"[RECOGNIZE] {elapsed:.0f}ms — {len(faces)} face(s), {len(new_attended)} new")

    return RecognizeResponse(
        status="success",
        faces=face_results,
        total_faces=len(face_results),
        new_attended=new_attended,
        total_attended=len(attended_set),
        elapsed_ms=round(elapsed, 1),
    )


@router.post("/recognize-fast", response_model=RecognizeResponse)
async def recognize_fast(
    background_tasks: BackgroundTasks,
    session_id: int = Form(...),
    faces: list[UploadFile] = File(...),
    face_positions: str = Form("[]"),
):
    """
    CPU-optimized: nhận diện từ face crops đã được MediaPipe detect ở client.

    Flow:
      1. Client (MediaPipe) detect mặt 30fps → crop + padding
      2. Gửi face crops (~200x200 mỗi cái) thay vì full frame (640x480)
      3. Server: opencv align trên crop nhỏ (~5ms) + Facenet512 embed (~300ms)
      4. Redis cache match (numpy vectorized ~1ms)
      5. Background: ghi DB PostgreSQL

    So với /recognize (full frame):
      - Bỏ RetinaFace (~1500ms) → tiết kiệm ~80% CPU
      - Bỏ Upscale/CLAHE/Sharpen → tiết kiệm thêm ~200ms
      - Tổng: ~350ms thay vì ~2600ms
    """
    # Validate session
    if session_id not in _active_sessions:
        raise HTTPException(status_code=400, detail="Phiên không hợp lệ!")

    session_info = _active_sessions[session_id]
    course_id = session_info["course_id"]

    t0 = time.time()

    # Parse face positions từ MediaPipe
    try:
        positions = json.loads(face_positions)
    except (json.JSONDecodeError, TypeError):
        positions = []

    # Đọc face crops từ request
    crops = []
    for face_file in faces:
        img_bytes = await face_file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        crop = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        crops.append(crop)

    if not crops:
        return RecognizeResponse(
            status="no_face",
            total_faces=0,
            total_attended=len(get_attended_set(course_id)),
        )

    # Extract embeddings từ crops (InsightFace ArcFace)
    embeddings_with_quality = extract_embeddings_from_crops(crops)

    # DEBUG: log kết quả extract
    for idx, (emb, q) in enumerate(embeddings_with_quality):
        if emb is None:
            logger.warning(f"[FAST] Crop #{idx}: ✗ InsightFace không detect được mặt")
        else:
            logger.info(f"[FAST] Crop #{idx}: ✓ embedding OK, quality={q:.2f}, emb_len={len(emb)}")

    # Lấy cache từ RAM
    cache_data = get_cached_embeddings(course_id)
    if not cache_data:
        raise HTTPException(status_code=500, detail="Cache trống! Hãy bắt đầu lại phiên.")

    # So khớp từng face trên cache
    face_results = []
    new_attended = []

    for i, (emb, quality) in enumerate(embeddings_with_quality):
        pos = positions[i] if i < len(positions) else {}
        face_box = {
            "xCenter": pos.get("xCenter", 0),
            "yCenter": pos.get("yCenter", 0),
        }

        if emb is None:
            face_results.append(FaceResult(recognized=False, face_box=face_box))
            continue

        match, score, debug = _match_from_cache(emb, cache_data, face_quality=quality)

        if match:
            student_id = match["student_id"]
            already = is_attended(course_id, student_id)

            if not already:
                mark_attended(course_id, student_id)
                background_tasks.add_task(_bg_save_attendance, session_id, student_id, score)
                new_attended.append({
                    "full_name": match["full_name"],
                    "student_code": match["student_code"],
                    "confidence": round(score * 100, 1),
                })
                logger.info(
                    f"[FAST] ✓ {match['full_name']} ({match['student_code']}) "
                    f"score={score:.3f} quality={quality:.2f}"
                )

            face_results.append(FaceResult(
                recognized=True,
                student_code=match["student_code"],
                full_name=match["full_name"],
                confidence=round(score * 100, 1),
                already_marked=already,
                face_box=face_box,
            ))
        else:
            face_results.append(FaceResult(recognized=False, face_box=face_box))

    elapsed = (time.time() - t0) * 1000
    attended_set = get_attended_set(course_id)

    logger.info(f"[FAST] {elapsed:.0f}ms — {len(crops)} crop(s), {len(new_attended)} new")

    return RecognizeResponse(
        status="success",
        faces=face_results,
        total_faces=len(face_results),
        new_attended=new_attended,
        total_attended=len(attended_set),
        elapsed_ms=round(elapsed, 1),
    )


@router.get("/status", response_model=AttendanceStatusResponse)
async def attendance_status(session_id: int):
    """Trạng thái phiên điểm danh hiện tại."""
    if session_id not in _active_sessions:
        return AttendanceStatusResponse()

    info = _active_sessions[session_id]
    course_id = info["course_id"]
    attended_ids = get_attended_set(course_id)

    # Lấy metadata từ cache
    cache_data = get_cached_embeddings(course_id)
    attended_list = []
    if cache_data:
        for sid in attended_ids:
            meta = cache_data["metadata"].get(sid, {})
            attended_list.append({
                "student_id": sid,
                "student_code": meta.get("student_code", "?"),
                "full_name": meta.get("full_name", "?"),
            })

    return AttendanceStatusResponse(
        session_id=session_id,
        session_name=info["session_name"],
        course_id=course_id,
        attended=attended_list,
        total=len(attended_ids),
    )


@router.post("/stop")
async def stop_attendance(session_id: int = Form(...)):
    """Kết thúc phiên điểm danh → clear Redis cache."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=400, detail="Phiên không tồn tại!")

    info = _active_sessions.pop(session_id)
    course_id = info["course_id"]

    # Lấy kết quả trước khi xóa cache
    attended_ids = get_attended_set(course_id)

    # Clear RAM cache
    clear_class_cache(course_id)

    logger.info(f"[STOP] Session #{session_id} — {len(attended_ids)} attended")

    return {
        "status": "success",
        "session_id": session_id,
        "total_attended": len(attended_ids),
        "message": f"Đã kết thúc phiên. {len(attended_ids)} SV đã điểm danh.",
    }


@router.get("/sessions")
def list_sessions(db: Session = Depends(get_db)):
    """Lịch sử tất cả phiên điểm danh."""
    sessions = db.query(AttendanceSession).order_by(AttendanceSession.id.desc()).all()
    result = []
    for s in sessions:
        records = db.query(AttendanceRecord).filter_by(session_id=s.id).all()
        record_details = []
        for r in records:
            student = db.query(Student).filter_by(id=r.student_id).first()
            if student:
                record_details.append({
                    "student_code": student.student_code,
                    "full_name": student.full_name,
                    "time": r.check_in_time.strftime("%H:%M:%S") if r.check_in_time else "",
                    "confidence": round(r.confidence * 100, 1) if r.confidence else 0,
                })

        course = db.query(Course).filter_by(id=s.course_id).first() if s.course_id else None
        result.append({
            "id": s.id,
            "name": s.session_name,
            "subject": s.subject,
            "date": s.session_date.strftime("%d/%m/%Y") if s.session_date else "",
            "course_code": course.course_code if course else None,
            "total": len(records),
            "records": record_details,
        })
    return result


@router.get("/active")
async def list_active_sessions():
    """Danh sách phiên điểm danh đang hoạt động."""
    result = []
    for sid, info in _active_sessions.items():
        course_id = info["course_id"]
        attended = get_attended_set(course_id)
        result.append({
            "session_id": sid,
            "course_id": course_id,
            "session_name": info["session_name"],
            "total_attended": len(attended),
        })
    return result
