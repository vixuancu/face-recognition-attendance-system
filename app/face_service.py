"""
Xử lý nhận diện khuôn mặt — InsightFace buffalo_l (SCRFD + ArcFace, ONNX Runtime).

Pipeline:
  - Đăng ký SV: đọc file ảnh → detect + embed → lưu DB
  - Webcam nhận diện: nhận face crops từ MediaPipe → embed → match RAM cache

InsightFace buffalo_l:
  - SCRFD: face detector (~10ms/frame CPU)
  - ArcFace: embedding model 512D (~80ms/frame CPU)
  - Tổng: ~90–150ms, nhanh hơn DeepFace Facenet512+RetinaFace ~3–5x
"""

import logging
import os

# ── Giới hạn số Thread C++ (InsightFace/ONNXRuntime) để tránh chiếm hết CPU ──
# Để phần CPU cho OpenCV làm tác vụ giải mã TCP Stream
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import shutil
from typing import Optional

import cv2
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config import (
    IMAGES_DIR, INSIGHTFACE_MODEL,
    COSINE_THRESHOLD, FACE_CONFIDENCE_MIN, FACE_MIN_PIXELS,
    TOP_K_MATCHES, MATCH_MARGIN_MIN, STUDENT_AGG_TOP_N,
    QUALITY_FACE_SIZE_GOOD, QUALITY_FACE_SIZE_MIN, QUALITY_THRESHOLD_PENALTY,
)
from app.models import Student, StudentPhoto, AttendanceSession, AttendanceRecord

logger = logging.getLogger("face_service")

# ════════════════════════════════════════════════════════════
#  INSIGHTFACE SINGLETON
# ════════════════════════════════════════════════════════════

_face_app = None


def get_face_app():
    """
    Lấy InsightFace FaceAnalysis (singleton, load lần đầu ~3-5s).
    buffalo_l = SCRFD (detection) + ArcFace (recognition, 512D embedding).
    """
    global _face_app
    if _face_app is None:
        import insightface
        logger.info(f"[InsightFace] Loading model '{INSIGHTFACE_MODEL}'...")
        _face_app = insightface.app.FaceAnalysis(
            name=INSIGHTFACE_MODEL,
            providers=["CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info(f"[InsightFace] ✓ Model '{INSIGHTFACE_MODEL}' loaded (SCRFD + ArcFace)")
    return _face_app


# ════════════════════════════════════════════════════════════
#  FACE QUALITY SCORING  (dùng cho adaptive threshold)
# ════════════════════════════════════════════════════════════

def _compute_face_quality(original_face_size: int) -> float:
    """
    Tính điểm chất lượng khuôn mặt [0.0 → 1.0] dựa trên kích thước.

    - face >= QUALITY_FACE_SIZE_GOOD (120px) → 1.0
    - face <= QUALITY_FACE_SIZE_MIN  (40px)  → 0.0
    """
    if original_face_size >= QUALITY_FACE_SIZE_GOOD:
        return 1.0
    if original_face_size <= QUALITY_FACE_SIZE_MIN:
        return 0.0
    return (original_face_size - QUALITY_FACE_SIZE_MIN) / (
        QUALITY_FACE_SIZE_GOOD - QUALITY_FACE_SIZE_MIN
    )


def _adaptive_threshold(face_quality: float, base_threshold: float = COSINE_THRESHOLD) -> float:
    """
    Mặt tốt (quality=1.0) → threshold = base (0.55).
    Mặt kém (quality=0.0) → threshold = base + penalty.
    """
    penalty = (1.0 - face_quality) * QUALITY_THRESHOLD_PENALTY
    return base_threshold + penalty


# ════════════════════════════════════════════════════════════
#  TRÍCH XUẤT EMBEDDING — từ file ảnh (dùng khi đăng ký SV)
# ════════════════════════════════════════════════════════════

def extract_embedding(image_path: str) -> Optional[list]:
    """
    Trích xuất ArcFace embedding (512D) từ 1 file ảnh.
    Dùng khi đăng ký sinh viên (không cần realtime).

    Returns:
        list[float] 512D nếu thành công, None nếu không phát hiện mặt.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"[extract_embedding] Không đọc được ảnh: {image_path}")
            return None

        app = get_face_app()
        faces = app.get(img)

        if not faces:
            logger.warning(f"[extract_embedding] Không phát hiện mặt: {image_path}")
            return None

        # Lấy mặt có confidence cao nhất
        best = max(faces, key=lambda f: f.det_score)
        if best.det_score < FACE_CONFIDENCE_MIN:
            logger.warning(
                f"[extract_embedding] Confidence quá thấp "
                f"({best.det_score:.2f}): {image_path}"
            )
            return None

        return best.embedding.tolist()

    except Exception as e:
        logger.error(f"[extract_embedding] ERROR {image_path}: {e}")
        return None


# ════════════════════════════════════════════════════════════
#  TRÍCH XUẤT EMBEDDING — từ face crops (webcam /recognize-fast)
#
#  Client đã dùng MediaPipe để detect + crop mặt sẵn.
#  Server KHÔNG chạy lại detector SCRFD (sẽ fail vì crop quá chặt).
#  Thay vào đó, dùng thẳng ArcFace recognition model:
#    resize crop → 112×112 → ArcFace inference → embedding 512D.
# ════════════════════════════════════════════════════════════

_rec_model = None


def _get_rec_model():
    """Lấy ArcFace recognition model (không qua detection)."""
    global _rec_model
    if _rec_model is None:
        app = get_face_app()
        # app.models là dict: {'recognition': ArcFaceONNX, 'detection': SCRFD, ...}
        if 'recognition' in app.models:
            _rec_model = app.models['recognition']
            logger.info(f"[InsightFace] Got ArcFace rec model: {_rec_model.__class__.__name__}")
        else:
            raise RuntimeError(
                f"Không tìm thấy 'recognition' model! "
                f"Available: {list(app.models.keys())}"
            )
    return _rec_model


def extract_embeddings_from_crops(
    face_crops: list[np.ndarray],
) -> list[tuple[Optional[list], float]]:
    """
    Trích xuất ArcFace embeddings từ face crops đã detect ở client (MediaPipe).

    Strategy:
      1. Thêm padding 50% → app.get() → SCRFD detect + align + ArcFace
         (tốt nhất, có face alignment chuẩn → cosine ~80-95%)
      2. Nếu SCRFD vẫn fail → dùng thẳng ArcFace: resize 112×112 → embed
         (fallback, chất lượng thấp hơn vì không align)

    Args:
        face_crops: List numpy arrays (BGR uint8), mỗi cái là 1 mặt đã crop.

    Returns:
        List of (embedding, quality_score).
    """
    app = get_face_app()
    results = []

    for crop in face_crops:
        if crop is None or crop.size == 0:
            results.append((None, 0.0))
            continue

        try:
            h, w = crop.shape[:2]
            face_size = max(h, w)

            # ── Strategy 1: Thêm padding rồi dùng app.get() ──
            # Padding 50% mỗi bên → SCRFD có đủ context để detect
            # → alignment chuẩn → embedding tốt nhất
            pad_ratio = 0.5
            pad_h = int(h * pad_ratio)
            pad_w = int(w * pad_ratio)
            padded = cv2.copyMakeBorder(
                crop, pad_h, pad_h, pad_w, pad_w,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

            # Đảm bảo padded image đủ lớn cho SCRFD
            ph, pw = padded.shape[:2]
            if max(ph, pw) < 128:
                scale = 128 / max(ph, pw)
                padded = cv2.resize(padded, None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_CUBIC)

            faces = app.get(padded)

            if faces:
                best = max(faces, key=lambda f: f.det_score)
                quality = round(_compute_face_quality(face_size), 3)
                logger.debug(
                    f"[CROP] Strategy 1 OK — det_score={best.det_score:.2f} "
                    f"quality={quality} size={face_size}px"
                )
                results.append((best.embedding.tolist(), quality))
                continue

            # ── Strategy 2: Dùng thẳng ArcFace (bỏ qua detection) ──
            logger.info(
                f"[CROP] Strategy 1 FAIL (SCRFD no detect) — "
                f"falling back to direct ArcFace, size={face_size}px"
            )
            rec_model = _get_rec_model()

            # Resize crop về 112×112
            aligned = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_CUBIC)

            # ArcFace expects: (1, 3, 112, 112), float32, normalized
            blob = cv2.dnn.blobFromImage(
                aligned, 1.0 / 127.5, (112, 112),
                (127.5, 127.5, 127.5), swapRB=True
            )

            # Inference
            embedding = rec_model.session.run(
                rec_model.output_names,
                {rec_model.input_names[0]: blob}
            )[0][0]

            quality = round(_compute_face_quality(face_size), 3) * 0.7  # giảm quality vì không align
            results.append((embedding.tolist(), quality))

        except Exception as e:
            logger.error(f"[extract_embeddings_from_crops] Crop error: {e}")
            results.append((None, 0.0))

    return results


# ════════════════════════════════════════════════════════════
#  SO KHỚP KHUÔN MẶT — PGVECTOR (SQL-level cosine distance)
#  Dùng khi không có RAM cache (fallback hoặc register flow)
# ════════════════════════════════════════════════════════════

def find_best_match_pgvector_v2(
    db: Session,
    face_embedding: list,
    face_quality: float = 1.0,
    base_threshold: float = COSINE_THRESHOLD,
) -> tuple[Optional[dict], float, dict]:
    """
    So khớp khuôn mặt V2 qua pgvector — chống nhầm người.

    Cải tiến:
    1. TOP-K (20 results) thay vì TOP-1
    2. Aggregate per-student: trung bình top-3 ảnh/SV
    3. Margin check: SV #1 phải dẫn trước SV #2 >= 0.04
    4. Adaptive threshold: mặt nhỏ/mờ → threshold cao hơn

    Returns:
        (match_dict | None, score, debug_info)
    """
    vec_str = "[" + ",".join(str(float(x)) for x in face_embedding) + "]"

    sql = text("""
        SELECT
            s.id AS student_id,
            s.student_code,
            s.full_name,
            1 - (sp.face_embedding <=> :query_vec) AS cosine_sim
        FROM student_photos sp
        JOIN students s ON s.id = sp.student_id
        ORDER BY sp.face_embedding <=> :query_vec
        LIMIT :top_k
    """)

    rows = db.execute(sql, {"query_vec": vec_str, "top_k": TOP_K_MATCHES}).fetchall()
    if not rows:
        return None, 0.0, {"reason": "no_data"}

    # Aggregate per student
    students = {}
    for row in rows:
        sid = row.student_id
        if sid not in students:
            students[sid] = {
                "student_id":   sid,
                "student_code": row.student_code,
                "full_name":    row.full_name,
                "scores":       [],
            }
        students[sid]["scores"].append(float(row.cosine_sim))

    for info in students.values():
        scores = sorted(info["scores"], reverse=True)
        top_n = scores[:min(STUDENT_AGG_TOP_N, len(scores))]
        info["agg_score"] = sum(top_n) / len(top_n)
        info["max_score"] = scores[0]
        info["n_photos_matched"] = len(scores)

    ranked = sorted(students.values(), key=lambda x: x["agg_score"], reverse=True)
    best   = ranked[0]
    second = ranked[1] if len(ranked) >= 2 else None

    effective_threshold = _adaptive_threshold(face_quality, base_threshold)
    margin = best["agg_score"] - (second["agg_score"] if second else 0.0)

    debug_info = {
        "face_quality":       face_quality,
        "effective_threshold": round(effective_threshold, 3),
        "best_student":        best["full_name"],
        "best_agg_score":      round(best["agg_score"], 4),
        "best_max_score":      round(best["max_score"], 4),
        "best_n_photos":       best["n_photos_matched"],
        "second_student":      second["full_name"] if second else None,
        "second_agg_score":    round(second["agg_score"], 4) if second else 0,
        "margin":              round(margin, 4),
        "margin_required":     MATCH_MARGIN_MIN,
    }

    if best["agg_score"] < effective_threshold:
        debug_info["reason"] = f"below_threshold ({best['agg_score']:.3f} < {effective_threshold:.3f})"
        return None, 0.0, debug_info

    if second and margin < MATCH_MARGIN_MIN:
        debug_info["reason"] = f"margin_too_small ({margin:.4f} < {MATCH_MARGIN_MIN})"
        return None, 0.0, debug_info

    debug_info["reason"] = "matched"
    logger.info(
        f"[match_v2] OK — {best['full_name']} agg={best['agg_score']:.3f} "
        f"margin={margin:.3f} quality={face_quality:.2f} threshold={effective_threshold:.3f}"
    )
    return {
        "student_id":   best["student_id"],
        "student_code": best["student_code"],
        "full_name":    best["full_name"],
    }, best["agg_score"], debug_info


def find_best_match_pgvector_batch(
    db: Session,
    face_embeddings: list[list],
    face_qualities: list[float] = None,
    threshold: float = COSINE_THRESHOLD,
) -> list[tuple[Optional[dict], float, dict]]:
    """So khớp nhiều khuôn mặt qua pgvector V2."""
    if face_qualities is None:
        face_qualities = [1.0] * len(face_embeddings)

    results = []
    for emb, quality in zip(face_embeddings, face_qualities):
        match, score, debug = find_best_match_pgvector_v2(
            db, emb, face_quality=quality, base_threshold=threshold
        )
        results.append((match, score, debug))
    return results


# ════════════════════════════════════════════════════════════
#  ĐĂNG KÝ SINH VIÊN + UPLOAD ẢNH
# ════════════════════════════════════════════════════════════

def register_student(
    db: Session,
    student_code: str,
    full_name: str,
    class_name: str,
    photo_paths: list[str],
) -> Optional[Student]:
    """
    Đăng ký 1 sinh viên:
      1. Tạo record trong bảng students
      2. Copy ảnh vào Images/{student_code}/
      3. Trích xuất ArcFace embedding → lưu vào student_photos
    """
    exists = db.query(Student).filter(Student.student_code == student_code).first()
    if exists:
        print(f"[!] Mã SV '{student_code}' đã tồn tại trong DB.")
        return None

    student = Student(
        student_code=student_code,
        full_name=full_name,
        class_name=class_name,
    )
    db.add(student)
    db.flush()

    student_dir = os.path.join(IMAGES_DIR, student_code)
    os.makedirs(student_dir, exist_ok=True)

    success = 0
    for idx, src_path in enumerate(photo_paths, start=1):
        if not os.path.isfile(src_path):
            print(f"    [!] File không tồn tại: {src_path}")
            continue

        ext = os.path.splitext(src_path)[1]
        dest_filename = f"{student_code}_{idx}{ext}"
        dest_path = os.path.join(student_dir, dest_filename)
        shutil.copy2(src_path, dest_path)

        embedding = extract_embedding(dest_path)
        if embedding is None:
            print(f"    [!] Ảnh {idx}: Không phát hiện khuôn mặt → bỏ qua")
            os.remove(dest_path)
            continue

        relative_path = os.path.relpath(dest_path, os.path.dirname(IMAGES_DIR))
        photo = StudentPhoto(
            student_id=student.id,
            photo_path=relative_path,
            face_embedding=embedding,
        )
        db.add(photo)
        success += 1
        print(f"    [✓] Ảnh {idx}: OK")

    if success == 0:
        db.rollback()
        print(f"[✗] Không có ảnh nào hợp lệ. Hủy đăng ký.")
        return None

    db.commit()
    print(f"[✓] Đăng ký thành công: {full_name} ({student_code}) — {success}/{len(photo_paths)} ảnh")
    return student


# ════════════════════════════════════════════════════════════
#  LOAD EMBEDDINGS TỪ DB
# ════════════════════════════════════════════════════════════

def load_all_embeddings(db: Session) -> list[dict]:
    """Load tất cả face embedding từ DB lên RAM."""
    rows = (
        db.query(
            Student.id,
            Student.student_code,
            Student.full_name,
            StudentPhoto.face_embedding,
        )
        .join(StudentPhoto, Student.id == StudentPhoto.student_id)
        .all()
    )

    data = []
    for student_id, code, name, emb in rows:
        data.append({
            "student_id":   student_id,
            "student_code": code,
            "full_name":    name,
            "embedding":    emb,
        })
    print(f"[i] Đã load {len(data)} embeddings từ DB")
    return data


# ════════════════════════════════════════════════════════════
#  GHI ĐIỂM DANH
# ════════════════════════════════════════════════════════════

def save_attendance_record(
    db: Session,
    session_id: int,
    student_id: int,
    confidence: float,
) -> bool:
    """Ghi 1 record điểm danh. Trả về True nếu mới, False nếu đã điểm danh."""
    exists = (
        db.query(AttendanceRecord)
        .filter_by(session_id=session_id, student_id=student_id)
        .first()
    )
    if exists:
        return False

    record = AttendanceRecord(
        session_id=session_id,
        student_id=student_id,
        confidence=confidence,
    )
    db.add(record)
    db.commit()
    return True


def create_session(db: Session, session_name: str, subject: str) -> AttendanceSession:
    """Tạo 1 phiên điểm danh mới."""
    session = AttendanceSession(session_name=session_name, subject=subject)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

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
        face_embedding: Vector 512D từ DeepFace/InsightFace
        cache_data: Output của load_class_embeddings_to_cache()
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
