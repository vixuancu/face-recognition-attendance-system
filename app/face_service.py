"""Xử lý nhận diện khuôn mặt: trích xuất embedding + so khớp."""

import os
import shutil
from typing import Optional

import cv2
import numpy as np
from deepface import DeepFace
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config import (
    IMAGES_DIR, FACE_MODEL, DETECTOR_BACKEND, COSINE_THRESHOLD,
    FRAME_UPSCALE_TARGET, FACE_CONFIDENCE_MIN, FACE_MIN_PIXELS,
    TOP_K_MATCHES, MATCH_MARGIN_MIN, STUDENT_AGG_TOP_N,
    QUALITY_FACE_SIZE_GOOD, QUALITY_FACE_SIZE_MIN, QUALITY_THRESHOLD_PENALTY,
)
from app.models import Student, StudentPhoto, AttendanceSession, AttendanceRecord


# ════════════════════════════════════════════════════════════
#  TRÍCH XUẤT EMBEDDING
# ════════════════════════════════════════════════════════════

def extract_embedding(image_path: str) -> Optional[list]:
    """
    Trích xuất face embedding (vector 512 chiều) từ 1 ảnh.

    Sử dụng enforce_detection=False để tránh crash khi detector yếu.
    Kiểm tra confidence >= 0.5 để đảm bảo ảnh hợp lệ.

    Returns:
        list[float] nếu thành công, None nếu thất bại.
    """
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name=FACE_MODEL,
            enforce_detection=False,
            detector_backend=DETECTOR_BACKEND,
        )
        if result and len(result) > 0:
            face = result[0]
            confidence = face.get("face_confidence", 0)
            if confidence < 0.5:
                print(f"    [!] Ảnh {image_path}: confidence quá thấp ({confidence:.2f}) → bỏ qua")
                return None
            return face["embedding"]
        return None
    except Exception as e:
        print(f"    [!] Không trích xuất được embedding từ {image_path}: {e}")
        return None


def extract_embedding_from_frame(
    frame, detector: str = DETECTOR_BACKEND
) -> tuple[Optional[list], Optional[dict]]:
    """
    Trích xuất face embedding từ 1 frame camera (numpy array).
    Chỉ trả về khuôn mặt ĐẦU TIÊN (backward-compatible).

    Returns:
        (embedding, facial_area) hoặc (None, None)
    """
    faces = extract_all_embeddings_from_frame(frame, detector)
    if faces:
        return faces[0]
    return None, None


def extract_all_embeddings_from_frame(
    frame, detector: str = DETECTOR_BACKEND
) -> list[tuple[list, dict]]:
    """
    Trích xuất face embedding cho TẤT CẢ khuôn mặt trong frame.

    Returns:
        List of (embedding, facial_area) cho mỗi khuôn mặt hợp lệ.
        facial_area = {"x": int, "y": int, "w": int, "h": int}
    """
    try:
        if frame is None:
            return []

        result = DeepFace.represent(
            img_path=frame,
            model_name=FACE_MODEL,
            enforce_detection=False,
            detector_backend=detector,
        )
        if not result:
            return []

        faces = []
        for face in result:
            confidence = face.get("face_confidence", 0)
            if confidence < 0.5:
                continue
            facial_area = face.get("facial_area", None)
            embedding = face.get("embedding", None)
            if embedding and facial_area:
                faces.append((embedding, facial_area))

        return faces
    except Exception as e:
        print(f"    [!] extract_all_embeddings_from_frame ERROR: {e}")
        return []


# ════════════════════════════════════════════════════════════
#  ENHANCED PIPELINE — KHOẢNG CÁCH XA (3–5m)
#
#  Vấn đề: camera ở xa → mặt nhỏ (~30-80px) → SSD miss,
#           embedding chất lượng thấp, ánh sáng không đều.
#
#  Pipeline 5 bước:
#    1. Upscale frame (2x) → mặt nhỏ trở nên lớn hơn
#    2. CLAHE toàn frame → cân bằng ánh sáng
#    3. RetinaFace detect → tốt nhất cho mặt nhỏ (tới ~20px)
#    4. Crop + enhance riêng từng mặt → sharpen chi tiết
#    5. Embed với detector="skip" → không detect lại
# ════════════════════════════════════════════════════════════

def _upscale_frame(frame: np.ndarray, target_max: int = FRAME_UPSCALE_TARGET):
    """
    Upscale frame nếu nhỏ hơn target — giúp detect khuôn mặt xa.
    Ví dụ: webcam 640x480 → upscale lên 1280x960 (2x).
    """
    h, w = frame.shape[:2]
    if max(h, w) >= target_max:
        return frame, 1.0
    scale = target_max / max(h, w)
    scale = min(scale, 2.5)  # Giới hạn tối đa 2.5x
    upscaled = cv2.resize(
        frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )
    return upscaled, scale


def _enhance_frame_global(frame: np.ndarray) -> np.ndarray:
    """
    CLAHE trên toàn frame — cân bằng contrast/ánh sáng.
    Hữu ích khi camera xa, ánh sáng phòng không đều.
    Input: BGR (OpenCV).
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _enhance_face_crop(face_img: np.ndarray) -> np.ndarray:
    """
    Sharpen chi tiết khuôn mặt — cải thiện embedding cho mặt xa/nhỏ.
    Dùng unsharp mask — hoạt động bất kể thứ tự kênh màu (BGR/RGB).
    """
    gaussian = cv2.GaussianBlur(face_img, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(face_img, 1.5, gaussian, -0.5, 0)
    return sharpened


def _compute_face_quality(face_img: np.ndarray, original_face_size: int) -> float:
    """
    Tính điểm chất lượng khuôn mặt [0.0 → 1.0].

    Dựa trên 2 yếu tố:
    - Kích thước mặt trong frame gốc (trước upscale)
    - Độ sắc nét (Laplacian variance)

    quality thấp → cần threshold cao hơn để tránh nhầm.
    """
    # 1. Size factor: face < 40px → 0.0, face >= 120px → 1.0
    if original_face_size >= QUALITY_FACE_SIZE_GOOD:
        size_factor = 1.0
    elif original_face_size <= QUALITY_FACE_SIZE_MIN:
        size_factor = 0.0
    else:
        size_factor = (original_face_size - QUALITY_FACE_SIZE_MIN) / (
            QUALITY_FACE_SIZE_GOOD - QUALITY_FACE_SIZE_MIN
        )

    # 2. Sharpness factor: Laplacian variance (normalized)
    if face_img is not None and face_img.size > 0:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # lap_var < 30 → rất mờ, > 200 → rất nét
        sharpness_factor = min(1.0, max(0.0, (lap_var - 20) / 180.0))
    else:
        sharpness_factor = 0.5

    # Combined: 60% size, 40% sharpness
    quality = 0.6 * size_factor + 0.4 * sharpness_factor
    return round(quality, 3)


def _adaptive_threshold(face_quality: float, base_threshold: float = COSINE_THRESHOLD) -> float:
    """
    Tính ngưỡng cosine similarity dựa trên chất lượng khuôn mặt.

    Mặt tốt (quality=1.0) → threshold = base (0.55)
    Mặt kém (quality=0.0) → threshold = base + penalty (0.65)

    Nghĩa là: mặt ở xa/mờ cần score CAO HƠN mới được match.
    """
    penalty = (1.0 - face_quality) * QUALITY_THRESHOLD_PENALTY
    return base_threshold + penalty


def extract_all_embeddings_enhanced(
    frame, detector: str = DETECTOR_BACKEND
) -> list[tuple[list, dict, float]]:
    """
    Pipeline nhận diện khuôn mặt tối ưu cho KHOẢNG CÁCH XA (3–5m).

    So với extract_all_embeddings_from_frame (chỉ gọi represent 1 lần):
    - Tách detection và embedding thành 2 bước riêng
    - Upscale frame trước detection → detect được mặt nhỏ hơn
    - Enhance từng face crop riêng → embedding chất lượng cao hơn
    - Dùng RetinaFace (không SSD) → tốt nhất cho mặt nhỏ
    - Tính face quality score → adaptive threshold chống nhầm

    Returns:
        List of (embedding, facial_area, quality_score).
        facial_area tọa độ TRÊN FRAME GỐC (đã map ngược từ upscale).
        quality_score: float [0.0 → 1.0] — chất lượng khuôn mặt.
    """
    try:
        if frame is None:
            return []

        h, w = frame.shape[:2]

        # ── Bước 1: Upscale frame ──
        frame_up, scale = _upscale_frame(frame)

        # ── Bước 2: CLAHE toàn frame (frame là BGR từ OpenCV) ──
        frame_enhanced = _enhance_frame_global(frame_up)

        # ── Bước 3: Detect faces bằng RetinaFace ──
        # extract_faces trả về: face (aligned, RGB float [0,1]), facial_area, confidence
        detected = DeepFace.extract_faces(
            img_path=frame_enhanced,
            detector_backend=detector,
            enforce_detection=False,
            align=True,
        )

        if not detected:
            return []

        faces = []
        for face_info in detected:
            confidence = face_info.get(
                "confidence", face_info.get("face_confidence", 0)
            )
            if confidence < FACE_CONFIDENCE_MIN:
                continue

            facial_area = face_info.get("facial_area", {})
            face_img = face_info.get("face", None)

            if face_img is None:
                continue

            # ── Bước 4: Enhance face crop ──
            # extract_faces trả về float32 [0,1] → chuyển sang uint8
            if face_img.dtype != np.uint8:
                face_uint8 = (face_img * 255).clip(0, 255).astype(np.uint8)
            else:
                face_uint8 = face_img.copy()

            # Upscale face nếu quá nhỏ (< 160px)
            fh, fw = face_uint8.shape[:2]
            if max(fh, fw) < FACE_MIN_PIXELS:
                face_scale = FACE_MIN_PIXELS / max(fh, fw)
                face_uint8 = cv2.resize(
                    face_uint8, None,
                    fx=face_scale, fy=face_scale,
                    interpolation=cv2.INTER_CUBIC,
                )

            # Sharpen chi tiết
            face_uint8 = _enhance_face_crop(face_uint8)

            # ── Tính face quality score ──
            # Kích thước mặt trong frame GỐC (trước upscale)
            orig_face_size = int(max(
                facial_area.get("w", 0), facial_area.get("h", 0)
            ) / scale)
            face_quality = _compute_face_quality(face_uint8, orig_face_size)

            # ── Bước 5: Extract embedding (skip detector — đã aligned) ──
            try:
                emb_result = DeepFace.represent(
                    img_path=face_uint8,
                    model_name=FACE_MODEL,
                    enforce_detection=False,
                    detector_backend="skip",
                )
                if emb_result and len(emb_result) > 0:
                    embedding = emb_result[0]["embedding"]
                    # Map tọa độ về frame gốc
                    orig_area = {
                        "x": int(facial_area.get("x", 0) / scale),
                        "y": int(facial_area.get("y", 0) / scale),
                        "w": int(facial_area.get("w", 0) / scale),
                        "h": int(facial_area.get("h", 0) / scale),
                    }
                    faces.append((embedding, orig_area, face_quality))
                    print(
                        f"    [face] size={orig_face_size}px "
                        f"quality={face_quality:.2f} "
                        f"confidence={confidence:.2f}"
                    )
            except Exception as e:
                print(f"    [!] Enhanced embed error: {e}")
                continue

        print(
            f"    [enhance] {w}x{h} → {frame_up.shape[1]}x{frame_up.shape[0]}"
            f" (scale {scale:.1f}x) | {len(detected)} detected → {len(faces)} valid"
        )
        return faces

    except ValueError as ve:
        print(f"    [!] extract_all_embeddings_enhanced ValueError: {ve}")
        return []

    except Exception as e:
        print(f"    [!] extract_all_embeddings_enhanced ERROR: {e}")
        return []


# ════════════════════════════════════════════════════════════
#  SO KHỚP KHUÔN MẶT — PGVECTOR (SQL-level cosine distance)
# ════════════════════════════════════════════════════════════

def find_best_match_pgvector(
    db: Session,
    face_embedding: list,
    threshold: float = COSINE_THRESHOLD,
) -> tuple[Optional[dict], float]:
    """
    Legacy: Tìm sinh viên khớp nhất bằng TOP-1 đơn giản.
    Dùng cho backward-compatible. Nên dùng find_best_match_pgvector_v2.
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
        LIMIT 1
    """)

    row = db.execute(sql, {"query_vec": vec_str}).fetchone()
    if row is None:
        return None, 0.0

    score = float(row.cosine_sim)
    if score >= threshold:
        return {
            "student_id": row.student_id,
            "student_code": row.student_code,
            "full_name": row.full_name,
        }, score

    return None, 0.0


def find_best_match_pgvector_v2(
    db: Session,
    face_embedding: list,
    face_quality: float = 1.0,
    base_threshold: float = COSINE_THRESHOLD,
) -> tuple[Optional[dict], float, dict]:
    """
    So khớp khuôn mặt V2 — chống nhầm người ở khoảng cách xa.

    Cải tiến so với v1:
    1. TOP-K (20 results) thay vì TOP-1
    2. Aggregate per-student: trung bình top-3 ảnh/SV → ổn định hơn max
    3. Margin check: SV #1 phải dẫn trước SV #2 đủ xa (>= 0.04)
    4. Adaptive threshold: mặt nhỏ/mờ → threshold cao hơn

    Args:
        face_embedding: Vector 512 chiều
        face_quality: Điểm chất lượng [0.0 → 1.0]
        base_threshold: Ngưỡng cơ bản (mặc định 0.55)

    Returns:
        (match_dict | None, score, debug_info)
        debug_info chứa thông tin chi tiết để debug/logging.
    """
    vec_str = "[" + ",".join(str(float(x)) for x in face_embedding) + "]"

    # Lấy TOP-K kết quả
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

    # ── Aggregate per student ──
    students = {}
    for row in rows:
        sid = row.student_id
        if sid not in students:
            students[sid] = {
                "student_id": sid,
                "student_code": row.student_code,
                "full_name": row.full_name,
                "scores": [],
            }
        students[sid]["scores"].append(float(row.cosine_sim))

    # Tính aggregated score cho mỗi SV: trung bình top-N ảnh
    for info in students.values():
        scores = sorted(info["scores"], reverse=True)
        top_n = scores[:min(STUDENT_AGG_TOP_N, len(scores))]
        info["agg_score"] = sum(top_n) / len(top_n)
        info["max_score"] = scores[0]
        info["n_photos_matched"] = len(scores)

    # Xếp hạng theo agg_score
    ranked = sorted(students.values(), key=lambda x: x["agg_score"], reverse=True)

    best = ranked[0]
    second = ranked[1] if len(ranked) >= 2 else None

    # ── Adaptive threshold ──
    effective_threshold = _adaptive_threshold(face_quality, base_threshold)

    # ── Margin check ──
    margin = best["agg_score"] - (second["agg_score"] if second else 0.0)

    debug_info = {
        "face_quality": face_quality,
        "effective_threshold": round(effective_threshold, 3),
        "best_student": best["full_name"],
        "best_agg_score": round(best["agg_score"], 4),
        "best_max_score": round(best["max_score"], 4),
        "best_n_photos": best["n_photos_matched"],
        "second_student": second["full_name"] if second else None,
        "second_agg_score": round(second["agg_score"], 4) if second else 0,
        "margin": round(margin, 4),
        "margin_required": MATCH_MARGIN_MIN,
    }

    # ── Quyết định ──
    # Điều kiện 1: Score phải vượt threshold
    if best["agg_score"] < effective_threshold:
        debug_info["reason"] = f"below_threshold ({best['agg_score']:.3f} < {effective_threshold:.3f})"
        print(f"    [match_v2] REJECT — {debug_info['reason']}")
        return None, 0.0, debug_info

    # Điều kiện 2: Margin phải đủ lớn (nếu có >1 SV)
    if second and margin < MATCH_MARGIN_MIN:
        debug_info["reason"] = f"margin_too_small ({margin:.4f} < {MATCH_MARGIN_MIN})"
        print(
            f"    [match_v2] REJECT — margin {margin:.4f} < {MATCH_MARGIN_MIN}  "
            f"| #{1} {best['full_name']}={best['agg_score']:.3f} "
            f"| #{2} {second['full_name']}={second['agg_score']:.3f}"
        )
        return None, 0.0, debug_info

    # ── Match thành công ──
    debug_info["reason"] = "matched"
    print(
        f"    [match_v2] OK — {best['full_name']} agg={best['agg_score']:.3f} "
        f"(max={best['max_score']:.3f}, {best['n_photos_matched']}photos) "
        f"margin={margin:.3f} quality={face_quality:.2f} threshold={effective_threshold:.3f}"
    )
    return {
        "student_id": best["student_id"],
        "student_code": best["student_code"],
        "full_name": best["full_name"],
    }, best["agg_score"], debug_info


def find_best_match_pgvector_batch(
    db: Session,
    face_embeddings: list[list],
    face_qualities: list[float] = None,
    threshold: float = COSINE_THRESHOLD,
) -> list[tuple[Optional[dict], float, dict]]:
    """
    So khớp NHIỀU khuôn mặt bằng pgvector V2 — chống nhầm người.

    Args:
        face_embeddings: List embedding vectors
        face_qualities: List quality scores [0-1] cho mỗi face (cùng thứ tự)
        threshold: Base threshold

    Returns:
        List of (match_dict, score, debug_info) cho mỗi face.
    """
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
#  LEGACY — Các hàm cũ (giữ lại cho CLI / backward-compatible)
# ════════════════════════════════════════════════════════════

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Tính cosine similarity giữa 2 vector."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def find_best_match(
    face_embedding: list,
    all_embeddings: list[dict],
    threshold: float = COSINE_THRESHOLD,
) -> tuple[Optional[dict], float]:
    """Legacy: so khớp bằng Python loop (dùng cho CLI)."""
    best_match = None
    best_score = -1.0

    for record in all_embeddings:
        score = cosine_similarity(face_embedding, record["embedding"])
        if score > best_score:
            best_score = score
            best_match = record

    if best_score >= threshold:
        return best_match, best_score
    return None, 0.0


def precompute_embedding_matrix(all_embeddings: list[dict]) -> np.ndarray:
    """Legacy: chuyển embeddings thành numpy matrix (cho CLI)."""
    if not all_embeddings:
        return np.array([], dtype=np.float32)

    matrix = np.array([e["embedding"] for e in all_embeddings], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def find_best_match_fast_batch(
    face_embeddings: list[list],
    embedding_matrix: np.ndarray,
    all_embeddings: list[dict],
    threshold: float = COSINE_THRESHOLD,
) -> list[tuple[Optional[dict], float]]:
    """Legacy: numpy batch matching (cho CLI)."""
    if embedding_matrix.size == 0 or not face_embeddings:
        return [(None, 0.0)] * len(face_embeddings)

    query_matrix = np.array(face_embeddings, dtype=np.float32)
    norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    query_matrix = query_matrix / norms

    sim_matrix = query_matrix @ embedding_matrix.T

    results = []
    for i in range(sim_matrix.shape[0]):
        best_idx = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i][best_idx])
        if best_score >= threshold:
            results.append((all_embeddings[best_idx], best_score))
        else:
            results.append((None, 0.0))

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
      3. Trích xuất embedding → lưu vào student_photos

    Args:
        photo_paths: Danh sách đường dẫn tuyệt đối đến ảnh gốc.
    """
    # Kiểm tra trùng mã SV
    exists = db.query(Student).filter(Student.student_code == student_code).first()
    if exists:
        print(f"[!] Mã SV '{student_code}' đã tồn tại trong DB.")
        return None

    # 1. Tạo student
    student = Student(
        student_code=student_code,
        full_name=full_name,
        class_name=class_name,
    )
    db.add(student)
    db.flush()  # Lấy student.id nhưng chưa commit

    # 2. Tạo thư mục Images/{student_code}/
    student_dir = os.path.join(IMAGES_DIR, student_code)
    os.makedirs(student_dir, exist_ok=True)

    # 3. Xử lý từng ảnh
    success = 0
    for idx, src_path in enumerate(photo_paths, start=1):
        if not os.path.isfile(src_path):
            print(f"    [!] File không tồn tại: {src_path}")
            continue

        # Copy ảnh vào thư mục sinh viên
        ext = os.path.splitext(src_path)[1]  # .jpg / .png
        dest_filename = f"{student_code}_{idx}{ext}"
        dest_path = os.path.join(student_dir, dest_filename)
        shutil.copy2(src_path, dest_path)

        # Trích xuất embedding
        embedding = extract_embedding(dest_path)
        if embedding is None:
            print(f"    [!] Ảnh {idx}: Không phát hiện khuôn mặt → bỏ qua")
            os.remove(dest_path)
            continue

        # Lưu DB
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
    """
    Load tất cả face embedding từ DB lên RAM.

    Returns:
        [{"student_id": 1, "student_code": "21IT001",
          "full_name": "Nguyễn Văn A", "embedding": [0.12, ...]}, ...]
    """
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
        data.append(
            {
                "student_id": student_id,
                "student_code": code,
                "full_name": name,
                "embedding": emb,
            }
        )
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
    """
    Ghi 1 record điểm danh. Trả về True nếu mới, False nếu đã điểm danh trước đó.
    """
    # Kiểm tra đã điểm danh chưa
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
