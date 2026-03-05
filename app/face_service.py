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


def extract_all_embeddings_enhanced(
    frame, detector: str = DETECTOR_BACKEND
) -> list[tuple[list, dict]]:
    """
    Pipeline nhận diện khuôn mặt tối ưu cho KHOẢNG CÁCH XA (3–5m).

    So với extract_all_embeddings_from_frame (chỉ gọi represent 1 lần):
    - Tách detection và embedding thành 2 bước riêng
    - Upscale frame trước detection → detect được mặt nhỏ hơn
    - Enhance từng face crop riêng → embedding chất lượng cao hơn
    - Dùng RetinaFace (không SSD) → tốt nhất cho mặt nhỏ

    Returns:
        List of (embedding, facial_area) — cùng format với hàm cũ.
        facial_area tọa độ TRÊN FRAME GỐC (đã map ngược từ upscale).
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
                    faces.append((embedding, orig_area))
            except Exception as e:
                print(f"    [!] Enhanced embed error: {e}")
                continue

        print(
            f"    [enhance] {w}x{h} → {frame_up.shape[1]}x{frame_up.shape[0]}"
            f" (scale {scale:.1f}x) | {len(detected)} detected → {len(faces)} valid"
        )
        return faces

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
    Tìm sinh viên khớp nhất bằng pgvector cosine distance.
    Toán tử <=> tính cosine distance (1 - cosine_similarity).

    SQL: SELECT ... ORDER BY embedding <=> query LIMIT 1

    Returns:
        (match_dict, cosine_similarity) hoặc (None, 0.0)
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


def find_best_match_pgvector_batch(
    db: Session,
    face_embeddings: list[list],
    threshold: float = COSINE_THRESHOLD,
) -> list[tuple[Optional[dict], float]]:
    """
    So khớp NHIỀU khuôn mặt bằng pgvector — mỗi face 1 query SQL.
    pgvector sử dụng HNSW/IVFFlat index nên mỗi query rất nhanh (< 1ms).

    Returns:
        List of (match_dict, score) cho mỗi face.
    """
    results = []
    for emb in face_embeddings:
        match, score = find_best_match_pgvector(db, emb, threshold)
        results.append((match, score))
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
