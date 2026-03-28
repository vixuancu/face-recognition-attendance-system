"""
Flask Web Server — Giao diện web cho hệ thống điểm danh.

Chạy:  python -m app.web
URL:   http://localhost:5000
"""

import os
import time
import uuid

import cv2
import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
)
from werkzeug.utils import secure_filename

from app.config import IMAGES_DIR, COSINE_THRESHOLD
from app.database import SessionLocal
from app.models import Student, AttendanceSession, AttendanceRecord
from app.face_service import (
    register_student,
    load_all_embeddings,
    extract_embedding,
    extract_embedding_from_frame,
    extract_all_embeddings_enhanced,
    find_best_match,
    find_best_match_pgvector_batch,
    find_best_match_pgvector_v2,
    save_attendance_record,
    create_session,
)

# ─── Flask app ─────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"),
    static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "static"),
)
app.secret_key = "face-attendance-secret-key-2026"

UPLOAD_TMP = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp_uploads")
os.makedirs(UPLOAD_TMP, exist_ok=True)

# Biến toàn cục cho phiên điểm danh hiện tại
current_session = {
    "id": None,
    "name": None,
    "attended": {},
}


# ═══════════════════════════════════════════════════════
#  TRANG CHỦ
# ═══════════════════════════════════════════════════════

@app.route("/")
def index():
    db = SessionLocal()
    try:
        total_students = db.query(Student).count()
        total_sessions = db.query(AttendanceSession).count()
        total_records = db.query(AttendanceRecord).count()
        return render_template(
            "index.html",
            total_students=total_students,
            total_sessions=total_sessions,
            total_records=total_records,
        )
    finally:
        db.close()


# ═══════════════════════════════════════════════════════
#  ĐĂNG KÝ SINH VIÊN
# ═══════════════════════════════════════════════════════

@app.route("/register", methods=["GET"])
def register_page():
    return render_template("register.html")


@app.route("/register", methods=["POST"])
def register_submit():
    student_code = request.form.get("student_code", "").strip()
    full_name = request.form.get("full_name", "").strip()
    class_name = request.form.get("class_name", "").strip()
    files = request.files.getlist("photos")

    # Validate
    if not student_code or not full_name:
        flash("Mã SV và Họ tên không được để trống!", "error")
        return redirect(url_for("register_page"))

    if not files or len(files) == 0 or files[0].filename == "":
        flash("Vui lòng chọn ít nhất 10 ảnh!", "error")
        return redirect(url_for("register_page"))

    valid_files = [f for f in files if f and f.filename]
    if len(valid_files) < 10:
        flash(f"Cần ít nhất 10 ảnh để đăng ký (bạn chọn {len(valid_files)} ảnh).", "error")
        return redirect(url_for("register_page"))

    # Lưu ảnh tạm
    tmp_paths = []
    for f in files:
        if f and f.filename:
            ext = os.path.splitext(f.filename)[1]
            tmp_name = f"{uuid.uuid4().hex}{ext}"
            tmp_path = os.path.join(UPLOAD_TMP, tmp_name)
            f.save(tmp_path)
            tmp_paths.append(tmp_path)

    # Đăng ký
    db = SessionLocal()
    try:
        student = register_student(db, student_code, full_name, class_name, tmp_paths)
        if student:
            flash(f"Đăng ký thành công: {full_name} ({student_code})", "success")
        else:
            flash(f"Đăng ký thất bại. Mã SV có thể đã tồn tại hoặc ảnh không hợp lệ.", "error")
    finally:
        db.close()
        # Xóa file tạm
        for p in tmp_paths:
            if os.path.exists(p):
                os.remove(p)

    return redirect(url_for("register_page"))


# ═══════════════════════════════════════════════════════
#  DANH SÁCH SINH VIÊN
# ═══════════════════════════════════════════════════════

@app.route("/students")
def students_page():
    db = SessionLocal()
    try:
        students = db.query(Student).order_by(Student.student_code).all()
        student_list = []
        for s in students:
            student_list.append({
                "id": s.id,
                "student_code": s.student_code,
                "full_name": s.full_name,
                "class_name": s.class_name or "",
                "num_photos": len(s.photos),
                "created_at": s.created_at.strftime("%d/%m/%Y") if s.created_at else "",
            })
        return render_template("students.html", students=student_list)
    finally:
        db.close()


@app.route("/api/students/<int:student_id>", methods=["DELETE"])
def delete_student(student_id):
    db = SessionLocal()
    try:
        student = db.query(Student).filter(Student.id == student_id).first()
        if not student:
            return jsonify({"status": "error", "message": "Không tìm thấy sinh viên"}), 404
        name = student.full_name
        db.delete(student)
        db.commit()
        return jsonify({"status": "success", "message": f"Đã xóa: {name}"})
    finally:
        db.close()


# ═══════════════════════════════════════════════════════
#  ĐIỂM DANH
# ═══════════════════════════════════════════════════════

@app.route("/attendance")
def attendance_page():
    return render_template("attendance.html")


@app.route("/api/attendance/start", methods=["POST"])
def start_session():
    """Tạo phiên điểm danh mới."""
    data = request.get_json()
    session_name = data.get("session_name", "Buổi học")
    subject = data.get("subject", "N/A")

    db = SessionLocal()
    try:
        att_session = create_session(db, session_name, subject)

        current_session["id"] = att_session.id
        current_session["name"] = session_name
        current_session["attended"] = {}
        return jsonify({
            "status": "success",
            "session_id": att_session.id,
            "message": f"Đã tạo phiên: {session_name}",
        })
    finally:
        db.close()


@app.route("/api/attendance/recognize", methods=["POST"])
def recognize_face():
    """
    Nhận diện NHIỀU khuôn mặt cùng lúc từ 1 frame.
    Sử dụng pgvector cosine distance — tìm kiếm trực tiếp trong DB.
    """
    if current_session["id"] is None:
        return jsonify({"status": "error", "message": "Chưa tạo phiên điểm danh!"}), 400

    # Đọc ảnh từ request
    if "image" in request.files:
        file = request.files["image"]
        img_bytes = file.read()
    elif request.is_json and "image_base64" in request.get_json():
        import base64
        img_b64 = request.get_json()["image_base64"]
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]
        img_bytes = base64.b64decode(img_b64)
    else:
        return jsonify({"status": "error", "message": "Không có ảnh!"})

    # Convert bytes → numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"status": "error", "message": "Ảnh không hợp lệ!"})

    t0 = time.time()

    # Enhanced pipeline: upscale → CLAHE → RetinaFace → enhance face → embed
    # Tối ưu cho khoảng cách xa 3–5m
    faces = extract_all_embeddings_enhanced(frame)
    if not faces:
        return jsonify({"status": "no_face", "message": "Không phát hiện khuôn mặt"})

    # pgvector V2 batch matching — chống nhầm người
    embeddings_list = [emb for emb, _, _ in faces]
    qualities_list = [quality for _, _, quality in faces]
    db = SessionLocal()
    try:
        matches = find_best_match_pgvector_batch(db, embeddings_list, qualities_list)

        elapsed = (time.time() - t0) * 1000
        print(f"[RECOGNIZE] {elapsed:.0f}ms — {len(faces)} face(s) detected (pgvector v2)")

        # Xây dựng response cho mỗi khuôn mặt
        face_results = []
        new_attended = []

        for i, ((emb, facial_area, quality), (match, score, debug)) in enumerate(zip(faces, matches)):
            face_box = {
                "x": facial_area.get("x", 0),
                "y": facial_area.get("y", 0),
                "w": facial_area.get("w", 0),
                "h": facial_area.get("h", 0),
            }

            if match:
                student_id = match["student_id"]
                already = student_id in current_session["attended"]

                if not already:
                    save_attendance_record(
                        db, current_session["id"], student_id, score
                    )
                    current_session["attended"][student_id] = {
                        "name": match["full_name"],
                        "code": match["student_code"],
                        "time": time.strftime("%H:%M:%S"),
                        "confidence": round(score * 100, 1),
                    }
                    new_attended.append({
                        "full_name": match["full_name"],
                        "student_code": match["student_code"],
                        "confidence": round(score * 100, 1),
                    })

                face_results.append({
                    "recognized": True,
                    "student_code": match["student_code"],
                    "full_name": match["full_name"],
                    "confidence": round(score * 100, 1),
                    "already_marked": already,
                    "face_box": face_box,
                })
            else:
                face_results.append({
                    "recognized": False,
                    "face_box": face_box,
                })

        return jsonify({
            "status": "success",
            "faces": face_results,
            "total_faces": len(face_results),
            "new_attended": new_attended,
            "total_attended": len(current_session["attended"]),
        })
    finally:
        db.close()


@app.route("/api/attendance/status")
def attendance_status():
    """Trả về trạng thái phiên điểm danh hiện tại."""
    return jsonify({
        "session_id": current_session["id"],
        "session_name": current_session["name"],
        "attended": list(current_session["attended"].values()),
        "total": len(current_session["attended"]),
    })


@app.route("/api/attendance/stop", methods=["POST"])
def stop_session():
    """Kết thúc phiên điểm danh."""
    result = {
        "total": len(current_session["attended"]),
        "attended": list(current_session["attended"].values()),
    }
    current_session["id"] = None
    current_session["name"] = None
    current_session["attended"] = {}
    return jsonify({"status": "success", **result})


# ═══════════════════════════════════════════════════════
#  LỊCH SỬ ĐIỂM DANH
# ═══════════════════════════════════════════════════════

@app.route("/history")
def history_page():
    db = SessionLocal()
    try:
        sessions = db.query(AttendanceSession).order_by(AttendanceSession.id.desc()).all()
        session_list = []
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
            session_list.append({
                "id": s.id,
                "name": s.session_name,
                "subject": s.subject,
                "date": s.session_date.strftime("%d/%m/%Y") if s.session_date else "",
                "total": len(records),
                "records": record_details,
            })
        return render_template("history.html", sessions=session_list)
    finally:
        db.close()


# ─── RUN ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  HỆ THỐNG ĐIỂM DANH SINH VIÊN CNTT")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
