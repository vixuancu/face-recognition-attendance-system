"""
CLI chính — Hệ thống điểm danh sinh viên CNTT.

Chạy:  python -m app.main
"""

import sys
import cv2

from app.database import SessionLocal
from app.models import Student
from app.face_service import (
    register_student,
    load_all_embeddings,
    extract_embedding_from_frame,
    find_best_match,
    save_attendance_record,
    create_session,
)


def menu():
    print()
    print("=" * 55)
    print("  HỆ THỐNG ĐIỂM DANH SINH VIÊN CNTT")
    print("  PostgreSQL + DeepFace (Facenet512)")
    print("=" * 55)
    print("  1.  Đăng ký sinh viên mới (upload 5 ảnh)")
    print("  2.  Bắt đầu điểm danh (camera)")
    print("  3.  Xem danh sách sinh viên đã đăng ký")
    print("  0.  Thoát")
    print("=" * 55)


# ────────────────────────────────────────────────────────
#  1. ĐĂNG KÝ SINH VIÊN
# ────────────────────────────────────────────────────────

def cmd_register():
    print("\n── ĐĂNG KÝ SINH VIÊN MỚI ──")
    code = input("  Mã sinh viên (VD: 21IT001): ").strip()
    if not code:
        print("  [!] Mã SV không được trống.")
        return

    name = input("  Họ tên: ").strip()
    cls = input("  Lớp: ").strip()

    photos = []
    print("  Nhập đường dẫn 5 ảnh khuôn mặt (Enter để bỏ qua nếu < 5):")
    for i in range(5):
        path = input(f"    Ảnh {i + 1}: ").strip().strip('"')
        if not path:
            break
        photos.append(path)

    if not photos:
        print("  [!] Cần ít nhất 1 ảnh.")
        return

    db = SessionLocal()
    try:
        register_student(db, code, name, cls, photos)
    finally:
        db.close()


# ────────────────────────────────────────────────────────
#  2. ĐIỂM DANH
# ────────────────────────────────────────────────────────

def cmd_attendance():
    print("\n── BẮT ĐẦU ĐIỂM DANH ──")

    db = SessionLocal()
    try:
        # Load embeddings
        all_emb = load_all_embeddings(db)
        if not all_emb:
            print("  [!] Chưa có sinh viên nào trong DB. Hãy đăng ký trước (chọn 1).")
            return

        # Tạo phiên điểm danh
        session_name = input("  Tên buổi học (VD: LT Python - Buổi 5): ").strip() or "Buổi học"
        subject = input("  Môn học: ").strip() or "N/A"
        att_session = create_session(db, session_name, subject)
        print(f"  [i] Tạo phiên #{att_session.id}: {session_name}")

        # Mở camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  [✗] Không thể mở camera!")
            return

        print("\n  🎥 Camera đã bật. Nhấn 'q' để thoát.")
        print("  " + "─" * 40)

        attended = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            small = cv2.resize(frame, (640, 480))

            # Trích xuất embedding
            emb, facial_area = extract_embedding_from_frame(small)

            label = "Scanning..."
            color = (200, 200, 200)

            if emb is not None:
                match, score = find_best_match(emb, all_emb)
                if match:
                    student_name = match["full_name"]
                    student_code = match["student_code"]
                    label = f"{student_name} ({score:.0%})"
                    color = (0, 255, 0)

                    if match["student_id"] not in attended:
                        ok = save_attendance_record(
                            db, att_session.id, match["student_id"], score
                        )
                        if ok:
                            attended.add(match["student_id"])
                            print(f"  [✓] Điểm danh: {student_name} ({student_code}) — {score:.1%}")
                else:
                    label = "Unknown"
                    color = (0, 0, 255)

            # Vẽ text lên frame
            cv2.putText(display, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(
                display,
                f"Da diem danh: {len(attended)} SV",
                (30, display.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Diem Danh - Nhan 'q' de thoat", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n  📊 Tổng cộng: {len(attended)} sinh viên đã điểm danh.")

    finally:
        db.close()


# ────────────────────────────────────────────────────────
#  3. XEM DANH SÁCH
# ────────────────────────────────────────────────────────

def cmd_list_students():
    print("\n── DANH SÁCH SINH VIÊN ──")
    db = SessionLocal()
    try:
        students = db.query(Student).order_by(Student.student_code).all()
        if not students:
            print("  (chưa có sinh viên nào)")
            return

        print(f"  {'Mã SV':<12} {'Họ tên':<30} {'Lớp':<15} {'Số ảnh'}")
        print("  " + "─" * 65)
        for s in students:
            n_photos = len(s.photos)
            print(f"  {s.student_code:<12} {s.full_name:<30} {s.class_name or '':<15} {n_photos}")
    finally:
        db.close()


# ────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────

def main():
    while True:
        menu()
        choice = input("  Chọn (0-3): ").strip()

        if choice == "1":
            cmd_register()
        elif choice == "2":
            cmd_attendance()
        elif choice == "3":
            cmd_list_students()
        elif choice == "0":
            print("  Tạm biệt!")
            sys.exit(0)
        else:
            print("  [!] Lựa chọn không hợp lệ.")


if __name__ == "__main__":
    main()
