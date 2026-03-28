"""
Re-extract embedding cho tất cả student_photos trong DB.

Vấn đề: Embedding cũ được tạo bởi DeepFace Facenet512 (norm ~23).
         InsightFace ArcFace tạo embedding trong không gian khác (norm ~1).
         → Cosine similarity giữa 2 loại ≈ 0 → không bao giờ match.

Script này:
  1. Duyệt tất cả student_photos
  2. Đọc ảnh gốc từ photo_path
  3. Dùng InsightFace extract_embedding() tạo ArcFace embedding mới
  4. UPDATE record trong DB với embedding mới
"""

import os
import sys
import math

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal
from app.models import Student, StudentPhoto
from app.face_service import extract_embedding
from app.config import IMAGES_DIR


def re_extract_all():
    db = SessionLocal()
    try:
        photos = (
            db.query(StudentPhoto)
            .join(Student, Student.id == StudentPhoto.student_id)
            .all()
        )
        print(f"Tìm thấy {len(photos)} ảnh trong DB")
        print("=" * 60)

        success = 0
        fail = 0
        skip = 0

        for photo in photos:
            student = db.query(Student).filter_by(id=photo.student_id).first()

            # Tìm file ảnh
            # photo_path có thể là relative path
            if os.path.isabs(photo.photo_path):
                full_path = photo.photo_path
            else:
                full_path = os.path.join(os.path.dirname(IMAGES_DIR), photo.photo_path)

            if not os.path.isfile(full_path):
                print(f"  [!] File không tồn tại: {full_path}")
                fail += 1
                continue

            # Kiểm tra embedding cũ
            old_emb = photo.face_embedding
            if old_emb is not None and len(old_emb) > 0:
                old_norm = math.sqrt(sum(v * v for v in old_emb[:512]))
                emb_type = "Facenet512" if old_norm > 5 else "ArcFace"
            else:
                old_norm = 0
                emb_type = "None"

            # Extract embedding mới bằng InsightFace
            new_emb = extract_embedding(full_path)

            if new_emb is None:
                print(f"  [✗] Photo #{photo.id} ({student.full_name}): "
                      f"Không detect được mặt — GIỮ embedding cũ")
                fail += 1
                continue

            new_norm = math.sqrt(sum(v * v for v in new_emb[:512]))

            # Update DB
            photo.face_embedding = new_emb
            db.commit()

            print(f"  [✓] Photo #{photo.id} ({student.full_name}): "
                  f"{emb_type} norm={old_norm:.1f} → ArcFace norm={new_norm:.4f}")
            success += 1

        print("=" * 60)
        print(f"Kết quả: {success} thành công, {fail} thất bại, {skip} bỏ qua")
        print(f"Tổng: {len(photos)} ảnh")

    finally:
        db.close()


if __name__ == "__main__":
    print("🔄 Re-extracting ALL embeddings: DeepFace → InsightFace ArcFace")
    print()
    re_extract_all()
    print()
    print("✅ Xong! Restart server rồi thử lại điểm danh.")
