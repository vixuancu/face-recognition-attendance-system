"""Router: Quản lý sinh viên."""

import os
import uuid

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.config import IMAGES_DIR
from app.models import Student
from app.schemas import StudentResponse
from app.face_service import register_student

router = APIRouter(prefix="/api/students", tags=["Students"])

UPLOAD_TMP = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tmp_uploads")
os.makedirs(UPLOAD_TMP, exist_ok=True)


@router.get("", response_model=list[StudentResponse])
def list_students(db: Session = Depends(get_db)):
    """Danh sách tất cả sinh viên."""
    students = db.query(Student).order_by(Student.student_code).all()
    result = []
    for s in students:
        result.append(StudentResponse(
            id=s.id,
            student_code=s.student_code,
            full_name=s.full_name,
            class_name=s.class_name,
            num_photos=len(s.photos),
            created_at=s.created_at.strftime("%d/%m/%Y") if s.created_at else "",
        ))
    return result


@router.post("/register")
async def register_student_api(
    student_code: str = Form(...),
    full_name: str = Form(...),
    class_name: str = Form(""),
    photos: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    """Đăng ký sinh viên mới + upload ảnh."""
    if not student_code or not full_name:
        raise HTTPException(status_code=400, detail="Mã SV và Họ tên không được trống!")

    valid_files = [f for f in photos if f and f.filename]
    if len(valid_files) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Cần ít nhất 10 ảnh (bạn upload {len(valid_files)} ảnh).",
        )

    # Lưu ảnh tạm
    tmp_paths = []
    for f in valid_files:
        ext = os.path.splitext(f.filename)[1]
        tmp_name = f"{uuid.uuid4().hex}{ext}"
        tmp_path = os.path.join(UPLOAD_TMP, tmp_name)
        content = await f.read()
        with open(tmp_path, "wb") as fout:
            fout.write(content)
        tmp_paths.append(tmp_path)

    try:
        student = register_student(db, student_code, full_name, class_name, tmp_paths)
        if student:
            return {
                "status": "success",
                "message": f"Đăng ký thành công: {full_name} ({student_code})",
                "student_id": student.id,
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="Đăng ký thất bại. Mã SV có thể đã tồn tại hoặc ảnh không hợp lệ.",
            )
    finally:
        for p in tmp_paths:
            if os.path.exists(p):
                os.remove(p)


@router.delete("/{student_id}")
def delete_student(student_id: int, db: Session = Depends(get_db)):
    """Xóa sinh viên."""
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Không tìm thấy sinh viên")

    name = student.full_name
    db.delete(student)
    db.commit()
    return {"status": "success", "message": f"Đã xóa: {name}"}
