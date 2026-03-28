"""Router: Trang web — Serve Jinja2 templates cho giao diện test."""

import os
from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Student, AttendanceSession, AttendanceRecord, Course, CourseEnrollment

router = APIRouter(tags=["Pages"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@router.get("/")
def index_page(request: Request, db: Session = Depends(get_db)):
    """Trang chủ."""
    total_students = db.query(Student).count()
    total_sessions = db.query(AttendanceSession).count()
    total_records = db.query(AttendanceRecord).count()
    total_courses = db.query(Course).count()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "total_students": total_students,
            "total_sessions": total_sessions,
            "total_records": total_records,
            "total_courses": total_courses,
        }
    )


@router.get("/register")
def register_page(request: Request):
    """Trang đăng ký sinh viên."""
    return templates.TemplateResponse(request=request, name="register.html")


@router.get("/students")
def students_page(request: Request, db: Session = Depends(get_db)):
    """Trang danh sách sinh viên."""
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
    return templates.TemplateResponse(
        request=request,
        name="students.html",
        context={"students": students_list}
    )


@router.get("/attendance")
def attendance_page(request: Request, db: Session = Depends(get_db)):
    """Trang điểm danh."""
    # Lấy danh sách lớp để hiển thị dropdown
    courses = db.query(Course).order_by(Course.course_code).all()
    course_list = []
    for c in courses:
        num_students = db.query(CourseEnrollment).filter(
            CourseEnrollment.course_id == c.id
        ).count()
        course_list.append({
            "id": c.id,
            "course_code": c.course_code,
            "course_name": c.course_name,
            "room": c.room or "",
            "num_students": num_students,
        })
    return templates.TemplateResponse(
        request=request,
        name="attendance.html",
        context={"courses": course_list}
    )

@router.get("/attendance-rtsp")
def attendance_rtsp_page(request: Request, db: Session = Depends(get_db)):
    """Trang điểm danh bằng camera IP (RTSP)."""
    # Lấy danh sách lớp để hiển thị dropdown
    courses = db.query(Course).order_by(Course.course_code).all()
    course_list = []
    for c in courses:
        num_students = db.query(CourseEnrollment).filter(
            CourseEnrollment.course_id == c.id
        ).count()
        course_list.append({
            "id": c.id,
            "course_code": c.course_code,
            "course_name": c.course_name,
            "room": c.room or "",
            "num_students": num_students,
        })
    return templates.TemplateResponse(
        request=request,
        name="attendance_rtsp.html",
        context={"courses": course_list}
    )


@router.get("/courses")
def courses_page(request: Request, db: Session = Depends(get_db)):
    """Trang quản lý lớp tín chỉ."""
    courses = db.query(Course).order_by(Course.course_code).all()
    course_list = []
    for c in courses:
        num_students = db.query(CourseEnrollment).filter(
            CourseEnrollment.course_id == c.id
        ).count()
        course_list.append({
            "id": c.id,
            "course_code": c.course_code,
            "course_name": c.course_name,
            "room": c.room or "",
            "semester": c.semester or "",
            "num_students": num_students,
        })
    return templates.TemplateResponse(
        request=request,
        name="courses.html",
        context={"courses": course_list}
    )


@router.get("/history")
def history_page(request: Request, db: Session = Depends(get_db)):
    """Trang lịch sử điểm danh."""
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

        course = db.query(Course).filter_by(id=s.course_id).first() if s.course_id else None
        session_list.append({
            "id": s.id,
            "name": s.session_name,
            "subject": s.subject,
            "date": s.session_date.strftime("%d/%m/%Y") if s.session_date else "",
            "course_code": course.course_code if course else "",
            "total": len(records),
            "records": record_details,
        })
    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={"sessions": session_list}
    )
