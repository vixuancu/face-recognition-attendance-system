"""Router: Quản lý lớp tín chỉ, lịch học, đăng ký sinh viên."""

from datetime import time as dt_time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Course, CourseSchedule, CourseEnrollment, Student
from app.schemas import (
    CourseCreate,
    CourseResponse,
    ScheduleCreate,
    ScheduleResponse,
    EnrollmentCreate,
    EnrollmentBulkCreate,
    StudentResponse,
)

router = APIRouter(prefix="/api/courses", tags=["Courses"])


# ════════════════════════════════════════════════════════════
#  CRUD: Lớp tín chỉ
# ════════════════════════════════════════════════════════════

@router.post("", response_model=CourseResponse)
def create_course(data: CourseCreate, db: Session = Depends(get_db)):
    """Tạo lớp tín chỉ mới."""
    existing = db.query(Course).filter(Course.course_code == data.course_code).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Mã lớp '{data.course_code}' đã tồn tại!")

    course = Course(
        course_code=data.course_code,
        course_name=data.course_name,
        room=data.room,
        semester=data.semester,
    )
    db.add(course)
    db.commit()
    db.refresh(course)

    return CourseResponse(
        id=course.id,
        course_code=course.course_code,
        course_name=course.course_name,
        room=course.room,
        semester=course.semester,
        num_students=0,
    )


@router.get("", response_model=list[CourseResponse])
def list_courses(db: Session = Depends(get_db)):
    """Danh sách tất cả lớp tín chỉ."""
    courses = db.query(Course).order_by(Course.course_code).all()
    result = []
    for c in courses:
        num_students = db.query(CourseEnrollment).filter(CourseEnrollment.course_id == c.id).count()
        result.append(CourseResponse(
            id=c.id,
            course_code=c.course_code,
            course_name=c.course_name,
            room=c.room,
            semester=c.semester,
            num_students=num_students,
        ))
    return result


@router.get("/{course_id}", response_model=CourseResponse)
def get_course(course_id: int, db: Session = Depends(get_db)):
    """Chi tiết lớp tín chỉ."""
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Không tìm thấy lớp tín chỉ")

    num_students = db.query(CourseEnrollment).filter(CourseEnrollment.course_id == course.id).count()
    return CourseResponse(
        id=course.id,
        course_code=course.course_code,
        course_name=course.course_name,
        room=course.room,
        semester=course.semester,
        num_students=num_students,
    )


@router.delete("/{course_id}")
def delete_course(course_id: int, db: Session = Depends(get_db)):
    """Xóa lớp tín chỉ."""
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Không tìm thấy lớp tín chỉ")

    name = course.course_name
    db.delete(course)
    db.commit()
    return {"status": "success", "message": f"Đã xóa lớp: {name}"}


# ════════════════════════════════════════════════════════════
#  Lịch học
# ════════════════════════════════════════════════════════════

@router.post("/{course_id}/schedules", response_model=ScheduleResponse)
def add_schedule(course_id: int, data: ScheduleCreate, db: Session = Depends(get_db)):
    """Thêm lịch học cho lớp."""
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Không tìm thấy lớp tín chỉ")

    if not (0 <= data.day_of_week <= 6):
        raise HTTPException(status_code=400, detail="day_of_week phải từ 0 (Mon) đến 6 (Sun)")

    # Parse time strings
    try:
        parts_start = data.start_time.split(":")
        parts_end = data.end_time.split(":")
        start = dt_time(int(parts_start[0]), int(parts_start[1]))
        end = dt_time(int(parts_end[0]), int(parts_end[1]))
    except (ValueError, IndexError):
        raise HTTPException(status_code=400, detail="Định dạng thời gian: HH:MM")

    schedule = CourseSchedule(
        course_id=course_id,
        day_of_week=data.day_of_week,
        start_time=start,
        end_time=end,
    )
    db.add(schedule)
    db.commit()
    db.refresh(schedule)

    return ScheduleResponse(
        id=schedule.id,
        day_of_week=schedule.day_of_week,
        start_time=schedule.start_time.strftime("%H:%M"),
        end_time=schedule.end_time.strftime("%H:%M"),
    )


@router.get("/{course_id}/schedules", response_model=list[ScheduleResponse])
def list_schedules(course_id: int, db: Session = Depends(get_db)):
    """Danh sách lịch học của lớp."""
    schedules = db.query(CourseSchedule).filter(CourseSchedule.course_id == course_id).all()
    return [
        ScheduleResponse(
            id=s.id,
            day_of_week=s.day_of_week,
            start_time=s.start_time.strftime("%H:%M"),
            end_time=s.end_time.strftime("%H:%M"),
        )
        for s in schedules
    ]


# ════════════════════════════════════════════════════════════
#  Đăng ký sinh viên vào lớp
# ════════════════════════════════════════════════════════════

@router.post("/{course_id}/enrollments")
def enroll_student(course_id: int, data: EnrollmentCreate, db: Session = Depends(get_db)):
    """Đăng ký 1 sinh viên vào lớp."""
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Không tìm thấy lớp tín chỉ")

    student = db.query(Student).filter(Student.id == data.student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Không tìm thấy sinh viên")

    existing = db.query(CourseEnrollment).filter_by(
        course_id=course_id, student_id=data.student_id
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Sinh viên đã đăng ký lớp này!")

    enrollment = CourseEnrollment(course_id=course_id, student_id=data.student_id)
    db.add(enrollment)
    db.commit()

    return {
        "status": "success",
        "message": f"Đã thêm {student.full_name} vào lớp {course.course_name}",
    }


@router.post("/{course_id}/enrollments/bulk")
def enroll_students_bulk(course_id: int, data: EnrollmentBulkCreate, db: Session = Depends(get_db)):
    """Đăng ký nhiều sinh viên vào lớp."""
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Không tìm thấy lớp tín chỉ")

    added = 0
    skipped = 0
    for sid in data.student_ids:
        student = db.query(Student).filter(Student.id == sid).first()
        if not student:
            skipped += 1
            continue

        existing = db.query(CourseEnrollment).filter_by(
            course_id=course_id, student_id=sid
        ).first()
        if existing:
            skipped += 1
            continue

        db.add(CourseEnrollment(course_id=course_id, student_id=sid))
        added += 1

    db.commit()
    return {
        "status": "success",
        "added": added,
        "skipped": skipped,
        "message": f"Đã thêm {added} SV, bỏ qua {skipped}",
    }


@router.get("/{course_id}/students", response_model=list[StudentResponse])
def list_course_students(course_id: int, db: Session = Depends(get_db)):
    """Danh sách sinh viên của lớp."""
    enrollments = (
        db.query(CourseEnrollment)
        .filter(CourseEnrollment.course_id == course_id)
        .all()
    )

    result = []
    for e in enrollments:
        s = e.student
        result.append(StudentResponse(
            id=s.id,
            student_code=s.student_code,
            full_name=s.full_name,
            class_name=s.class_name,
            num_photos=len(s.photos),
            created_at=s.created_at.strftime("%d/%m/%Y") if s.created_at else "",
        ))
    return result


@router.delete("/{course_id}/enrollments/{student_id}")
def remove_enrollment(course_id: int, student_id: int, db: Session = Depends(get_db)):
    """Xóa sinh viên khỏi lớp."""
    enrollment = db.query(CourseEnrollment).filter_by(
        course_id=course_id, student_id=student_id
    ).first()
    if not enrollment:
        raise HTTPException(status_code=404, detail="Không tìm thấy đăng ký")

    db.delete(enrollment)
    db.commit()
    return {"status": "success", "message": "Đã xóa sinh viên khỏi lớp"}
