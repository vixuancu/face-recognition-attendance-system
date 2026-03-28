"""ORM models cho hệ thống điểm danh."""

from datetime import datetime, date, time

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Date,
    Time,
    DateTime,
    ForeignKey,
    UniqueConstraint,
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship

from app.database import Base


# ════════════════════════════════════════════════════════════
#  SINH VIÊN
# ════════════════════════════════════════════════════════════

class Student(Base):
    """Bảng sinh viên."""

    __tablename__ = "students"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_code = Column(String(20), unique=True, nullable=False, comment="Mã SV: 21IT001")
    full_name = Column(String(100), nullable=False)
    class_name = Column(String(50), comment="Lớp: 21CNTT1")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    photos = relationship("StudentPhoto", back_populates="student", cascade="all, delete-orphan")
    attendance_records = relationship("AttendanceRecord", back_populates="student")
    enrollments = relationship("CourseEnrollment", back_populates="student", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Student {self.student_code} - {self.full_name}>"


class StudentPhoto(Base):
    """Bảng ảnh + face embedding. Mỗi SV có nhiều ảnh (khuyến nghị 10)."""

    __tablename__ = "student_photos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    photo_path = Column(String(500), nullable=False, comment="Đường dẫn tương đối đến ảnh")
    face_embedding = Column(Vector(512), nullable=False, comment="Vector 512 chiều (pgvector)")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    student = relationship("Student", back_populates="photos")

    def __repr__(self):
        return f"<Photo student_id={self.student_id} path={self.photo_path}>"


# ════════════════════════════════════════════════════════════
#  LỚP TÍN CHỈ
# ════════════════════════════════════════════════════════════

class Course(Base):
    """Lớp tín chỉ (VD: INT1340 - Lập trình Python)."""

    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    course_code = Column(String(20), unique=True, nullable=False, comment="Mã lớp tín chỉ: INT1340")
    course_name = Column(String(200), nullable=False, comment="Tên môn: Lập trình Python")
    room = Column(String(50), comment="Phòng: A305")
    semester = Column(String(20), comment="Học kỳ: 2025-2026-2")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    schedules = relationship("CourseSchedule", back_populates="course", cascade="all, delete-orphan")
    enrollments = relationship("CourseEnrollment", back_populates="course", cascade="all, delete-orphan")
    sessions = relationship("AttendanceSession", back_populates="course")

    def __repr__(self):
        return f"<Course {self.course_code} - {self.course_name}>"


class CourseSchedule(Base):
    """Lịch học từng buổi của lớp tín chỉ."""

    __tablename__ = "course_schedules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    course_id = Column(Integer, ForeignKey("courses.id", ondelete="CASCADE"), nullable=False)
    day_of_week = Column(Integer, nullable=False, comment="0=Mon, 1=Tue, ..., 6=Sun")
    start_time = Column(Time, nullable=False, comment="Giờ bắt đầu: 07:00")
    end_time = Column(Time, nullable=False, comment="Giờ kết thúc: 09:30")

    # Relationships
    course = relationship("Course", back_populates="schedules")

    def __repr__(self):
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_str = days[self.day_of_week] if 0 <= self.day_of_week <= 6 else "?"
        return f"<Schedule {day_str} {self.start_time}-{self.end_time}>"


class CourseEnrollment(Base):
    """Sinh viên đăng ký lớp tín chỉ."""

    __tablename__ = "course_enrollments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    course_id = Column(Integer, ForeignKey("courses.id", ondelete="CASCADE"), nullable=False)
    student_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    enrolled_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("course_id", "student_id", name="uq_course_student"),
    )

    # Relationships
    course = relationship("Course", back_populates="enrollments")
    student = relationship("Student", back_populates="enrollments")

    def __repr__(self):
        return f"<Enrollment course={self.course_id} student={self.student_id}>"


# ════════════════════════════════════════════════════════════
#  ĐIỂM DANH
# ════════════════════════════════════════════════════════════

class AttendanceSession(Base):
    """Bảng phiên điểm danh (mỗi buổi học = 1 session)."""

    __tablename__ = "attendance_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_name = Column(String(200), comment="VD: Lập trình Python - Buổi 5")
    subject = Column(String(100))
    session_date = Column(Date, default=date.today)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=True, comment="FK lớp tín chỉ (nullable cho backward-compat)")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    records = relationship("AttendanceRecord", back_populates="session", cascade="all, delete-orphan")
    course = relationship("Course", back_populates="sessions")

    def __repr__(self):
        return f"<Session {self.id} - {self.session_name}>"


class AttendanceRecord(Base):
    """Bảng kết quả điểm danh."""

    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("attendance_sessions.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    check_in_time = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default="Present")
    confidence = Column(Float, comment="Cosine similarity score")

    # Không cho điểm danh trùng trong cùng 1 session
    __table_args__ = (
        UniqueConstraint("session_id", "student_id", name="uq_session_student"),
    )

    # Relationships
    session = relationship("AttendanceSession", back_populates="records")
    student = relationship("Student", back_populates="attendance_records")

    def __repr__(self):
        return f"<Attendance session={self.session_id} student={self.student_id} status={self.status}>"
