"""ORM models cho hệ thống điểm danh."""

from datetime import datetime, date

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Date,
    DateTime,
    ForeignKey,
    UniqueConstraint,
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship

from app.database import Base


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

    def __repr__(self):
        return f"<Student {self.student_code} - {self.full_name}>"


class StudentPhoto(Base):
    """Bảng ảnh + face embedding. Mỗi SV có nhiều ảnh (khuyến nghị 5)."""

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


class AttendanceSession(Base):
    """Bảng phiên điểm danh (mỗi buổi học = 1 session)."""

    __tablename__ = "attendance_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_name = Column(String(200), comment="VD: Lập trình Python - Buổi 5")
    subject = Column(String(100))
    session_date = Column(Date, default=date.today)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    records = relationship("AttendanceRecord", back_populates="session", cascade="all, delete-orphan")

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
