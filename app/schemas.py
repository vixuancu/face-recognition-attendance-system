"""Pydantic schemas — Request/Response models cho FastAPI."""

from datetime import date, time, datetime
from typing import Optional
from pydantic import BaseModel


# ════════════════════════════════════════════════════════════
#  STUDENT
# ════════════════════════════════════════════════════════════

class StudentResponse(BaseModel):
    id: int
    student_code: str
    full_name: str
    class_name: Optional[str] = None
    num_photos: int = 0
    created_at: Optional[str] = None

    class Config:
        from_attributes = True


# ════════════════════════════════════════════════════════════
#  COURSE
# ════════════════════════════════════════════════════════════

class CourseCreate(BaseModel):
    course_code: str
    course_name: str
    room: Optional[str] = None
    semester: Optional[str] = None


class CourseResponse(BaseModel):
    id: int
    course_code: str
    course_name: str
    room: Optional[str] = None
    semester: Optional[str] = None
    num_students: int = 0

    class Config:
        from_attributes = True


class ScheduleCreate(BaseModel):
    day_of_week: int  # 0=Mon...6=Sun
    start_time: str   # "07:00"
    end_time: str     # "09:30"


class ScheduleResponse(BaseModel):
    id: int
    day_of_week: int
    start_time: str
    end_time: str

    class Config:
        from_attributes = True


class EnrollmentCreate(BaseModel):
    student_id: int


class EnrollmentBulkCreate(BaseModel):
    student_ids: list[int]


# ════════════════════════════════════════════════════════════
#  ATTENDANCE
# ════════════════════════════════════════════════════════════

class AttendanceStartRequest(BaseModel):
    course_id: int
    session_name: Optional[str] = None
    subject: Optional[str] = None


class AttendanceStartResponse(BaseModel):
    status: str
    session_id: int
    course_id: int
    session_name: str
    total_students: int
    cached_embeddings: int
    message: str


class FaceResult(BaseModel):
    recognized: bool
    student_code: Optional[str] = None
    full_name: Optional[str] = None
    confidence: Optional[float] = None
    already_marked: bool = False
    face_box: Optional[dict] = None


class RecognizeResponse(BaseModel):
    status: str
    faces: list[FaceResult] = []
    total_faces: int = 0
    new_attended: list[dict] = []
    total_attended: int = 0
    elapsed_ms: Optional[float] = None


class AttendanceStatusResponse(BaseModel):
    session_id: Optional[int] = None
    session_name: Optional[str] = None
    course_id: Optional[int] = None
    attended: list[dict] = []
    total: int = 0


class AttendanceRecordResponse(BaseModel):
    student_code: str
    full_name: str
    time: str
    confidence: float


class SessionHistoryResponse(BaseModel):
    id: int
    name: Optional[str] = None
    subject: Optional[str] = None
    date: Optional[str] = None
    course_code: Optional[str] = None
    total: int = 0
    records: list[AttendanceRecordResponse] = []
