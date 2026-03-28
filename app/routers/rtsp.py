import asyncio
import time
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Course, CourseEnrollment, Student, StudentPhoto
from app.face_service import create_session
from app.ram_cache import load_class_embeddings_to_cache, clear_class_cache
from app.rtsp_worker import get_worker_manager
from app.config import RTSP_STREAM_FPS

logger = logging.getLogger("rtsp_router")
router = APIRouter(prefix="/api/rtsp", tags=["RTSP Camera"])

class RTSPStartRequest(BaseModel):
    camera_id: str
    rtsp_url: str
    course_id: int
    session_name: Optional[str] = None
    subject: Optional[str] = None

class RTSPStopRequest(BaseModel):
    camera_id: str

@router.post("/start")
async def start_rtsp_camera(data: RTSPStartRequest, db: Session = Depends(get_db)):
    """Bắt đầu phiên điểm danh cho RTSP Camera."""
    manager = get_worker_manager()

    if manager.get_worker(data.camera_id):
        raise HTTPException(status_code=400, detail=f"Camera {data.camera_id} đang chạy!")

    # Validate course
    course = db.query(Course).filter(Course.id == data.course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Không tìm thấy lớp tín chỉ!")

    # Lấy sinh viên
    enrollments = db.query(CourseEnrollment).filter(CourseEnrollment.course_id == data.course_id).all()
    if not enrollments:
        raise HTTPException(status_code=400, detail="Lớp chưa có sinh viên nào!")

    student_ids = [e.student_id for e in enrollments]
    students_data = []
    
    for sid in student_ids:
        student = db.query(Student).filter(Student.id == sid).first()
        if not student: continue
        photos = db.query(StudentPhoto).filter(StudentPhoto.student_id == sid).all()
        if not photos: continue

        students_data.append({
            "student_id": sid,
            "student_code": student.student_code,
            "full_name": student.full_name,
            "photos": [{"photo_id": p.id, "embedding": list(p.face_embedding)} for p in photos]
        })

    # Cache
    load_class_embeddings_to_cache(data.course_id, students_data)

    # Tạo session
    s_name = data.session_name or f"Camera {data.camera_id} - {course.course_code}"
    db_session = create_session(db, s_name, data.subject or "Điểm danh RTSP")

    # Khởi động worker
    manager.create_worker(data.camera_id, data.rtsp_url, db_session.id, data.course_id)

    return {
        "status": "success",
        "camera_id": data.camera_id,
        "session_id": db_session.id,
        "session_name": s_name,
        "total_students": len(students_data)
    }

@router.post("/stop")
async def stop_rtsp_camera(data: RTSPStopRequest):
    manager = get_worker_manager()
    worker = manager.get_worker(data.camera_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Không tìm thấy camera!")
    
    attended_count = worker.attended_count
    course_id = worker.course_id
    manager.stop_worker(data.camera_id)
    clear_class_cache(course_id)
    
    return {"status": "success", "total_attended": attended_count}

@router.get("/status")
def get_all_status():
    manager = get_worker_manager()
    return manager.list_workers()

@router.get("/status/{camera_id}")
def get_camera_status(camera_id: str):
    manager = get_worker_manager()
    worker = manager.get_worker(camera_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Not found")
    
    return {
        "camera_id": camera_id,
        "connected": worker.connected,
        "processed_count": worker.processed_count,
        "attended_count": worker.attended_count
    }

async def mjpeg_generator(camera_id: str):
    manager = get_worker_manager()
    # Check quickly, but don't overwhelm loop
    delay = 1.0 / (RTSP_STREAM_FPS * 2) 
    last_frame_bytes = None
    
    while True:
        worker = manager.get_worker(camera_id)
        if not worker or not worker.running:
            break
            
        frame_bytes = worker.get_stream_frame()
        if frame_bytes and frame_bytes != last_frame_bytes:
            last_frame_bytes = frame_bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        await asyncio.sleep(delay)

@router.get("/stream/{camera_id}")
async def mjpeg_stream(camera_id: str):
    return StreamingResponse(
        mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

async def sse_generator(request: Request, camera_id: str):
    manager = get_worker_manager()
    last_processed_count = -1
    last_heartbeat = time.time()
    
    while True:
        if await request.is_disconnected():
            break
            
        worker = manager.get_worker(camera_id)
        if not worker or not worker.running:
            break
            
        if worker.processed_count > last_processed_count:
            last_processed_count = worker.processed_count
            results = worker.get_latest_results()
            if results:
                data = json.dumps({
                    "faces": results, 
                    "total_attended": worker.attended_count, 
                    "connected": worker.connected
                })
                yield f"data: {data}\n\n"
        
        # heartbeat giữ kết nối SSE không bị browser huỷ
        if time.time() - last_heartbeat > 5.0:
            yield ": heartbeat\n\n"
            last_heartbeat = time.time()
            
        await asyncio.sleep(0.1)

@router.get("/results/{camera_id}")
async def sse_results(request: Request, camera_id: str):
    return StreamingResponse(
        sse_generator(request, camera_id),
        media_type="text/event-stream"
    )
