import cv2
import time
import threading
import logging
import traceback
import os
from typing import Optional

from app.config import (
    RTSP_PROCESS_INTERVAL, RTSP_RECONNECT_DELAY, RTSP_FRAME_WIDTH, RTSP_FRAME_HEIGHT,
    RTSP_JPEG_QUALITY
)
from app.face_service import get_face_app, _match_from_cache, _compute_face_quality, save_attendance_record
from app.ram_cache import get_cached_embeddings, is_attended, mark_attended
from app.database import SessionLocal

logger = logging.getLogger("rtsp_worker")

class CameraWorker:
    def __init__(self, camera_id: str, rtsp_url: str, session_id: int, course_id: int):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.session_id = session_id
        self.course_id = course_id
        
        self.running = False
        self._lock = threading.Lock()
        
        self._latest_frame = None
        self._annotated_frame = None
        self._latest_results = []
        
        self._thread_reader: Optional[threading.Thread] = None
        self._thread_processor: Optional[threading.Thread] = None
        
        self._cap = None
        
        self.connected = False
        self.processed_count = 0
        self.attended_count = 0

    def start(self):
        self.running = True
        self._thread_reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread_processor = threading.Thread(target=self._processor_loop, daemon=True)
        self._thread_reader.start()
        self._thread_processor.start()
        logger.info(f"Worker {self.camera_id} started")

    def stop(self):
        self.running = False
        if self._cap:
            self._cap.release()
        if self._thread_reader:
            self._thread_reader.join(timeout=2)
        if self._thread_processor:
            self._thread_processor.join(timeout=2)
        logger.info(f"Worker {self.camera_id} stopped")

    def get_stream_frame(self) -> Optional[bytes]:
        with self._lock:
            frame = self._annotated_frame if self._annotated_frame is not None else self._latest_frame
            if frame is None:
                return None
            
            # Encode to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), RTSP_JPEG_QUALITY])
            if ret:
                return buffer.tobytes()
        return None

    def get_latest_results(self) -> list:
        with self._lock:
            return self._latest_results.copy()

    def _reader_loop(self):
        while self.running:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self._cap.isOpened():
                logger.warning(f"Worker {self.camera_id}: Cannot open RTSP stream. Retrying in {RTSP_RECONNECT_DELAY}s")
                self.connected = False
                time.sleep(RTSP_RECONNECT_DELAY)
                continue
            
            self.connected = True
            logger.info(f"Worker {self.camera_id}: Connected to RTSP stream")

            while self.running and self._cap.isOpened():
                ret, frame = self._cap.read()
                if not ret:
                    logger.warning(f"Worker {self.camera_id}: Lost connection. Reconnecting...")
                    self.connected = False
                    break

                # Resize if needed
                h, w = frame.shape[:2]
                if w > RTSP_FRAME_WIDTH or h > RTSP_FRAME_HEIGHT:
                    frame = cv2.resize(frame, (RTSP_FRAME_WIDTH, RTSP_FRAME_HEIGHT))

                with self._lock:
                    self._latest_frame = frame

            if self._cap:
                self._cap.release()
            self.connected = False

    def _processor_loop(self):
        # preload model so it won't block the first iteration
        logger.info(f"Worker {self.camera_id}: Preloading InsightFace model...")
        get_face_app()

        cache_data = get_cached_embeddings(self.course_id)
        if not cache_data or cache_data["embeddings"].size == 0:
            logger.error(f"Worker {self.camera_id}: Cache empty for course {self.course_id}")
            # we keep running, in case cache is updated later
        
        while self.running:
            time.sleep(RTSP_PROCESS_INTERVAL)
            
            with self._lock:
                frame = self._latest_frame.copy() if self._latest_frame is not None else None
            
            if frame is None:
                continue

            try:
                self._process_frame(frame, cache_data)
            except Exception as e:
                logger.error(f"Worker {self.camera_id} processing error: {e}")
                logger.error(traceback.format_exc())

    def _process_frame(self, frame, cache_data):
        app = get_face_app()
        faces = app.get(frame)
        
        annotated_frame = frame.copy()
        current_results = []
        
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            size = max(x2 - x1, y2 - y1)
            quality = _compute_face_quality(size)
            
            match_info, score, dbg = _match_from_cache(f.embedding, cache_data, quality)
            
            if match_info:
                sid = match_info["student_id"]
                scode = match_info["student_code"]
                sname = match_info["full_name"]
                
                already = is_attended(self.course_id, sid)
                if not already:
                    mark_attended(self.course_id, sid)
                    self.attended_count += 1
                    # Ghi DB
                    try:
                        db = SessionLocal()
                        save_attendance_record(db, self.session_id, sid, score)
                    except Exception as e:
                        logger.error(f"DB Error: {e}")
                    finally:
                        db.close()
                
                # Vẽ box
                color = (0, 255, 255) if already else (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                text = f"{sname} ({score:.2f})"
                cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                current_results.append({
                    "recognized": True,
                    "student_code": scode,
                    "full_name": sname,
                    "confidence": round(score * 100, 1),
                    "already_marked": already
                })
            else:
                # Không nhận ra
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                current_results.append({"recognized": False})
        
        self.processed_count += 1
        with self._lock:
            self._annotated_frame = annotated_frame
            self._latest_results = current_results

class CameraWorkerManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CameraWorkerManager, cls).__new__(cls)
                    cls._instance.workers = {}
        return cls._instance

    def create_worker(self, camera_id: str, rtsp_url: str, session_id: int, course_id: int) -> CameraWorker:
        if camera_id in self.workers:
            self.stop_worker(camera_id)
        
        w = CameraWorker(camera_id, rtsp_url, session_id, course_id)
        w.start()
        self.workers[camera_id] = w
        return w

    def stop_worker(self, camera_id: str):
        w = self.workers.get(camera_id)
        if w:
            w.stop()
            del self.workers[camera_id]

    def stop_all(self):
        for w in list(self.workers.values()):
            w.stop()
        self.workers.clear()

    def get_worker(self, camera_id: str) -> Optional[CameraWorker]:
        return self.workers.get(camera_id)

    def list_workers(self) -> list:
        return [
            {
                "camera_id": cid,
                "connected": w.connected,
                "processed_count": w.processed_count,
                "attended_count": w.attended_count
            }
            for cid, w in self.workers.items()
        ]

def get_worker_manager() -> CameraWorkerManager:
    return CameraWorkerManager()
