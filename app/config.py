import os

# ─── PostgreSQL (Docker) ───────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://root:123456@localhost:5432/attendance_db",
)

# ─── Đường dẫn lưu ảnh gốc ───────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "Images")

# ─── InsightFace config ────────────────────────────────────
INSIGHTFACE_MODEL = "buffalo_l"   # SCRFD (detection) + ArcFace (512D embedding)

# ─── Matching thresholds (tối ưu cho ≤ 2m) ────────────────
# Ở ≤2m, webcam 720p → mặt ~100-200px → chất lượng cao
# → có thể dùng ngưỡng chặt chẽ hơn, chính xác hơn
COSINE_THRESHOLD    = 0.45        # Hạ xuống 0.45 vì ArcFace raw embedding (chưa L2-norm)
                                  # Ở ≤2m face lớn rõ → score tự nhiên cao hơn
FACE_CONFIDENCE_MIN = 0.6         # Chỉ nhận mặt có det_score ≥ 0.6 (lọc false positive)
FACE_MIN_PIXELS     = 80          # Ở ≤2m face luôn ≥ 80px. Reject < 80 (quá xa)

# ─── Anti-misidentification ────────────────────────────────
TOP_K_MATCHES       = 20          # Số kết quả lấy từ pgvector để phân tích
MATCH_MARGIN_MIN    = 0.03        # Hạ nhẹ để dễ match hơn khi chỉ có 1-2 SV trong cache
STUDENT_AGG_TOP_N   = 3           # Trung bình top-N ảnh/SV (aggregation)

# ─── Face quality scoring (tối ưu ≤ 2m) ──────────────────
QUALITY_FACE_SIZE_GOOD    = 100   # Ở ≤2m, face ≥ 100px → quality = 1.0 (trước 120)
QUALITY_FACE_SIZE_MIN     = 60    # Face 60-100px → quality tỷ lệ. < 60 → quá xa (trước 40)
QUALITY_THRESHOLD_PENALTY = 0.08  # Giảm penalty vì ≤2m mặt luôn đủ lớn (trước 0.10)

# ─── RTSP Camera Configuration ──────────────────────────────
RTSP_DEFAULT_URL      = os.getenv("RTSP_URL", "")
RTSP_PROCESS_INTERVAL = 1.5      # Xử lý AI mỗi 1.5 giây (đủ cho ≤2m)
RTSP_RECONNECT_DELAY  = 3.0      # Chờ 3s trước khi reconnect 
RTSP_STREAM_FPS       = 15      # FPS MJPEG stream gửi browser (tiết kiệm bandwidth)
RTSP_FRAME_WIDTH      = 1280     # Resize frame input → 720p
RTSP_FRAME_HEIGHT     = 720      # (InsightFace det_size=640 nên 720p là đủ)
RTSP_JPEG_QUALITY     = 70       # Giảm chất lượng xíu để nhẹ băng thông khi 30FPS

# Tạo thư mục Images nếu chưa có
os.makedirs(IMAGES_DIR, exist_ok=True)
