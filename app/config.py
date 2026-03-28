import os

# ─── PostgreSQL (Docker) ───────────────────────────────────
# postgresql+psycopg2://user:password@host:port/database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://root:123456@localhost:5432/attendance_db",
)

# ─── Redis ─────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "7200"))  # 2 giờ

# ─── Đường dẫn lưu ảnh gốc ───────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "Images")

# ─── DeepFace config ──────────────────────────────────────
FACE_MODEL = "Facenet512"          # Vector 512 chiều, chính xác cao
DETECTOR_BACKEND = "retinaface"    # Chính xác nhất (dùng khi đăng ký SV)
DETECTOR_BACKEND_FAST = "opencv"   # Dùng cho real-time crops (Haar ~2ms trên crop nhỏ)
COSINE_THRESHOLD = 0.55            # Ngưỡng cosine similarity (thấp hơn = chặt hơn)

# ─── Distance recognition (≤ 2.5m, tối ưu CPU) ───────────
FRAME_UPSCALE_TARGET = 640         # Giảm từ 960 — 2.5m không cần upscale mạnh
FACE_CONFIDENCE_MIN = 0.5          # Ngưỡng confidence (MediaPipe đã lọc trước)
FACE_MIN_PIXELS = 100              # Giảm từ 160 — mặt ở 2.5m đủ lớn

# ─── Anti-misidentification (chống điểm danh nhầm) ───────
TOP_K_MATCHES = 20                 # Số lượng kết quả trả về từ pgvector để phân tích
MATCH_MARGIN_MIN = 0.04            # Khoảng cách tối thiểu giữa SV #1 và SV #2 (tránh nhầm)
STUDENT_AGG_TOP_N = 3              # Lấy trung bình top-N ảnh/SV (aggregation mạnh hơn max)
QUALITY_FACE_SIZE_GOOD = 120       # Mặt >= 120px → chất lượng tốt (threshold bình thường)
QUALITY_FACE_SIZE_MIN = 40         # Mặt < 40px → chất lượng rất thấp (threshold cao nhất)
QUALITY_THRESHOLD_PENALTY = 0.10   # Penalty tối đa thêm vào threshold khi mặt nhỏ/mờ

# Tạo thư mục Images nếu chưa có
os.makedirs(IMAGES_DIR, exist_ok=True)
