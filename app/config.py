import os

# ─── PostgreSQL (Docker) ───────────────────────────────────
# postgresql+psycopg2://user:password@host:port/database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://root:123456@localhost:5432/attendance_db",
)

# ─── Đường dẫn lưu ảnh gốc ───────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "Images")

# ─── DeepFace config ──────────────────────────────────────
FACE_MODEL = "Facenet512"          # Vector 512 chiều, chính xác cao
DETECTOR_BACKEND = "retinaface"    # Chính xác nhất (dùng khi đăng ký & real-time)
DETECTOR_BACKEND_FAST = "ssd"      # Legacy: chỉ dùng cho CLI nếu cần nhanh
COSINE_THRESHOLD = 0.55            # Ngưỡng cosine similarity (thấp hơn = chặt hơn)

# ─── Long-distance recognition (3–5m) ─────────────────────
FRAME_UPSCALE_TARGET = 1280        # Upscale frame nhỏ để phát hiện mặt xa tốt hơn
FACE_CONFIDENCE_MIN = 0.3          # Ngưỡng confidence thấp hơn cho mặt ở xa
FACE_MIN_PIXELS = 160              # Kích thước tối thiểu face crop (Facenet cần 160x160)

# Tạo thư mục Images nếu chưa có
os.makedirs(IMAGES_DIR, exist_ok=True)
