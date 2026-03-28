# 🎓 HỆ THỐNG ĐIỂM DANH SINH VIÊN QUA NHẬN DIỆN KHUÔN MẶT

> **Source base test** — FastAPI + InsightFace (ArcFace 512D) + PostgreSQL pgvector + RAM Cache.  
> Dự án này là nền tảng AI nhận diện khuôn mặt hoàn chỉnh, có thể tái sử dụng (port) sang bất kỳ hệ thống quản lý nào.

---

## 📑 MỤC LỤC

1. [Tổng quan kiến trúc](#-tổng-quan-kiến-trúc)
2. [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
3. [Danh sách thư viện cần cài đặt](#-danh-sách-thư-viện-cần-cài-đặt-chi-tiết)
4. [Hướng dẫn cài đặt từng bước](#-hướng-dẫn-cài-đặt-từng-bước-windows)
5. [Khởi tạo Database](#-khởi-tạo-database)
6. [Khởi chạy hệ thống](#-khởi-chạy-hệ-thống)
7. [Giải thích Logic A → Z](#-giải-thích-logic-a--z)
   - [Tầng 1: Database Models](#tầng-1-database-models--appmodelspy)
   - [Tầng 2: Face Service (Lõi AI)](#tầng-2-face-service-lõi-ai--appface_servicepy)
   - [Tầng 3: RAM Cache](#tầng-3-ram-cache--appram_cachepy)
   - [Tầng 4: API Routers](#tầng-4-api-routers--approuters)
   - [Tầng 5: Web UI (Jinja2 Templates)](#tầng-5-web-ui-jinja2-templates)
8. [Luồng hoạt động End-to-End](#-luồng-hoạt-động-end-to-end)
9. [Các file phụ trợ](#-các-file-phụ-trợ)
10. [Cách áp dụng sang dự án khác](#-cách-áp-dụng-sang-dự-án-khác)

---

## 🏗 Tổng quan kiến trúc

```
┌──────────────────────────────────────────────────────────────────────┐
│                           CLIENT (Browser)                           │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  MediaPipe Face Detection (JavaScript, chạy trên trình duyệt)  │ │
│  │  → Detect mặt 30fps → Crop từng mặt → Gửi lên Server          │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────┬───────────────────────────────────────────┘
                           │  HTTP POST (face crops)
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        SERVER (FastAPI + Uvicorn)                     │
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐ │
│  │ InsightFace   │   │  RAM Cache   │   │   PostgreSQL + pgvector  │ │
│  │ (ArcFace      │──▶│ (Python dict)│──▶│   (Lưu trữ vĩnh viễn)   │ │
│  │  buffalo_l)   │   │ cosine match │   │   Vector 512D + ORM     │ │
│  └──────────────┘   └──────────────┘   └──────────────────────────┘ │
│         │                                        ▲                   │
│         │              BackgroundTask             │                   │
│         └────────────────────────────────────────┘                   │
│              (ghi DB bất đồng bộ, không block API)                   │
└──────────────────────────────────────────────────────────────────────┘
```

**Tóm gọn:** Client detect mặt → Server nhúng vector ArcFace 512D → So với cache RAM (numpy) → Trả kết quả tức thì → Ghi DB ở nền.

---

## 📂 Cấu trúc thư mục

```
face-recognition-attendance-system/
│
├── app/                          # ★ Package chính (FastAPI backend)
│   ├── __init__.py
│   ├── config.py                 # Cấu hình: DB URL, ngưỡng cosine, model name
│   ├── database.py               # SQLAlchemy engine + session factory
│   ├── models.py                 # ORM models (7 bảng)
│   ├── schemas.py                # Pydantic request/response schemas
│   ├── face_service.py           # ★★★ LÕI AI: InsightFace embed + matching pgvector
│   ├── ram_cache.py              # RAM cache thay Redis (Python dict + numpy)
│   ├── redis_client.py           # [Legacy] Redis client async (không dùng nữa)
│   ├── main.py                   # FastAPI app entry point + warmup
│   ├── web.py                    # [Legacy] Flask web server (bản cũ)
│   └── routers/                  # FastAPI routers
│       ├── attendance.py         # ★★ TRỌNG TÂM: /start, /recognize, /recognize-fast, /stop
│       ├── students.py           # CRUD sinh viên + upload ảnh đăng ký
│       ├── courses.py            # CRUD lớp tín chỉ, lịch, enrollment
│       └── pages.py              # Serve HTML templates (Jinja2)
│
├── templates/                    # Jinja2 HTML templates
│   ├── base.html                 # Layout chung
│   ├── index.html                # Trang chủ
│   ├── register.html             # Đăng ký SV (upload ảnh)
│   ├── students.html             # Danh sách SV
│   ├── attendance.html           # ★ Trang điểm danh (webcam + MediaPipe)
│   ├── courses.html              # Quản lý lớp tín chỉ
│   └── history.html              # Lịch sử điểm danh
│
├── Images/                       # Ảnh gốc sinh viên (Images/{mã_SV}/)
├── tmp_uploads/                  # Thư mục tạm khi upload
├── migrations/                   # Alembic migrations
│
├── init_db.py                    # Script tạo DB + pgvector extension + tables
├── re_extract_embeddings.py      # Migration: chuyển embedding DeepFace → InsightFace
├── test_matching.py              # Test accuracy: self-match, noise, margin, attack
├── deepface_attendance.py        # [Legacy] Bản gốc DeepFace + OpenCV
├── attendance_gui.py             # [Legacy] Bản gốc Tkinter GUI
├── requirements.txt              # Danh sách thư viện
├── alembic.ini                   # Config Alembic
└── README.md                     # ← File này
```

---

## 📦 Danh sách thư viện cần cài đặt (CHI TIẾT)

### Nhóm 1: Core AI / Computer Vision

| Thư viện        | Version       | Công dụng                                                                                                    |
| --------------- | ------------- | ------------------------------------------------------------------------------------------------------------ |
| `insightface`   | ≥ 0.7.3       | **Lõi AI.** Chứa model buffalo_l (SCRFD detector + ArcFace embedder). Trích xuất vector khuôn mặt 512 chiều. |
| `onnxruntime`   | ≥ 1.18.0      | Runtime thực thi model ONNX (InsightFace dùng ONNX format). Bắt buộc phải cài.                               |
| `opencv-python` | ≥ 4.9.0       | Đọc/ghi ảnh, resize, padding, decode ảnh từ bytes. Là xương sống xử lý ảnh.                                  |
| `numpy`         | ≥ 1.24, < 2.0 | Tính toán ma trận: cosine similarity, normalize vector. Dùng ở mọi nơi.                                      |

### Nhóm 2: Database

| Thư viện          | Version | Công dụng                                                                             |
| ----------------- | ------- | ------------------------------------------------------------------------------------- |
| `SQLAlchemy`      | 2.0.36  | ORM — map Python class ↔ bảng PostgreSQL. Dùng query, insert, update.                 |
| `psycopg2-binary` | 2.9.10  | Driver kết nối Python → PostgreSQL. Bắt buộc.                                         |
| `alembic`         | 1.14.1  | Quản lý migration database (tạo/sửa bảng tự động).                                    |
| `pgvector`        | ≥ 0.3.6 | Extension hỗ trợ kiểu `Vector(512)` trong SQLAlchemy + toán tử `<=>` cosine distance. |

### Nhóm 3: Web Framework (FastAPI)

| Thư viện            | Version   | Công dụng                                                                             |
| ------------------- | --------- | ------------------------------------------------------------------------------------- |
| `fastapi`           | ≥ 0.115.0 | Web framework chính. Xử lý routing, dependency injection, background tasks.           |
| `uvicorn[standard]` | ≥ 0.34.0  | ASGI server chạy FastAPI. `[standard]` cài thêm `uvloop` + `httptools` cho hiệu năng. |
| `python-multipart`  | ≥ 0.0.9   | Parse form data (upload ảnh qua `multipart/form-data`). FastAPI yêu cầu.              |
| `jinja2`            | ≥ 3.1.0   | Template engine — render HTML cho giao diện web test.                                 |

### Nhóm 4: Utilities

| Thư viện | Version  | Công dụng                                                    |
| -------- | -------- | ------------------------------------------------------------ |
| `Pillow` | ≥ 10.0.0 | Thao tác ảnh bổ sung (crop, convert format). Dùng gián tiếp. |

---

## 🛠 Hướng dẫn cài đặt từng bước (Windows)

### Bước 0: Yêu cầu hệ thống

- **Python 3.11** (bắt buộc — file `.whl` được build cho 3.11)
- **PostgreSQL** đã cài và chạy (có extension `pgvector`)
- **Git** (để clone repo)

### Bước 1: Clone dự án

```bash
git clone <repo-url>
cd face-recognition-attendance-system
```

### Bước 2: Tạo Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Bước 3: Cài InsightFace (QUAN TRỌNG — tránh lỗi C++ Build Tools)

> ⚠️ **LƯU Ý QUAN TRỌNG KHI CÀI ĐẶT TRÊN WINDOWS** ⚠️
> Nếu bạn chạy thẳng `pip install -r requirements.txt` và gặp lỗi **`error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools"`** đi kèm **`Building wheel for insightface (pyproject.toml) did not run successfully`**, hãy làm theo 1 trong 2 giải pháp dưới đây thay vì cài trực tiếp:

**Giải pháp 1: Dùng file `.whl` prebuilt (Nhanh nhất)**

#### 3a. Tải file wheel prebuilt

- **Link:** [insightface-0.7.3-cp311-cp311-win_amd64.whl](https://huggingface.co/hanamizuki-ai/insightface-releases/resolve/main/insightface-0.7.3-cp311-cp311-win_amd64.whl)
- Mở trình duyệt → truy cập link → tải về → **copy file `.whl` vào thư mục gốc dự án**

#### 3b. Cài từ file đã tải

```bash
.venv\Scripts\pip install insightface-0.7.3-cp311-cp311-win_amd64.whl
```

#### 3c. Cài ONNX Runtime

```bash
.venv\Scripts\pip install onnxruntime
```

**Giải pháp 2: Cài Visual Studio C++ Build Tools (Chính thống)**
Nếu bạn không cài qua file `.whl` được, bạn sẽ bắt buộc phải cài trình biên dịch C++ của Microsoft:

1. Tải [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2. Chạy file cài đặt, chọn mục **"Desktop development with C++"** (chú ý chọn bản có Windows 10/11 SDK).
3. Đợi quá trình cài đặt hoàn tất (vài GB).
4. Khởi động lại máy và chạy lại lệnh `pip install -r requirements.txt`.

### Bước 4: Cài các thư viện còn lại

```bash
.venv\Scripts\pip install -r requirements.txt
```

### Bước 5: Kiểm tra cài đặt

```bash
python -c "import insightface; print('InsightFace OK:', insightface.__version__)"
python -c "import onnxruntime; print('ONNX Runtime OK:', onnxruntime.__version__)"
python -c "import fastapi; print('FastAPI OK:', fastapi.__version__)"
python -c "import cv2; print('OpenCV OK:', cv2.__version__)"
```

---

## 🗄 Khởi tạo Database

> ⚠️ **LỖI THƯỜNG GẶP:** `Connection refused (0x0000274D/10061)`
> Nếu chạy server mà terminal báo lỗi này, tức là **PostgreSQL chưa được bật** hoặc đang chạy sai port. Hãy đảm bảo bạn đã cài PostgreSQL, nó đang được **Start** trong Services (Windows), và port 5432 trên localhost là đúng.

### 5a. Tạo PostgreSQL database

Mở pgAdmin hoặc psql:

```sql
CREATE DATABASE attendance_db;
```

### 5b. Cấu hình kết nối

Mở file `app/config.py`, sửa dòng `DATABASE_URL`:

```python
DATABASE_URL = "postgresql+psycopg2://username:password@localhost:5432/attendance_db"
```

_Mặc định: user=`root`, password=`123456`_

### 5c. Chạy script khởi tạo bảng

```bash
python init_db.py
```

Script này tự động:

1. Kiểm tra database `attendance_db` tồn tại chưa → tạo nếu chưa có
2. Kích hoạt extension `pgvector`: `CREATE EXTENSION IF NOT EXISTS vector`
3. Tạo tất cả 7 bảng từ ORM models

---

## 🚀 Khởi chạy hệ thống

```bash
uvicorn app.main:app --reload --port 8000
```

| URL                                | Mô tả                                 |
| ---------------------------------- | ------------------------------------- |
| `http://localhost:8000/`           | Giao diện web (trang chủ)             |
| `http://localhost:8000/register`   | Đăng ký sinh viên (upload ≥ 10 ảnh)   |
| `http://localhost:8000/attendance` | Trang điểm danh (webcam + MediaPipe)  |
| `http://localhost:8000/docs`       | Swagger API docs (test API trực tiếp) |
| `http://localhost:8000/health`     | Health check + trạng thái cache       |

**Lưu ý:** Lần khởi chạy đầu tiên, server tự động load model InsightFace buffalo_l (~3-5 giây) ở background thread. Request đầu tiên không bị delay.

---

## 🧠 Giải thích Logic A → Z

### Tầng 1: Database Models — `app/models.py`

Hệ thống có **7 bảng**, chia 3 nhóm:

#### Nhóm Sinh viên

```
┌─────────────────┐            ┌──────────────────────┐
│    students      │ 1 ────── N │   student_photos      │
├─────────────────┤            ├──────────────────────┤
│ id (PK)          │            │ id (PK)               │
│ student_code     │            │ student_id (FK)       │
│ full_name        │            │ photo_path            │
│ class_name       │            │ face_embedding        │ ← Vector(512) pgvector
│ created_at       │            │ created_at            │
└─────────────────┘            └──────────────────────┘
```

- Mỗi sinh viên có **nhiều ảnh** (khuyến nghị ≥ 10 ảnh).
- Mỗi ảnh lưu **1 vector nhúng (embedding)** 512 chiều kiểu `Vector(512)` của pgvector.
- Càng nhiều ảnh (góc khác nhau, ánh sáng khác nhau) → hệ thống nhận diện càng chuẩn.

#### Nhóm Lớp tín chỉ

```
┌──────────────────┐          ┌──────────────────────┐
│    courses        │ 1 ──── N │   course_schedules    │
├──────────────────┤          ├──────────────────────┤
│ id (PK)           │          │ id (PK)               │
│ course_code       │          │ course_id (FK)        │
│ course_name       │          │ day_of_week (0-6)     │
│ room              │          │ start_time            │
│ semester          │          │ end_time              │
└──────────────────┘          └──────────────────────┘
        │
        │ N ──── N (qua bảng trung gian)
        ▼
┌──────────────────────┐
│  course_enrollments   │
├──────────────────────┤
│ id (PK)               │
│ course_id (FK)        │
│ student_id (FK)       │
│ UNIQUE(course, student)│
└──────────────────────┘
```

- Mỗi lớp có lịch học (thứ, giờ bắt đầu/kết thúc).
- Bảng `course_enrollments` là quan hệ N-N giữa sinh viên và lớp.

#### Nhóm Điểm danh

```
┌───────────────────────┐          ┌──────────────────────┐
│  attendance_sessions   │ 1 ──── N │  attendance_records   │
├───────────────────────┤          ├──────────────────────┤
│ id (PK)                │          │ id (PK)               │
│ session_name           │          │ session_id (FK)       │
│ subject                │          │ student_id (FK)       │
│ session_date           │          │ check_in_time         │
│ course_id (FK, nullable)│         │ status ("Present")    │
│ created_at             │          │ confidence (float)    │ ← Cosine similarity
└───────────────────────┘          │ UNIQUE(session, student)│
                                   └──────────────────────┘
```

- Mỗi buổi học = 1 `attendance_session`.
- Mỗi SV được điểm danh = 1 `attendance_record`, lưu kèm `confidence` (điểm cosine).
- Constraint `UNIQUE(session_id, student_id)` đảm bảo **không điểm danh trùng**.

---

### Tầng 2: Face Service (Lõi AI) — `app/face_service.py`

Đây là file **quan trọng nhất**. Mọi logic AI nằm ở đây.

#### 2.1. InsightFace Singleton — `get_face_app()`

```python
_face_app = None   # Biến toàn cục, chỉ load 1 lần

def get_face_app():
    global _face_app
    if _face_app is None:
        _face_app = insightface.app.FaceAnalysis(
            name="buffalo_l",                    # Model buffalo_l
            providers=["CPUExecutionProvider"],   # Chạy trên CPU
        )
        _face_app.prepare(ctx_id=0, det_size=(640, 640))  # Kích thước detect
    return _face_app
```

**Giải thích:**

- `buffalo_l` là bộ model chứa 2 thành phần:
  - **SCRFD** — Face Detection: phát hiện vị trí khuôn mặt trong ảnh (~10ms/frame)
  - **ArcFace** — Face Embedding: chuyển khuôn mặt thành vector 512 chiều (~80ms/frame)
- Dùng singleton pattern: chỉ load model 1 lần vào RAM, các request sau dùng lại.
- `det_size=(640, 640)` — resize ảnh input về 640×640 trước khi detect.

#### 2.2. Trích xuất Embedding từ file ảnh — `extract_embedding()`

Dùng khi **đăng ký sinh viên** (không cần realtime):

```
Ảnh gốc (file) → cv2.imread() → InsightFace app.get()
                                      ↓
                                  Phát hiện mặt (SCRFD)
                                      ↓
                                  Align mặt (xoay, căn chỉnh)
                                      ↓
                                  Trích xuất embedding (ArcFace)
                                      ↓
                                  Vector 512 chiều (list[float])
```

**Logic chi tiết:**

1. Đọc ảnh bằng OpenCV
2. `app.get(img)` → trả về danh sách faces, mỗi face có: `bbox`, `det_score`, `embedding`
3. Lấy mặt có `det_score` cao nhất (nếu ảnh có nhiều mặt)
4. Kiểm tra `det_score >= FACE_CONFIDENCE_MIN` (0.6) — loại bỏ false positive
5. Trả về `embedding.tolist()` — vector 512D

#### 2.3. Trích xuất từ Face Crops (Webcam) — `extract_embeddings_from_crops()`

Dùng cho endpoint `/recognize-fast`. Client đã crop sẵn mặt bằng MediaPipe:

```
Face crop (từ MediaPipe)
    ↓
┌─────────────── Strategy 1 (ưu tiên) ───────────────────┐
│ Thêm padding 50% → SCRFD detect lại → Align → ArcFace │ ← Chất lượng CAO
│ (pad_ratio = 0.5 mỗi bên)                              │
└─────────────────────────────────────────────────────────┘
    ↓ nếu SCRFD fail (mặt crop quá chặt)
┌─────────────── Strategy 2 (dự phòng) ──────────────────┐
│ Resize thẳng về 112×112 → blobFromImage → ArcFace     │ ← Chất lượng THẤP hơn
│ (không qua alignment)                                   │    quality × 0.7
└─────────────────────────────────────────────────────────┘
```

**Tại sao cần 2 strategy?**

- MediaPipe crop rất chặt (sát viền mặt) → SCRFD cần thêm context xung quanh (trán, cằm, tai) mới detect được.
- Padding 50% = thêm viền đen xung quanh ảnh crop → SCRFD thấy đủ context → align chuẩn → embedding tốt.
- Nếu vẫn fail → dùng thẳng ArcFace model, chấp nhận giảm chất lượng (nhân quality × 0.7).

#### 2.4. Face Quality Scoring — `_compute_face_quality()`

```python
def _compute_face_quality(original_face_size: int) -> float:
    # face >= 100px (cấu hình QUALITY_FACE_SIZE_GOOD) → quality = 1.0
    # face <= 60px  (cấu hình QUALITY_FACE_SIZE_MIN)  → quality = 0.0
    # ở giữa → tỷ lệ tuyến tính
```

**Ý nghĩa:** Mặt lớn (gần camera) → quality cao → ngưỡng match dễ hơn.  
Mặt nhỏ (xa camera) → quality thấp → ngưỡng match khắt khe hơn (tránh nhầm).

#### 2.5. Adaptive Threshold — `_adaptive_threshold()`

```python
def _adaptive_threshold(face_quality, base_threshold=0.45):
    penalty = (1.0 - face_quality) * 0.08    # QUALITY_THRESHOLD_PENALTY
    return base_threshold + penalty
```

**Ví dụ:**
| Face size | Quality | Threshold | Ý nghĩa |
|---|---|---|---|
| 150px | 1.0 | 0.45 | Mặt rõ → dùng ngưỡng gốc |
| 80px | 0.4 | 0.498 | Mặt trung bình → nâng ngưỡng |
| 50px | 0.0 | 0.53 | Mặt mờ → ngưỡng rất chặt |

#### 2.6. So khớp pgvector V2 — `find_best_match_pgvector_v2()`

Đây là thuật toán **chống nhầm người** cốt lõi:

```
Embedding đầu vào
    ↓
pgvector: SELECT TOP 20 gần nhất (cosine distance <=>)
    ↓
Gom nhóm theo student_id
    ↓
Mỗi SV: lấy trung bình top-3 ảnh khớp nhất (aggregation)
    ↓
Xếp hạng: SV có agg_score cao nhất = best
    ↓
Kiểm tra 3 điều kiện:
    ├── (1) agg_score >= adaptive_threshold?
    ├── (2) margin (best - second) >= 0.03?
    └── (3) Nếu cả 2 OK → ✓ MATCH, ngược lại → ✗ REJECT
```

**Tại sao cần Margin Check?**  
Giả sử SV A và SV B giống nhau (cùng kiểu tóc, cùng kính):

- Nếu chỉ check threshold: SV A score=0.52, SV B score=0.50 → match SV A ✗ DỄ NHẦM
- Với margin check: margin = 0.52 - 0.50 = 0.02 < 0.03 → **REJECT** ✓ AN TOÀN

**Tại sao lấy trung bình top-3?**  
Mỗi SV có ~10 ảnh. Nếu chỉ lấy score cao nhất, có thể 1 ảnh đặc biệt "trùng" dẫn đến nhầm. Trung bình top-3 ổn định hơn — SV thật sự giống mới có nhiều ảnh cùng score cao.

---

### Tầng 3: RAM Cache — `app/ram_cache.py`

**Vấn đề:** Mỗi frame webcam phải so embedding với TẤT CẢ ảnh SV trong DB → query DB liên tục → chậm.

**Giải pháp:** Khi bắt đầu phiên điểm danh, load hết embeddings của lớp đó lên RAM (Python `dict` + `numpy array`):

```python
_cache = {}   # dict toàn cục

_cache[course_id] = {
    "embeddings":  np.ndarray (N, 512),    # Ma trận N vectors × 512 chiều
    "student_ids": [1, 1, 1, 2, 2, ...],   # Mỗi embedding thuộc SV nào
    "photo_ids":   [10, 11, 12, 20, 21..], # ID ảnh tương ứng
    "metadata":    {
        1: {"student_code": "21IT001", "full_name": "Nguyễn Văn A"},
        2: {"student_code": "21IT002", "full_name": "Trần Thị B"},
    },
    "attended":    {1},                     # set() SV đã điểm danh
    "loaded_at":   "2026-03-28T00:00:00",
    "n_students":  2,
}
```

**Tốc độ matching trên cache (~1-2ms):**

```python
# Vectorized cosine similarity (numpy, KHÔNG loop Python)
query_norm = query / norm(query)              # Normalize query
emb_normed = emb_matrix / norms               # Normalize tất cả DB
similarities = emb_normed @ query_norm         # Dot product = cosine sim
# → Trả về array (N,) scores trong ~1ms cho 1000 embeddings
```

**So sánh:**
| Phương pháp | Tốc độ mỗi frame |
|---|---|
| Query DB pgvector mỗi frame | ~50-100ms |
| RAM Cache (numpy vectorized) | ~1-2ms |

---

### Tầng 4: API Routers — `app/routers/`

#### 4.1. `attendance.py` — ★ ROUTER QUAN TRỌNG NHẤT

**Endpoint 1: `POST /api/attendance/start`**  
_Bắt đầu phiên điểm danh_

```
Input: { course_id: 1 }

Logic:
  1. Kiểm tra course tồn tại
  2. Lấy danh sách SV đăng ký lớp (course_enrollments)
  3. Query tất cả student_photos (embedding) của các SV đó
  4. Load hết lên RAM cache: load_class_embeddings_to_cache()
  5. Tạo attendance_session mới trong DB
  6. Lưu session_id vào _active_sessions dict

Output: { session_id, total_students, cached_embeddings }
```

**Endpoint 2: `POST /api/attendance/recognize`**  
_Nhận diện từ full frame (server tự detect mặt)_

```
Input: session_id + image (full frame 640×480)

Logic:
  1. Decode ảnh bytes → numpy array
  2. InsightFace app.get(frame) → detect + embed TẤT CẢ mặt trong frame
  3. Với mỗi mặt → _match_from_cache() so khớp trên RAM
  4. Nếu match:
     a. Kiểm tra đã điểm danh chưa (is_attended)
     b. Nếu chưa → mark_attended() (RAM) + background_task ghi DB
  5. Trả response ngay (không chờ ghi DB)

Output: { faces: [...], new_attended: [...], elapsed_ms }
```

**Endpoint 3: `POST /api/attendance/recognize-fast`** ★ NHANH NHẤT  
_Nhận diện từ face crops (client MediaPipe đã detect sẵn)_

```
Input: session_id + faces[] (file uploads, mỗi file = 1 mặt crop)
       + face_positions (JSON vị trí trên frame)

Logic:
  1. Decode từng face crop → numpy array
  2. extract_embeddings_from_crops(crops):
     → Strategy 1: pad 50% → SCRFD + ArcFace
     → Strategy 2: resize 112×112 → ArcFace trực tiếp
  3. _match_from_cache() so khớp trên RAM
  4. Kết quả tương tự /recognize

Tại sao nhanh hơn /recognize?
  - Bỏ bước detect trên server (client đã detect bằng MediaPipe)
  - Ảnh gửi nhỏ hơn (200×200 thay vì 640×480)
  - Phù hợp cho real-time 30fps
```

**Endpoint 4: `POST /api/attendance/stop`**  
_Kết thúc phiên → clear cache_

```
Logic:
  1. Lấy attended_set trước khi xóa
  2. clear_class_cache(course_id) → giải phóng RAM
  3. Xóa session khỏi _active_sessions
```

#### Hàm `_match_from_cache()` — So khớp trên RAM cache

Đây là bản tối ưu tốc độ của `find_best_match_pgvector_v2()`, chạy trên numpy thay vì SQL:

```python
def _match_from_cache(face_embedding, cache_data, face_quality=1.0):
    # 1. Normalize query vector
    query_norm = query / norm(query)

    # 2. Normalize tất cả cached embeddings
    emb_normed = emb_matrix / norms           # (N, 512)

    # 3. Cosine similarity = dot product (đã normalize)
    similarities = emb_normed @ query_norm    # (N,) — ~1ms

    # 4. Aggregate per student (trung bình top-3)
    for i, sim in enumerate(similarities):
        students[student_ids[i]]["scores"].append(sim)

    # 5. Xếp hạng → best, second
    # 6. Adaptive threshold check
    # 7. Margin check (best.agg - second.agg >= 0.03)
    # 8. Return match hoặc None
```

#### 4.2. `students.py` — CRUD Sinh viên

| Endpoint                 | Method | Mô tả                        |
| ------------------------ | ------ | ---------------------------- |
| `/api/students`          | GET    | Danh sách SV                 |
| `/api/students/register` | POST   | Đăng ký SV + upload ≥ 10 ảnh |
| `/api/students/{id}`     | DELETE | Xóa SV                       |

**Flow đăng ký SV:**

```
Upload 10 ảnh → Lưu tạm vào tmp_uploads/
    ↓
face_service.register_student():
    ├── Tạo record bảng students
    ├── Copy ảnh vào Images/{student_code}/
    ├── Với mỗi ảnh:
    │   ├── extract_embedding() → vector 512D
    │   ├── Nếu không detect mặt → bỏ qua ảnh
    │   └── Lưu StudentPhoto (path + embedding) vào DB
    └── Nếu không có ảnh hợp lệ → rollback
```

#### 4.3. `courses.py` — CRUD Lớp tín chỉ

| Endpoint                             | Method     | Mô tả                 |
| ------------------------------------ | ---------- | --------------------- |
| `/api/courses`                       | GET/POST   | Danh sách / Tạo lớp   |
| `/api/courses/{id}`                  | GET/DELETE | Chi tiết / Xóa lớp    |
| `/api/courses/{id}/schedules`        | GET/POST   | Lịch học              |
| `/api/courses/{id}/enrollments`      | POST       | Thêm 1 SV vào lớp     |
| `/api/courses/{id}/enrollments/bulk` | POST       | Thêm nhiều SV vào lớp |
| `/api/courses/{id}/students`         | GET        | Danh sách SV của lớp  |

#### 4.4. `pages.py` — Serve HTML

Mỗi route (`/`, `/register`, `/attendance`, ...) render Jinja2 template tương ứng. Truyền dữ liệu từ DB vào context cho template hiển thị.

---

### Tầng 5: Web UI (Jinja2 Templates)

#### `attendance.html` — Trang điểm danh (★ quan trọng)

Trang này chứa logic JavaScript phía client:

```
1. User chọn lớp → POST /api/attendance/start → nhận session_id + load cache
2. Bật webcam getUserMedia()
3. Mỗi frame:
   a. MediaPipe Face Detection detect mặt realtime 30fps (chạy trên browser)
   b. Crop từng khuôn mặt thành blob
   c. POST /api/attendance/recognize-fast (gửi face crops)
   d. Nhận kết quả → hiển thị tên + confidence lên canvas
4. User bấm Stop → POST /api/attendance/stop → clear cache
```

**Tại sao dùng MediaPipe ở client?**

- Detect mặt ở browser bằng WebAssembly → ~5ms/frame, 30fps
- Chỉ gửi mặt đã crop (~200×200) thay vì full frame (640×480)
- Giảm bandwidth + giảm CPU server

---

## 🔄 Luồng hoạt động End-to-End

### A. Luồng ĐĂNG KÝ sinh viên (1 lần)

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│ User upload  │────▶│ FastAPI nhận ảnh │────▶│ InsightFace          │
│ ≥ 10 ảnh     │     │ lưu tmp_uploads/ │     │ extract_embedding()  │
└─────────────┘     └──────────────────┘     │ ảnh → vector 512D    │
                                              └──────────┬───────────┘
                                                         │
                               ┌─────────────────────────▼──────────┐
                               │ PostgreSQL                          │
                               │ INSERT students + student_photos   │
                               │ (photo_path + face_embedding)      │
                               │ + Copy ảnh → Images/{code}/        │
                               └────────────────────────────────────┘
```

### B. Luồng ĐIỂM DANH (realtime, mỗi buổi học)

```
╔══════════════════════════════════════════════════════════════════╗
║ PHASE 1: Bắt đầu phiên (1 lần)                                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  POST /start {course_id: 1}                                     ║
║      ↓                                                           ║
║  DB: SELECT student_photos WHERE student IN lớp 1               ║
║      ↓                                                           ║
║  RAM Cache: load numpy matrix (N × 512)                         ║
║  + Tạo attendance_session                                       ║
║      ↓                                                           ║
║  Response: { session_id: 5, cached: 100 embeddings }            ║
╚══════════════════════════════════════════════════════════════════╝
                           ↓
╔══════════════════════════════════════════════════════════════════╗
║ PHASE 2: Nhận diện liên tục (lặp lại mỗi frame)                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Browser: MediaPipe → detect mặt → crop → POST /recognize-fast ║
║      ↓                                                           ║
║  Server: InsightFace embed crop → vector 512D                   ║
║      ↓                                                           ║
║  RAM: numpy cosine similarity → aggregate → threshold → margin  ║
║      ↓                                                           ║
║  ┌──── MATCH ─────┐    ┌──── NO MATCH ─────┐                   ║
║  │ mark_attended() │    │ Return recognized  │                   ║
║  │ BackgroundTask  │    │ = false            │                   ║
║  │ → INSERT DB     │    └───────────────────┘                   ║
║  └────────────────┘                                             ║
║      ↓                                                           ║
║  Response: { recognized: true, "Nguyễn Văn A", confidence: 85% }║
║  (~90-150ms tổng, 1-2ms cho matching)                           ║
╚══════════════════════════════════════════════════════════════════╝
                           ↓
╔══════════════════════════════════════════════════════════════════╗
║ PHASE 3: Kết thúc phiên (1 lần)                                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  POST /stop {session_id: 5}                                     ║
║      ↓                                                           ║
║  clear_class_cache() → Giải phóng RAM                           ║
║  Kết quả đã ghi vào DB ở background tasks                      ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 🔧 Các file phụ trợ

### `app/config.py` — Tất cả hằng số cấu hình

| Hằng số                     | Giá trị                                 | Ý nghĩa                                              |
| --------------------------- | --------------------------------------- | ---------------------------------------------------- |
| `DATABASE_URL`              | `postgresql+psycopg2://root:123456@...` | Chuỗi kết nối DB                                     |
| `INSIGHTFACE_MODEL`         | `"buffalo_l"`                           | Tên model InsightFace                                |
| `COSINE_THRESHOLD`          | `0.45`                                  | Ngưỡng cosine tối thiểu để nhận diện                 |
| `FACE_CONFIDENCE_MIN`       | `0.6`                                   | Ngưỡng confidence phát hiện mặt (lọc false positive) |
| `FACE_MIN_PIXELS`           | `80`                                    | Kích thước mặt tối thiểu (pixel). Dưới → quá xa      |
| `TOP_K_MATCHES`             | `20`                                    | Số kết quả lấy từ pgvector để phân tích              |
| `MATCH_MARGIN_MIN`          | `0.03`                                  | Khoảng cách tối thiểu giữa SV #1 và SV #2            |
| `STUDENT_AGG_TOP_N`         | `3`                                     | Trung bình top-N ảnh tốt nhất mỗi SV                 |
| `QUALITY_FACE_SIZE_GOOD`    | `100` px                                | Mặt ≥ 100px → quality = 1.0                          |
| `QUALITY_FACE_SIZE_MIN`     | `60` px                                 | Mặt ≤ 60px → quality = 0.0                           |
| `QUALITY_THRESHOLD_PENALTY` | `0.08`                                  | Penalty tối đa khi quality = 0                       |

### `re_extract_embeddings.py` — Migration DeepFace → InsightFace

Khi chuyển từ DeepFace (Facenet512) sang InsightFace (ArcFace), embedding cũ không tương thích:

- Facenet512: norm ≈ 23, không gian vector khác
- ArcFace: norm ≈ 1, không gian vector khác

Script này duyệt tất cả `student_photos` → đọc lại ảnh gốc → extract embedding mới bằng InsightFace → UPDATE DB.

### `test_matching.py` — Test độ chính xác

6 test cases:
| TC | Tên | Mô tả |
|---|---|---|
| TC1 | Self-match | Embedding vs chính nó → phải match 100% |
| TC2 | Cross-student | Mỗi embedding phải match đúng SV, không nhầm |
| TC3 | Noise simulation | Thêm Gaussian noise mô phỏng khoảng cách xa (σ=0.02→0.20) |
| TC4 | Margin analysis | Kiểm tra khoảng cách SV #1 vs SV #2 ở các mức noise |
| TC5 | Adaptive threshold | Quality score → threshold hoạt động đúng |
| TC6 | Worst case attack | Blend embedding SV A + SV B → kiểm tra V2 có chặn được |

### `app/redis_client.py` — [Legacy, không sử dụng]

Bản cũ dùng Redis async để cache. Đã thay bằng `ram_cache.py` (Python dict thuần túy, không cần service ngoài).

### `app/web.py` — [Legacy, không sử dụng]

Bản Flask cũ. Đã thay hoàn toàn bằng FastAPI (`app/main.py` + `app/routers/`).

---

## 🚀 Cách áp dụng sang dự án khác

### Những gì bạn CẦN giữ (core)

1. **`app/face_service.py`** — Copy nguyên. Đây là toàn bộ logic AI.
2. **`app/ram_cache.py`** — Copy nguyên. Cache tối ưu tốc độ.
3. **`app/config.py`** — Copy và sửa `DATABASE_URL`, các threshold theo nhu cầu.
4. **Models:** `Student`, `StudentPhoto` (bảng + embedding), `AttendanceSession`, `AttendanceRecord`.
5. **Logic `_match_from_cache()`** trong `app/routers/attendance.py` — thuật toán matching cốt lõi.

### Những gì bạn CÓ THỂ bỏ

- `web.py`, `deepface_attendance.py`, `attendance_gui.py` — bản legacy.
- `redis_client.py` — đã thay bằng RAM cache.
- `templates/` — nếu frontend dùng React/Next.js thì không cần Jinja2.
- `re_extract_embeddings.py`, `test_matching.py` — chỉ cần trong phase test.

### Checklist khi port sang dự án mới

1. ☐ Cài `insightface` (file `.whl`) + `onnxruntime` + `opencv-python` + `numpy`
2. ☐ PostgreSQL có `pgvector` extension
3. ☐ Tạo bảng `students` + `student_photos` (cột `face_embedding Vector(512)`)
4. ☐ Copy `face_service.py` → sửa import theo cấu trúc dự án mới
5. ☐ Copy `ram_cache.py` → không cần sửa
6. ☐ Tạo endpoint đăng ký SV (upload ≥ 10 ảnh → extract_embedding → lưu DB)
7. ☐ Tạo endpoint start session (load cache từ DB)
8. ☐ Tạo endpoint recognize (embed + match cache + background ghi DB)
9. ☐ Tạo endpoint stop (clear cache)
10. ☐ Frontend: tích hợp MediaPipe hoặc gửi full frame
