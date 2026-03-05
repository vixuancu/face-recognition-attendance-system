"""Migrate face_embedding from JSON to pgvector vector(512).

Revision ID: 002
Revises: 001
Create Date: 2026-03-05

Chuyển cột face_embedding từ json/jsonb → vector(512) của pgvector.
Dữ liệu cũ (JSON array) được convert sang vector type.
Thêm HNSW index cho cosine similarity search nhanh.
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Kích hoạt pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 2. Thêm cột mới kiểu vector
    op.execute("ALTER TABLE student_photos ADD COLUMN face_embedding_vec vector(512)")

    # 3. Migrate dữ liệu: JSON array → vector
    # Xử lý cả json và jsonb: convert về text rồi cast sang vector
    op.execute("""
        UPDATE student_photos
        SET face_embedding_vec = face_embedding::text::vector(512)
        WHERE face_embedding IS NOT NULL
    """)

    # 4. Xóa cột cũ, đổi tên cột mới
    op.execute("ALTER TABLE student_photos DROP COLUMN face_embedding")
    op.execute("ALTER TABLE student_photos RENAME COLUMN face_embedding_vec TO face_embedding")
    op.execute("ALTER TABLE student_photos ALTER COLUMN face_embedding SET NOT NULL")

    # 5. Tạo HNSW index cho cosine distance — tìm kiếm nhanh O(log n)
    op.execute("""
        CREATE INDEX idx_face_embedding_hnsw
        ON student_photos
        USING hnsw (face_embedding vector_cosine_ops)
    """)


def downgrade() -> None:
    # Xóa index
    op.execute("DROP INDEX IF EXISTS idx_face_embedding_hnsw")

    # Chuyển ngược vector → json
    op.execute("ALTER TABLE student_photos ADD COLUMN face_embedding_json json")
    op.execute("""
        UPDATE student_photos
        SET face_embedding_json = face_embedding::text::json
        WHERE face_embedding IS NOT NULL
    """)
    op.execute("ALTER TABLE student_photos DROP COLUMN face_embedding")
    op.execute("ALTER TABLE student_photos RENAME COLUMN face_embedding_json TO face_embedding")
    op.execute("ALTER TABLE student_photos ALTER COLUMN face_embedding SET NOT NULL")
