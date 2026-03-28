"""
Script thay thế đơn giản cho Alembic.
Chạy: python init_db.py

Tạo database "attendance_db" (nếu chưa có) rồi tạo tất cả bảng.
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from app.database import engine, Base
from app.models import (  # noqa: F401
    Student, StudentPhoto,
    Course, CourseSchedule, CourseEnrollment,
    AttendanceSession, AttendanceRecord,
)


DB_NAME = "attendance_db"
PG_USER = "root"
PG_PASS = "123456"
PG_HOST = "localhost"
PG_PORT = 5432


def create_database_if_not_exists():
    """Kết nối vào PostgreSQL mặc định, tạo DB nếu chưa có."""
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASS, dbname="postgres"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
    exists = cur.fetchone()

    if not exists:
        cur.execute(f'CREATE DATABASE "{DB_NAME}"')
        print(f"[✓] Database '{DB_NAME}' đã được tạo.")
    else:
        print(f"[i] Database '{DB_NAME}' đã tồn tại.")

    cur.close()
    conn.close()


def create_pgvector_extension():
    """Tạo pgvector extension trong database attendance_db."""
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASS, dbname=DB_NAME
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    print("[✓] pgvector extension đã được kích hoạt.")
    cur.close()
    conn.close()


def create_tables():
    """Tạo tất cả bảng từ ORM models."""
    Base.metadata.create_all(bind=engine)
    print("[✓] Tất cả bảng đã được tạo:")
    for table_name in Base.metadata.tables:
        print(f"    • {table_name}")


if __name__ == "__main__":
    print("=" * 50)
    print("  KHỞI TẠO DATABASE — v2.0 (FastAPI + Redis)")
    print("=" * 50)
    create_database_if_not_exists()
    create_pgvector_extension()
    create_tables()
    print("\n[✓] Hoàn tất! Chạy server:")
    print("    uvicorn app.main:app --reload --port 8000")
    print("    Mở: http://localhost:8000")
    print("    API docs: http://localhost:8000/docs")
