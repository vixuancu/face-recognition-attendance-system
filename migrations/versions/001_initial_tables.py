"""Initial tables: students, student_photos, attendance_sessions, attendance_records

Revision ID: 001
Revises: None
Create Date: 2026-03-05
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── pgvector extension ──
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── students ──
    op.create_table(
        "students",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("student_code", sa.String(20), unique=True, nullable=False),
        sa.Column("full_name", sa.String(100), nullable=False),
        sa.Column("class_name", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # ── student_photos ──
    op.create_table(
        "student_photos",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "student_id",
            sa.Integer(),
            sa.ForeignKey("students.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("photo_path", sa.String(500), nullable=False),
        sa.Column("face_embedding", Vector(512), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # ── attendance_sessions ──
    op.create_table(
        "attendance_sessions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_name", sa.String(200), nullable=True),
        sa.Column("subject", sa.String(100), nullable=True),
        sa.Column("session_date", sa.Date(), server_default=sa.func.current_date()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # ── attendance_records ──
    op.create_table(
        "attendance_records",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "session_id",
            sa.Integer(),
            sa.ForeignKey("attendance_sessions.id"),
            nullable=False,
        ),
        sa.Column(
            "student_id",
            sa.Integer(),
            sa.ForeignKey("students.id"),
            nullable=False,
        ),
        sa.Column("check_in_time", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("status", sa.String(20), server_default="Present"),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.UniqueConstraint("session_id", "student_id", name="uq_session_student"),
    )


def downgrade() -> None:
    op.drop_table("attendance_records")
    op.drop_table("attendance_sessions")
    op.drop_table("student_photos")
    op.drop_table("students")
