"""Add course tables + link attendance_sessions to courses.

Revision ID: 003
Revises: 002
Create Date: 2026-03-14

Thêm:
- courses: lớp tín chỉ
- course_schedules: lịch học
- course_enrollments: SV đăng ký lớp
- attendance_sessions.course_id: FK lớp tín chỉ
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── courses ──
    op.create_table(
        "courses",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("course_code", sa.String(20), unique=True, nullable=False),
        sa.Column("course_name", sa.String(200), nullable=False),
        sa.Column("room", sa.String(50), nullable=True),
        sa.Column("semester", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # ── course_schedules ──
    op.create_table(
        "course_schedules",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "course_id",
            sa.Integer(),
            sa.ForeignKey("courses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("day_of_week", sa.Integer(), nullable=False),
        sa.Column("start_time", sa.Time(), nullable=False),
        sa.Column("end_time", sa.Time(), nullable=False),
    )

    # ── course_enrollments ──
    op.create_table(
        "course_enrollments",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "course_id",
            sa.Integer(),
            sa.ForeignKey("courses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "student_id",
            sa.Integer(),
            sa.ForeignKey("students.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("enrolled_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("course_id", "student_id", name="uq_course_student"),
    )

    # ── Thêm course_id vào attendance_sessions ──
    op.add_column(
        "attendance_sessions",
        sa.Column("course_id", sa.Integer(), sa.ForeignKey("courses.id"), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("attendance_sessions", "course_id")
    op.drop_table("course_enrollments")
    op.drop_table("course_schedules")
    op.drop_table("courses")
