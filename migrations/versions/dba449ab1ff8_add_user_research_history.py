"""add_user_research_history

Revision ID: dba449ab1ff8
Revises: 7716b8b9e9d1
Create Date: 2026-04-17 02:58:35.787692

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dba449ab1ff8'
down_revision: Union[str, Sequence[str], None] = '7716b8b9e9d1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — per-user episodic memory (G-12)."""
    op.create_table(
        "user_research_history",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("api_key_hash", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("topic", sa.String(length=500), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute(
        "ALTER TABLE user_research_history ADD COLUMN IF NOT EXISTS topic_vec vector(1536)"
    )
    op.create_index(
        "ix_user_research_history_api_key_hash",
        "user_research_history",
        ["api_key_hash"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_user_research_history_api_key_hash", table_name="user_research_history"
    )
    op.drop_table("user_research_history")
