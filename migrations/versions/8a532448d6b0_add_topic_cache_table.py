"""add_topic_cache_table

Revision ID: 8a532448d6b0
Revises: a0fa68762bab
Create Date: 2026-04-17 01:09:28.828084

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8a532448d6b0'
down_revision: Union[str, Sequence[str], None] = 'a0fa68762bab'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: topic-level semantic cache for entire finished reports."""
    op.create_table(
        "topic_cache",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("topic", sa.String(500), nullable=False),
        sa.Column("final_report", sa.Text(), nullable=False),
        sa.Column("citations_json", sa.Text(), nullable=True),
        sa.Column("hit_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute("ALTER TABLE topic_cache ADD COLUMN IF NOT EXISTS topic_vec vector(1536)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS topic_cache_vec_idx "
        "ON topic_cache USING ivfflat (topic_vec vector_cosine_ops) WITH (lists = 10)"
    )


def downgrade() -> None:
    op.drop_table("topic_cache")
