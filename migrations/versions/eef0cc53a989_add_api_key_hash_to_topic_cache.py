"""add_api_key_hash_to_topic_cache

Revision ID: eef0cc53a989
Revises: 8a532448d6b0
Create Date: 2026-04-17 02:35:12.365743

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'eef0cc53a989'
down_revision: Union[str, Sequence[str], None] = '8a532448d6b0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — per-user scoping on the semantic report cache (C-3)."""
    op.add_column(
        "topic_cache",
        sa.Column("api_key_hash", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_topic_cache_api_key_hash",
        "topic_cache",
        ["api_key_hash"],
    )


def downgrade() -> None:
    op.drop_index("ix_topic_cache_api_key_hash", table_name="topic_cache")
    op.drop_column("topic_cache", "api_key_hash")
