"""add_api_key_hash_to_sessions

Revision ID: aef5b97b1098
Revises: b4ad0941dc2e
Create Date: 2026-04-16 17:04:44.823629

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'aef5b97b1098'
down_revision: Union[str, Sequence[str], None] = 'b4ad0941dc2e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # SHA-256 hex = 64 chars. Nullable so dev-mode (no API key) still works.
    op.add_column(
        "research_sessions",
        sa.Column("api_key_hash", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_research_sessions_api_key_hash",
        "research_sessions",
        ["api_key_hash"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_research_sessions_api_key_hash", table_name="research_sessions")
    op.drop_column("research_sessions", "api_key_hash")
