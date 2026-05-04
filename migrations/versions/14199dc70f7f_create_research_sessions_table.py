"""create research sessions table

Revision ID: 14199dc70f7f
Revises: 
Create Date: 2026-04-15 22:08:47.940367

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '14199dc70f7f'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension (needed for future RAG embeddings)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "research_sessions",
        sa.Column("id", sa.UUID(as_uuid=True), primary_key=True),
        sa.Column("topic", sa.String(500), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("final_report", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("research_sessions")
