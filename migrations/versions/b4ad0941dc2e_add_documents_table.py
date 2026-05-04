"""add_documents_table

Revision ID: b4ad0941dc2e
Revises: dfc8875db4cf
Create Date: 2026-04-16 00:02:23.243031

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b4ad0941dc2e'
down_revision: Union[str, Sequence[str], None] = 'dfc8875db4cf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("topic", sa.String(500), nullable=False),
        sa.Column("section", sa.String(500), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", sa.Text(), nullable=True),  # stored as JSON string, cast to vector in queries
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_vec vector(1536)")
    op.execute("CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING ivfflat (embedding_vec vector_cosine_ops) WITH (lists = 10)")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("documents")
