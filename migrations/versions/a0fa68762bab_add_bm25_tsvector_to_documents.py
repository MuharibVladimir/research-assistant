"""add_bm25_tsvector_to_documents

Revision ID: a0fa68762bab
Revises: aef5b97b1098
Create Date: 2026-04-17 01:00:18.661858

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a0fa68762bab'
down_revision: Union[str, Sequence[str], None] = 'aef5b97b1098'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add generated tsvector column + GIN index for BM25-style search."""
    op.execute(
        """
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS documents_content_tsv_idx "
        "ON documents USING GIN (content_tsv)"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS documents_content_tsv_idx")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS content_tsv")
