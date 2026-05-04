"""add_parent_doc_cols_to_documents

Revision ID: 7716b8b9e9d1
Revises: abece50dbc97
Create Date: 2026-04-17 02:55:43.352157

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7716b8b9e9d1'
down_revision: Union[str, Sequence[str], None] = 'abece50dbc97'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — parent-document context columns (G-9)."""
    op.add_column("documents", sa.Column("parent_doc_id", sa.UUID(), nullable=True))
    op.add_column("documents", sa.Column("parent_content", sa.Text(), nullable=True))
    op.add_column("documents", sa.Column("chunk_offset_start", sa.Integer(), nullable=True))
    op.add_column("documents", sa.Column("chunk_offset_end", sa.Integer(), nullable=True))
    op.create_index("ix_documents_parent_doc_id", "documents", ["parent_doc_id"])


def downgrade() -> None:
    op.drop_index("ix_documents_parent_doc_id", table_name="documents")
    op.drop_column("documents", "chunk_offset_end")
    op.drop_column("documents", "chunk_offset_start")
    op.drop_column("documents", "parent_content")
    op.drop_column("documents", "parent_doc_id")
