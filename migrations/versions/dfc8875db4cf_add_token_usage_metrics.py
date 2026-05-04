"""add token usage metrics

Revision ID: dfc8875db4cf
Revises: 14199dc70f7f
Create Date: 2026-04-15 22:44:16.453773

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dfc8875db4cf'
down_revision: Union[str, Sequence[str], None] = '14199dc70f7f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("research_sessions", sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("research_sessions", sa.Column("completion_tokens", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("research_sessions", sa.Column("total_tokens", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("research_sessions", sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0.0"))


def downgrade() -> None:
    op.drop_column("research_sessions", "cost_usd")
    op.drop_column("research_sessions", "total_tokens")
    op.drop_column("research_sessions", "completion_tokens")
    op.drop_column("research_sessions", "prompt_tokens")
