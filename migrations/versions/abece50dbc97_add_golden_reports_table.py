"""add_golden_reports_table

Revision ID: abece50dbc97
Revises: eef0cc53a989
Create Date: 2026-04-17 02:45:50.532612

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'abece50dbc97'
down_revision: Union[str, Sequence[str], None] = 'eef0cc53a989'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — G-1 golden dataset for inter-annotator-agreement eval."""
    op.create_table(
        "golden_reports",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("topic", sa.String(500), nullable=False),
        sa.Column("report", sa.Text(), nullable=False),
        sa.Column("annotations_json", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_golden_reports_topic", "golden_reports", ["topic"])

    op.create_table(
        "eval_runs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("run_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("git_sha", sa.String(40), nullable=True),
        sa.Column("metric_scores_json", sa.Text(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_eval_runs_run_date", "eval_runs", ["run_date"])


def downgrade() -> None:
    op.drop_index("ix_eval_runs_run_date", table_name="eval_runs")
    op.drop_table("eval_runs")
    op.drop_index("ix_golden_reports_topic", table_name="golden_reports")
    op.drop_table("golden_reports")
