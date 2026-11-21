"""add rag_set corpus fields

Revision ID: a3f8c1d2e9b4
Revises: 557f3078bc28
Create Date: 2026-11-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a3f8c1d2e9b4'
down_revision: Union[str, Sequence[str], None] = '557f3078bc28'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'rag_sets',
        sa.Column('kind', sa.String(length=16), nullable=False,
                  server_default=sa.text("'user'")),
    )
    op.add_column(
        'rag_sets',
        sa.Column('visibility', sa.String(length=16), nullable=False,
                  server_default=sa.text("'private'")),
    )
    op.add_column(
        'rag_sets',
        sa.Column('index_name', sa.String(length=128), nullable=False,
                  server_default=sa.text("'kb-v2'")),
    )
    op.create_check_constraint(
        'ck_rag_kind', 'rag_sets', "kind IN ('user','technical')"
    )
    op.create_check_constraint(
        'ck_rag_visibility', 'rag_sets', "visibility IN ('private','shared','public')"
    )


def downgrade() -> None:
    op.drop_constraint('ck_rag_kind', 'rag_sets', type_='check')
    op.drop_constraint('ck_rag_visibility', 'rag_sets', type_='check')
    op.drop_column('rag_sets', 'index_name')
    op.drop_column('rag_sets', 'visibility')
    op.drop_column('rag_sets', 'kind')
