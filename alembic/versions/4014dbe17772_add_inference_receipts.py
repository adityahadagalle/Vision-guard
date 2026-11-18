"""add inference receipts

Revision ID: 4014dbe17772
Revises: 0cb6499cdd96
Create Date: 2026-11-17 16:59:04.546915

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4014dbe17772"
down_revision: str | Sequence[str] | None = "0cb6499cdd96"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "inference_receipts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("edge_device_id", sa.String(length=128), nullable=False),
        sa.Column("timestamp_iso", sa.String(length=64), nullable=False),
        sa.Column("sequence_number", sa.Integer(), nullable=False),
        sa.Column("model_sha256", sa.String(length=64), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("inference_output", sa.JSON(), nullable=False),
        sa.Column("previous_receipt_hash", sa.String(length=64), nullable=True),
        sa.Column("payload_hash", sa.String(length=64), nullable=False),
        sa.Column("signature_b64", sa.Text(), nullable=False),
        sa.Column("public_key_pem", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "edge_device_id", "sequence_number", name="uq_inference_device_sequence"
        ),
    )
    with op.batch_alter_table("inference_receipts", schema=None) as batch_op:
        batch_op.create_index(
            "idx_inference_receipts_device_seq",
            ["edge_device_id", "sequence_number"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_inference_receipts_edge_device_id"),
            ["edge_device_id"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_inference_receipts_payload_hash"),
            ["payload_hash"],
            unique=True,
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("inference_receipts", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_inference_receipts_payload_hash"))
        batch_op.drop_index(batch_op.f("ix_inference_receipts_edge_device_id"))
        batch_op.drop_index("idx_inference_receipts_device_seq")

    op.drop_table("inference_receipts")
