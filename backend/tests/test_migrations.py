from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import inspect

from app.db import Base, engine


def test_migrations_match_models():
    """Fails when a model changes without a new migration."""
    with engine.connect() as conn:
        assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []


def test_downgrade_base_then_upgrade_head(alembic_cfg):
    command.downgrade(alembic_cfg, "base")
    with engine.connect() as conn:
        assert inspect(conn).get_table_names() == ["alembic_version"]
    command.upgrade(alembic_cfg, "head")
    with engine.connect() as conn:
        assert set(inspect(conn).get_table_names()) >= {t.name for t in Base.metadata.sorted_tables}
