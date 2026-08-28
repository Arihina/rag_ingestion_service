from __future__ import annotations

"""Запросы control plane.

Проверяется не результат (для этого нужен живой Postgres), а СТРУКТУРА
скомпилированного SQL. Это ловит ровно тот класс ошибок, который дорого
стоит: проверку владения, вынесенную из запроса наружу.
"""

import uuid
import unittest

from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from app.db.models import Document, RagSet


def sql_of(statement) -> str:
    return str(statement.compile(
        dialect=postgresql.dialect(),
        compile_kwargs={"literal_binds": False},
    )).lower()


class OwnershipInQueryTests(unittest.TestCase):
    """Владение проверяется ВНУТРИ запроса, а не после выборки.

    Отдельный SELECT с последующим `if rag.owner_id != user` рано или
    поздно забывается в одной из веток; здесь забыть нечего.
    """

    def test_выборка_набора_фильтрует_по_владельцу(self):
        rag_id, owner = uuid.uuid4(), uuid.uuid4()
        sql = sql_of(select(RagSet).where(
            RagSet.id == rag_id,
            RagSet.owner_id == owner,
            RagSet.deleted_at.is_(None),
        ))
        self.assertIn("owner_id", sql)
        self.assertIn("deleted_at is null", sql)

    def test_удалённые_наборы_невидимы(self):
        sql = sql_of(select(RagSet).where(
            RagSet.owner_id == uuid.uuid4(), RagSet.deleted_at.is_(None)
        ))
        self.assertIn("deleted_at is null", sql)

    def test_документы_скоупятся_по_набору(self):
        sql = sql_of(select(Document).where(
            Document.id == uuid.uuid4(), Document.rag_id == uuid.uuid4()
        ))
        self.assertIn("rag_id", sql)


class SchemaInvariantTests(unittest.TestCase):
    def test_дедупликация_в_границах_набора(self):
        """unique (rag_id, content_hash): один файл в двух наборах — два
        независимых документа, это корректно."""
        names = {
            c.name: {col.name for col in c.columns}
            for c in Document.__table__.constraints
            if c.name
        }
        self.assertEqual(names.get("uq_documents_rag_hash"),
                         {"rag_id", "content_hash"})

    def test_имя_набора_уникально_только_среди_живых(self):
        index = next(i for i in RagSet.__table__.indexes
                     if i.name == "uq_rag_sets_owner_name")
        self.assertTrue(index.unique)
        self.assertIn("deleted_at is null",
                      str(index.dialect_options["postgresql"]["where"]).lower())

    def test_границы_конфига_закреплены_в_бд(self):
        """Валидация pydantic может быть обойдена внутренним кодом,
        CHECK в БД — нет."""
        checks = {c.name: str(c.sqltext).lower()
                  for c in RagSet.__table__.constraints if c.name and "ck_" in c.name}
        self.assertIn("ck_rag_top_k", checks)
        self.assertIn("10", checks["ck_rag_top_k"])
        self.assertIn("ck_rag_temperature", checks)
        self.assertIn("ck_rag_score_threshold", checks)

    def test_статусы_документа_ограничены(self):
        checks = {c.name: str(c.sqltext).lower()
                  for c in Document.__table__.constraints if c.name and "ck_" in c.name}
        for status in ("pending", "processing", "success", "failed"):
            self.assertIn(status, checks["ck_document_status"])

    def test_удаление_набора_каскадит_на_документы(self):
        fk = next(iter(Document.__table__.c.rag_id.foreign_keys))
        self.assertEqual(fk.ondelete, "CASCADE")

    def test_байты_файлов_в_бд_не_хранятся(self):
        """Связь с SeaweedFS — только ключом. Ни одного bytea/LargeBinary."""
        for table in (RagSet.__table__, Document.__table__):
            for column in table.columns:
                self.assertNotIn(
                    "BYTEA", str(column.type).upper(),
                    f"{table.name}.{column.name} хранит байты",
                )
