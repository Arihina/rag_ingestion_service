from __future__ import annotations

"""HTTP-слой платформенного API.

Postgres здесь не поднимается: сессия и репозиторий подменяются, а
проверяются коды ответов, скоупинг и форма тела — то, на что опирается
фронт и мастер.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.db import get_session
from app.db.repo import DocumentCounts
from tests.base import FixtureTestCase

OWNER = "11111111-1111-1111-1111-111111111111"
OTHER = "22222222-2222-2222-2222-222222222222"


def fake_rag(**overrides):
    rag = MagicMock()
    rag.id = uuid.UUID("33333333-3333-3333-3333-333333333333")
    rag.name = "Регламенты"
    rag.description = None
    rag.icon_key = None
    rag.prompt = None
    rag.temperature = 0.3
    rag.top_k = 5
    rag.score_threshold = 0.4
    rag.created_at = rag.updated_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for key, value in overrides.items():
        setattr(rag, key, value)
    return rag


class ApiTestCase(FixtureTestCase):
    """Приложение собирается без lifespan: модели и клиенты не нужны."""

    @classmethod
    def setUpClass(cls):
        # Настоящее приложение, а не сборка роутеров руками: так
        # проверяется и проводка, и обработчики ошибок (без них
        # невалидное тело давало бы 422 вместо 400). TestClient без
        # контекстного менеджера lifespan не запускает, поэтому модели
        # и клиенты не поднимаются.
        from app.main import app

        session = AsyncMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.rollback = AsyncMock()
        session.delete = AsyncMock()
        app.dependency_overrides[get_session] = lambda: session

        cls.session = session
        cls.client = TestClient(app, raise_server_exceptions=False)

    def head(self, user: str = OWNER) -> dict:
        return {"X-User-Id": user}


class AuthTests(ApiTestCase):
    def test_без_заголовка_401(self):
        self.assertEqual(self.client.get("/v1/platform/rags").status_code, 401)

    def test_не_uuid_даёт_401(self):
        # Значение ASCII намеренно: заголовки HTTP — latin-1, и кириллица
        # не дойдёт до приложения, упав раньше, в клиенте.
        for value in ("not-a-uuid", "12345", ""):
            with self.subTest(value=value):
                response = self.client.get(
                    "/v1/platform/rags", headers={"X-User-Id": value}
                )
                self.assertEqual(response.status_code, 401)


class OwnershipTests(ApiTestCase):
    def test_чужой_набор_даёт_404_а_не_403(self):
        """Существование чужого ресурса — тоже информация."""
        with patch("app.api.rags.repo.get_rag", AsyncMock(return_value=None)):
            response = self.client.get(
                f"/v1/platform/rags/{uuid.uuid4()}", headers=self.head(OTHER)
            )
        self.assertEqual(response.status_code, 404)

    def test_удаление_чужого_документа_404(self):
        with patch("app.api.rags.repo.get_rag", AsyncMock(return_value=None)):
            response = self.client.delete(
                f"/v1/platform/rags/{uuid.uuid4()}/documents/{uuid.uuid4()}",
                headers=self.head(OTHER),
            )
        self.assertEqual(response.status_code, 404)


class ValidationTests(ApiTestCase):
    def test_top_k_сверх_потолка(self):
        response = self.client.post(
            "/v1/platform/rags", headers=self.head(), json={"name": "x", "top_k": 50}
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]
                         ["type"], "invalid_request_error")

    def test_температура_вне_диапазона(self):
        response = self.client.post(
            "/v1/platform/rags", headers=self.head(), json={"name": "x", "temperature": 3}
        )
        self.assertEqual(response.status_code, 400)

    def test_порог_вне_диапазона(self):
        response = self.client.post(
            "/v1/platform/rags", headers=self.head(),
            json={"name": "x", "score_threshold": 1.5},
        )
        self.assertEqual(response.status_code, 400)

    def test_ошибка_в_формате_платформы(self):
        """Тот же формат, что у остальных агентов: фронт не должен
        заводить исключение для одного сервиса."""
        body = self.client.post(
            "/v1/platform/rags", headers=self.head(), json={"top_k": 50}
        ).json()
        self.assertEqual(set(body["error"]), {
                         "message", "type", "param", "code"})


class ReadTests(ApiTestCase):
    def test_ответ_несёт_производный_статус_и_счётчики(self):
        counts = DocumentCounts(
            total=3, ready=1, failed=1, pending=1, chunks=42)
        with patch("app.api.rags.repo.get_rag", AsyncMock(return_value=fake_rag())), \
                patch("app.api.rags.repo.counts_for", AsyncMock(return_value=counts)):
            body = self.client.get(
                f"/v1/platform/rags/{uuid.uuid4()}", headers=self.head()
            ).json()

        self.assertEqual(body["status"], "ready")
        self.assertEqual(body["chunks_total"], 42)
        self.assertTrue(body["has_pending"])
        self.assertEqual(body["documents"],
                         {"total": 3, "ready": 1, "failed": 1, "pending": 1})
        self.assertEqual(body["config"]["top_k"], 5)

    def test_список_берёт_счётчики_одним_запросом(self):
        """Иначе N+1 на каждой отрисовке списка наборов."""
        rags = [fake_rag(), fake_rag()]
        many = AsyncMock(return_value={r.id: DocumentCounts() for r in rags})
        with patch("app.api.rags.repo.list_rags", AsyncMock(return_value=rags)), \
                patch("app.api.rags.repo.counts_for_many", many), \
                patch("app.api.rags.repo.counts_for", AsyncMock()) as per_one:
            self.client.get("/v1/platform/rags", headers=self.head())
        many.assert_awaited_once()
        per_one.assert_not_awaited()


class IconTests(ApiTestCase):
    def test_svg_отклоняется(self):
        """Иконка отдаётся с того же origin, что и фронт."""
        with patch("app.api.rags.repo.get_rag", AsyncMock(return_value=fake_rag())):
            response = self.client.put(
                f"/v1/platform/rags/{uuid.uuid4()}/icon", headers=self.head(),
                files={"file": ("i.svg", self.load_bytes(
                    "icons", "evil.svg"), "image/svg+xml")},
            )
        self.assertEqual(response.status_code, 415)

    def test_подделка_под_png_отклоняется(self):
        """Валидация разбором, а не по первым байтам."""
        with patch("app.api.rags.repo.get_rag", AsyncMock(return_value=fake_rag())), \
                patch("app.api.rags.ObjectStorage", MagicMock()):
            response = self.client.put(
                f"/v1/platform/rags/{uuid.uuid4()}/icon", headers=self.head(),
                files={"file": ("i.png",
                                self.load_bytes("icons", "not_an_image.png"), "image/png")},
            )
        self.assertEqual(response.status_code, 415)

    def test_валидный_png_переупаковывается(self):
        storage = MagicMock()
        counts = AsyncMock(return_value=DocumentCounts())
        with patch("app.api.rags.repo.get_rag", AsyncMock(return_value=fake_rag())), \
                patch("app.api.rags.repo.counts_for", counts), \
                patch("app.api.rags.ObjectStorage", MagicMock(return_value=storage)):
            response = self.client.put(
                f"/v1/platform/rags/{uuid.uuid4()}/icon", headers=self.head(),
                files={"file": ("i.png", self.load_bytes(
                    "icons", "large.png"), "image/png")},
            )
        self.assertEqual(response.status_code, 200)
        _, _, data, content_type = storage.put_bytes.call_args[0]
        self.assertEqual(content_type, "image/png")
        self.assertTrue(data.startswith(b"\x89PNG"))
        self.assertLess(len(data), len(self.load_bytes("icons", "large.png")))


class ArchiveUploadTests(ApiTestCase):
    def test_не_zip_отклоняется(self):
        with patch("app.api.rags.repo.get_rag", AsyncMock(return_value=fake_rag())):
            response = self.client.post(
                f"/v1/platform/rags/{uuid.uuid4()}/imports/archive", headers=self.head(),
                files={"file": ("corpus.tar.gz", b"x", "application/gzip")},
            )
        self.assertEqual(response.status_code, 415)


class AdminRoutingTests(ApiTestCase):
    def test_админские_пути_не_под_v1(self):
        """Catch-all мастера форвардит любой путь под v1/ без разбора —
        /v1/admin/... уехал бы наружу автоматически."""
        from app.api import admin

        for route in admin.router.routes:
            self.assertFalse(route.path.startswith("/v1"), route.path)
            self.assertTrue(route.path.startswith("/admin"), route.path)


class PortSeparationTests(ApiTestCase):
    """Разведение портов — единственный барьер после удаления API-ключа,
    поэтому состав ручек на каждом порту проверяется явно."""

    @staticmethod
    def _paths(application) -> set[str]:
        return set(application.openapi()["paths"])

    def test_служебных_ручек_нет_на_платформенном_порту(self):
        from app.main import app

        for prefix in ("/embed", "/v1/internal", "/admin"):
            self.assertFalse(
                any(p.startswith(prefix) for p in self._paths(app)),
                f"{prefix} доступен на платформенном порту",
            )

    def test_платформенных_ручек_нет_на_внутреннем_порту(self):
        """Обратное тоже важно: /v1/platform скоупится по X-User-Id,
        и открывать его там, где заголовок никто не проставляет, незачем."""
        from app.main import internal_app

        self.assertFalse(
            any(p.startswith("/v1/platform")
                for p in self._paths(internal_app))
        )

    def test_служебные_ручки_есть_на_внутреннем_порту(self):
        from app.main import internal_app

        paths = self._paths(internal_app)
        self.assertIn("/embed", paths)
        self.assertIn("/v1/internal/rags/{rag_id}", paths)

    def test_health_есть_на_обоих(self):
        """Проверять живость должны и мастер, и оркестратор контейнеров."""
        from app.main import app, internal_app

        self.assertIn("/health", self._paths(app))
        self.assertIn("/health", self._paths(internal_app))

    def test_единый_формат_ошибок_на_обоих_портах(self):
        from app.main import app, internal_app

        for application in (app, internal_app):
            with self.subTest(app=application.title):
                self.assertIn(Exception, application.exception_handlers)

    def test_self_url_указывает_на_внутренний_порт(self):
        """Воркер ходит за векторами именно туда."""
        from app.config import settings

        self.assertIn(str(settings.internal_port), settings.self_url)
