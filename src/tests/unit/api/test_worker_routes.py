"""Tests for worker REST routes — GET /v1/workers, /v1/workers/stats, /v1/workers/{worker_id}."""

import time

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.settings import Settings
from api.workers.registry import WorkerRegistry


@pytest.fixture
def client():
    """Create a test client with lifecycle manager (lifespan runs)."""
    settings = Settings(auto_start=False)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def registry(client) -> WorkerRegistry:
    """Return the live WorkerRegistry attached to the test app."""
    return client.app.state.worker_registry


@pytest.mark.unit
class TestListWorkers:
    """Test GET /v1/workers — list all registered workers."""

    def test_list_workers_empty(self, client):
        """GET /v1/workers returns empty list when no workers registered."""
        response = client.get("/v1/workers")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["data"]["workers"] == []
        assert body["data"]["count"] == 0

    def test_list_workers_with_one_worker(self, client, registry):
        """GET /v1/workers returns a single registered worker."""
        registry.register("worker-1", {"cpu_cores": 8, "gpu": True})

        response = client.get("/v1/workers")
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["count"] == 1
        worker = body["data"]["workers"][0]
        assert worker["worker_id"] == "worker-1"
        assert worker["capabilities"] == {"cpu_cores": 8, "gpu": True}
        assert worker["idle"] is True
        assert worker["health_score"] == 1.0
        assert worker["tasks_completed"] == 0
        assert worker["tasks_failed"] == 0
        assert worker["active_task_id"] is None

    def test_list_workers_with_multiple_workers(self, client, registry):
        """GET /v1/workers returns all registered workers."""
        registry.register("worker-a", {"cpu_cores": 4})
        registry.register("worker-b", {"cpu_cores": 16})
        registry.register("worker-c", {"cpu_cores": 2})

        response = client.get("/v1/workers")
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["count"] == 3
        worker_ids = {w["worker_id"] for w in body["data"]["workers"]}
        assert worker_ids == {"worker-a", "worker-b", "worker-c"}

    def test_list_workers_serialization_fields(self, client, registry):
        """GET /v1/workers serializes all expected fields for each worker."""
        registry.register("worker-1", {})
        reg = registry.get("worker-1")
        reg.tasks_completed = 5
        reg.tasks_failed = 2
        reg.active_task_id = "task-42"

        response = client.get("/v1/workers")
        assert response.status_code == 200
        worker = response.json()["data"]["workers"][0]
        expected_keys = {
            "worker_id",
            "capabilities",
            "connected_at",
            "last_heartbeat",
            "tasks_completed",
            "tasks_failed",
            "active_task_id",
            "health_score",
            "idle",
            # METRICS-MON R1.3 / seed-04: enriched heartbeat fields.
            "in_flight_tasks",
            "last_task_completed_at",
            "rss_mb",
        }
        assert set(worker.keys()) == expected_keys
        assert worker["tasks_completed"] == 5
        assert worker["tasks_failed"] == 2
        assert worker["active_task_id"] == "task-42"
        assert worker["idle"] is False
        assert worker["health_score"] == pytest.approx(5 / 7)
        # Default enriched fields for a worker that hasn't sent an R1.3 heartbeat
        assert worker["in_flight_tasks"] == 0
        assert worker["last_task_completed_at"] is None
        assert worker["rss_mb"] is None

    def test_list_workers_surfaces_enriched_heartbeat_fields(self, client, registry):
        """METRICS-MON R1.3: when a worker reports enriched fields, /v1/workers shows them."""
        registry.register("worker-r13", {})
        registry.heartbeat("worker-r13", in_flight_tasks=2, last_task_completed_at=1745816400.0, rss_mb=412.7)
        response = client.get("/v1/workers")
        assert response.status_code == 200
        worker = next(w for w in response.json()["data"]["workers"] if w["worker_id"] == "worker-r13")
        assert worker["in_flight_tasks"] == 2
        assert worker["last_task_completed_at"] == 1745816400.0
        assert worker["rss_mb"] == 412.7

    def test_list_workers_response_envelope(self, client):
        """GET /v1/workers wraps response in standard envelope."""
        response = client.get("/v1/workers")
        body = response.json()
        assert "status" in body
        assert "data" in body
        assert "meta" in body
        assert "timestamp" in body["meta"]
        assert "version" in body["meta"]


@pytest.mark.unit
class TestWorkerStats:
    """Test GET /v1/workers/stats — aggregate worker statistics."""

    def test_stats_empty_registry(self, client):
        """GET /v1/workers/stats returns zeroed stats when no workers registered."""
        response = client.get("/v1/workers/stats")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        data = body["data"]
        assert data["total"] == 0
        assert data["idle"] == 0
        assert data["busy"] == 0
        assert data["stale"] == 0
        assert data["total_tasks_completed"] == 0
        assert data["total_tasks_failed"] == 0
        assert data["average_health_score"] == 0.0
        assert "timestamp" in data

    def test_stats_all_idle(self, client, registry):
        """GET /v1/workers/stats counts idle workers correctly."""
        registry.register("w1", {})
        registry.register("w2", {})

        response = client.get("/v1/workers/stats")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["total"] == 2
        assert data["idle"] == 2
        assert data["busy"] == 0
        assert data["stale"] == 0

    def test_stats_with_busy_worker(self, client, registry):
        """GET /v1/workers/stats counts busy workers correctly."""
        registry.register("w1", {})
        registry.register("w2", {})
        registry.assign_task("w1", "task-1")

        response = client.get("/v1/workers/stats")
        data = response.json()["data"]
        assert data["total"] == 2
        assert data["idle"] == 1
        assert data["busy"] == 1

    def test_stats_with_stale_worker(self, client):
        """GET /v1/workers/stats counts stale workers correctly."""
        # Create app with very short heartbeat timeout to make workers go stale
        settings = Settings(auto_start=False, remote_workers_heartbeat_timeout=0.01)
        app = create_app(settings)
        with TestClient(app) as c:
            reg = c.app.state.worker_registry
            reg.register("w1", {})
            time.sleep(0.02)  # Let heartbeat expire

            response = c.get("/v1/workers/stats")
            data = response.json()["data"]
            assert data["total"] == 1
            assert data["stale"] == 1
            # Stale workers are not idle (they failed is_alive check)
            assert data["idle"] == 0

    def test_stats_aggregate_tasks(self, client, registry):
        """GET /v1/workers/stats aggregates tasks across all workers."""
        registry.register("w1", {})
        registry.register("w2", {})
        w1 = registry.get("w1")
        w2 = registry.get("w2")
        w1.tasks_completed = 10
        w1.tasks_failed = 2
        w2.tasks_completed = 5
        w2.tasks_failed = 1

        response = client.get("/v1/workers/stats")
        data = response.json()["data"]
        assert data["total_tasks_completed"] == 15
        assert data["total_tasks_failed"] == 3

    def test_stats_average_health_score(self, client, registry):
        """GET /v1/workers/stats computes average health correctly."""
        registry.register("w1", {})
        registry.register("w2", {})
        # w1: 8/10 = 0.8 health
        registry.get("w1").tasks_completed = 8
        registry.get("w1").tasks_failed = 2
        # w2: 6/10 = 0.6 health
        registry.get("w2").tasks_completed = 6
        registry.get("w2").tasks_failed = 4

        response = client.get("/v1/workers/stats")
        data = response.json()["data"]
        # Average: (0.8 + 0.6) / 2 = 0.7
        assert data["average_health_score"] == pytest.approx(0.7, abs=1e-4)

    def test_stats_response_envelope(self, client):
        """GET /v1/workers/stats wraps response in standard envelope."""
        response = client.get("/v1/workers/stats")
        body = response.json()
        assert body["status"] == "success"
        assert "meta" in body
        assert "timestamp" in body["meta"]


@pytest.mark.unit
class TestGetWorker:
    """Test GET /v1/workers/{worker_id} — get specific worker details."""

    def test_get_worker_found(self, client, registry):
        """GET /v1/workers/{id} returns worker details when found."""
        registry.register("worker-42", {"cpu_cores": 16, "memory_gb": 64})

        response = client.get("/v1/workers/worker-42")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        worker = body["data"]
        assert worker["worker_id"] == "worker-42"
        assert worker["capabilities"] == {"cpu_cores": 16, "memory_gb": 64}
        assert worker["idle"] is True
        assert worker["health_score"] == 1.0

    def test_get_worker_not_found(self, client):
        """GET /v1/workers/{id} returns 404 for unknown worker."""
        response = client.get("/v1/workers/nonexistent-worker")
        assert response.status_code == 404
        body = response.json()
        assert "nonexistent-worker" in body["detail"]

    def test_get_worker_with_task_assigned(self, client, registry):
        """GET /v1/workers/{id} reflects assigned task state."""
        registry.register("worker-1", {})
        registry.assign_task("worker-1", "task-99")

        response = client.get("/v1/workers/worker-1")
        assert response.status_code == 200
        worker = response.json()["data"]
        assert worker["active_task_id"] == "task-99"
        assert worker["idle"] is False

    def test_get_worker_with_task_history(self, client, registry):
        """GET /v1/workers/{id} shows completed and failed task counts."""
        registry.register("worker-1", {})
        reg = registry.get("worker-1")
        reg.tasks_completed = 15
        reg.tasks_failed = 3

        response = client.get("/v1/workers/worker-1")
        worker = response.json()["data"]
        assert worker["tasks_completed"] == 15
        assert worker["tasks_failed"] == 3
        assert worker["health_score"] == pytest.approx(15 / 18)

    def test_get_worker_timestamps(self, client, registry):
        """GET /v1/workers/{id} includes connected_at and last_heartbeat timestamps."""
        registry.register("worker-1", {})

        response = client.get("/v1/workers/worker-1")
        worker = response.json()["data"]
        assert isinstance(worker["connected_at"], float)
        assert isinstance(worker["last_heartbeat"], float)
        assert worker["connected_at"] > 0
        assert worker["last_heartbeat"] > 0

    def test_get_worker_response_envelope(self, client, registry):
        """GET /v1/workers/{id} wraps response in standard envelope."""
        registry.register("worker-1", {})

        response = client.get("/v1/workers/worker-1")
        body = response.json()
        assert body["status"] == "success"
        assert "data" in body
        assert "meta" in body


@pytest.mark.unit
class TestWorkerRegistryNotInitialized:
    """Test routes when worker_registry is not set on app.state."""

    def test_list_workers_no_registry(self):
        """GET /v1/workers returns 503 when registry not initialized."""
        settings = Settings(auto_start=False)
        app = create_app(settings)
        with TestClient(app, raise_server_exceptions=False) as c:
            # Remove the registry to simulate uninitialized state
            del c.app.state.worker_registry
            response = c.get("/v1/workers")
            assert response.status_code == 503

    def test_get_stats_no_registry(self):
        """GET /v1/workers/stats returns 503 when registry not initialized."""
        settings = Settings(auto_start=False)
        app = create_app(settings)
        with TestClient(app, raise_server_exceptions=False) as c:
            del c.app.state.worker_registry
            response = c.get("/v1/workers/stats")
            assert response.status_code == 503

    def test_get_worker_no_registry(self):
        """GET /v1/workers/{id} returns 503 when registry not initialized."""
        settings = Settings(auto_start=False)
        app = create_app(settings)
        with TestClient(app, raise_server_exceptions=False) as c:
            del c.app.state.worker_registry
            response = c.get("/v1/workers/worker-1")
            assert response.status_code == 503
