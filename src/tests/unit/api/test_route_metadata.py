"""Every route declares an ``operation_id``, and every envelope route a ``response_model``.

Two gaps over ONE set of 47 decorators, pinned together because they are one mechanical pass
(splitting them rewrites every decorator twice):

* ``operation_id`` was absent on all 47. FastAPI's default is ``<handler>_<path>_<method>``, so a
  generated SDK's method names were coupled to the handler name, the router prefix AND the version
  prefix -- renaming a handler or bumping ``/v1`` silently renamed every client method. Each route
  now pins ``operation_id`` to its handler name alone.
* ``response_model`` was absent on 46 of 47 (``APD-CASCOR-003``). It is now declared on the 44
  routes that build their body with ``success_response()``.

**Why declaring the envelope is wire-neutral, and what that depends on.** ``success_response()``
returns ``ResponseEnvelope(...).model_dump()``, so an enveloped body has *already* round-tripped
through the exact model ``response_model=`` re-applies -- the second pass is idempotent by
construction, not merely "``data`` is ``Any``". That guarantee holds only while every enveloped
route actually goes through the helper, so ``TestEnvelopeUniformity`` pins that property directly:
a future handler that hand-builds a dict would be filtered against the envelope and silently lose
every field outside ``status``/``data``/``meta``, with nothing failing.

**The three health routes are deliberately excluded from the response_model half**, and that
exclusion is pinned so it reads as a decision rather than an oversight. They do not use the
envelope: ``readiness_probe`` already declares ``ReadinessResponse``, while ``health_check`` and
``liveness_probe`` return bare dicts on the documented cross-service API-02 ``{status, version,
service}`` base shared with juniper-data and juniper-canopy. Declaring a model on those two is
**not** wire-neutral -- measured: an optional field absent from the 200 body reappears as an
explicit ``"error": null`` once a model is declared, because ``response_model_exclude_none``
defaults to ``False``. Giving them their own models is a cross-repo wire decision, not this defect.

Routes are read from each ``APIRouter`` rather than from ``app.routes``: on FastAPI >= 0.137 the
app holds ``_IncludedRouter`` dataclasses that carry no ``path`` or ``methods`` attribute at all,
so an ``app.routes`` walk silently sees nothing.
"""

import ast
from pathlib import Path

import pytest
from fastapi.routing import APIRoute

from api.models.common import ResponseEnvelope
from api.routes import admin, dataset, decision_boundary, health, history, metrics, network, snapshots, training, workers

ROUTERS = {
    "admin": admin.router,
    "dataset": dataset.router,
    "decision_boundary": decision_boundary.router,
    "health": health.router,
    "history": history.router,
    "metrics": metrics.router,
    "network": network.router,
    "snapshots": snapshots.router,
    "training": training.router,
    "workers": workers.router,
}

TOTAL_ROUTES = 47
ENVELOPE_ROUTES = 44

# The handlers that legitimately sit outside the envelope. Spelled out so that adding a
# fourth one is a deliberate edit to this list, not a silent exemption.
NON_ENVELOPE_HANDLERS = {"health_check", "liveness_probe", "readiness_probe"}
# Of those, the two that must NOT gain a response_model without a cross-service wire decision.
BARE_DICT_HANDLERS = {"health_check", "liveness_probe"}

ROUTES_DIR = Path(admin.__file__).parent

# The PUBLISHED contract. These are the method names a generated SDK exposes, so they are a
# wire-level surface: changing one is a breaking change for every consumer and must fail a test
# rather than ride along with a refactor. They happen to equal today's handler names, but they are
# NOT pinned to them -- see TestOperationIds.
PUBLISHED_OPERATION_IDS = {
    "add_hidden_unit",
    "cancel_dataset_stage",
    "cancel_swap_dataset_live",
    "clear_metrics",
    "create_network",
    "delete_hidden_unit",
    "delete_network",
    "get_dataset",
    "get_dataset_data",
    "get_dataset_swap_events",
    "get_decision_boundary",
    "get_experimental_functions",
    "get_metrics",
    "get_metrics_history",
    "get_network",
    "get_params",
    "get_pending_dataset",
    "get_snapshot",
    "get_snapshot_dataset_swaps",
    "get_stats",
    "get_status",
    "get_topology",
    "get_transport_stats",
    "get_worker",
    "get_worker_stats",
    "health_check",
    "list_snapshots",
    "list_workers",
    "liveness_probe",
    "patch_weights",
    "pause_training",
    "readiness_probe",
    "replay_control_endpoint",
    "reset_training",
    "restore_snapshot",
    "resume_snapshot",
    "resume_training",
    "retrain_from_snapshot",
    "save_snapshot",
    "set_experimental_functions",
    "stage_dataset",
    "start_replay_endpoint",
    "start_training",
    "stop_training",
    "swap_dataset_live",
    "undo_clear_metrics",
    "update_training_params",
}


def _api_routes():
    """Every ``APIRoute`` across every router, with the router name that owns it."""
    for name, router in ROUTERS.items():
        for route in router.routes:
            if isinstance(route, APIRoute):
                yield name, route


def _declared_decorator_kwargs() -> dict[str, set[str]]:
    """Keyword names each handler's ``@router.<method>(...)`` decorator DECLARES, read from source.

    Read from source rather than from ``APIRoute`` because FastAPI *infers* ``response_model`` from
    the handler's return annotation -- every ``-> dict`` handler reports a non-``None``
    ``route.response_model`` whether or not the decorator says anything. Only the source can
    distinguish "declared" from "inferred", and this defect is about what is declared.
    """
    declared: dict[str, set[str]] = {}
    for path in sorted(ROUTES_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                if not isinstance(dec, ast.Call):
                    continue
                func = dec.func
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "router":
                    declared.setdefault(node.name, set()).update(kw.arg for kw in dec.keywords if kw.arg)
    return declared


def _handlers_using_success_response() -> set[str]:
    """Handler names whose body is built by ``success_response()``, read from source."""
    using = set()
    for path in sorted(ROUTES_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "success_response" for call in ast.walk(node)):
                using.add(node.name)
    return using


@pytest.mark.unit
class TestRouteCensus:
    """The pins below are only meaningful if they see every route."""

    def test_expected_number_of_routes(self):
        assert len(list(_api_routes())) == TOTAL_ROUTES


@pytest.mark.unit
class TestOperationIds:
    """A generated SDK's method names must not move when a path or prefix does."""

    def test_every_route_declares_an_operation_id(self):
        missing = [f"{name}:{route.path}" for name, route in _api_routes() if not route.operation_id]
        assert not missing, f"routes with FastAPI's default operation_id: {sorted(missing)}"

    def test_published_ids_are_frozen(self):
        """The published set is the contract; a rename must be a deliberate edit to this list.

        Deliberately NOT asserted against ``endpoint.__name__``. The whole point of an explicit
        ``operation_id`` is that it DECOUPLES the published name from the handler -- so renaming a
        handler must change nothing here. Asserting they match would re-couple them and reinstate
        the defect in a new form. The sibling ``APD-DATA-023`` close pins the same property, and
        its mutation matrix carries the handler rename as an expected-SURVIVAL row.
        """
        assert {route.operation_id for _, route in _api_routes()} == PUBLISHED_OPERATION_IDS

    def test_operation_ids_are_unique(self):
        ids = [route.operation_id for _, route in _api_routes()]
        duplicates = {value for value in ids if ids.count(value) > 1}
        assert not duplicates, f"duplicate operation_id: {sorted(duplicates)}"


@pytest.mark.unit
class TestResponseModels:
    """APD-CASCOR-003 -- and the deliberate exclusions."""

    def test_every_envelope_route_declares_the_envelope(self):
        missing = [f"{name}:{route.path}" for name, route in _api_routes() if route.endpoint.__name__ not in NON_ENVELOPE_HANDLERS and route.response_model is not ResponseEnvelope]
        assert not missing, f"envelope routes with no response_model=ResponseEnvelope: {sorted(missing)}"

    def test_expected_number_of_envelope_routes(self):
        declared = [route for _, route in _api_routes() if route.response_model is ResponseEnvelope]
        assert len(declared) == ENVELOPE_ROUTES

    def test_bare_dict_health_routes_declare_no_response_model(self):
        """Deliberate: declaring one adds an explicit null field to the 200 body.

        Checked against the DECORATOR SOURCE, not ``route.response_model`` -- FastAPI infers the
        latter from the ``-> dict`` return annotation, so it is never ``None`` and an
        identity check against it would pass no matter what the decorator said.
        """
        declared = _declared_decorator_kwargs()
        wrong = sorted(name for name in BARE_DICT_HANDLERS if "response_model" in declared.get(name, set()))
        assert not wrong, f"a response_model here changes the cross-service health wire: {wrong}"


@pytest.mark.unit
class TestEnvelopeUniformity:
    """The property the response_model declaration's wire-safety rests on."""

    def test_every_enveloped_route_builds_its_body_with_success_response(self):
        """A handler that hand-builds a dict would be filtered against the envelope and lose fields."""
        using = _handlers_using_success_response()
        assert using, "no handler found using success_response -- the pin would be vacuous"
        offenders = [route.endpoint.__name__ for _, route in _api_routes() if route.response_model is ResponseEnvelope and route.endpoint.__name__ not in using]
        assert not offenders, f"declared as ResponseEnvelope but does not go through success_response(): {sorted(offenders)}"
