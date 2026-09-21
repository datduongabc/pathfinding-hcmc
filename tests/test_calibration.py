import json
import os
from datetime import datetime

import networkx as nx
import pytest

from src import calibration

GRAPH_PATH = "data/graph.graphml"


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        # Any exception will do: the code under test only needs it to
        # propagate, not to be a requests-specific type.
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, payload_by_point=None, default_payload=None):
        self.payload_by_point = payload_by_point or {}
        self.default_payload = default_payload
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        point = params["point"]
        payload = self.payload_by_point.get(point, self.default_payload)
        return FakeResponse(payload)


def _flow_payload(current, free_flow, confidence=0.9):
    return {"flowSegmentData": {"currentSpeed": current, "freeFlowSpeed": free_flow, "confidence": confidence}}


def test_fetch_corridor_snapshot_computes_ratio():
    corridors = {"Test St": (10.0, 106.0)}
    session = FakeSession(default_payload=_flow_payload(20, 40))
    snapshot = calibration.fetch_corridor_snapshot("fake-key", corridors, session=session)
    assert snapshot["corridors"]["Test St"]["ratio"] == pytest.approx(0.5)
    assert 0.0 <= snapshot["hour"] < 24.0


def test_snapshot_hour_is_hcmc_local_time_not_the_machines(monkeypatch):
    """The time-of-day model is in HCMC hours (UTC+7). A snapshot stamped in
    the machine's own timezone would be fitted against the wrong multiplier."""
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 21, 5, 30, tzinfo=tz)  # 05:30 in the tz asked for

    monkeypatch.setattr(calibration, "datetime", FixedDatetime)
    session = FakeSession(default_payload=_flow_payload(20, 40))
    snapshot = calibration.fetch_corridor_snapshot("fake-key", {"Test St": (10.0, 106.0)}, session=session)

    assert snapshot["hour"] == pytest.approx(5.5)
    assert snapshot["captured_at"].endswith("+07:00")


def test_fetch_corridor_snapshot_collects_all_corridors_concurrently():
    """Requests run in parallel, so results arrive in any order; every
    corridor must still land in the output."""
    corridors = {
        "A St": (10.0, 106.0),
        "B St": (10.1, 106.1),
        "C St": (10.2, 106.2),
    }
    session = FakeSession(payload_by_point={
        "10.0,106.0": _flow_payload(10, 20),
        "10.1,106.1": _flow_payload(20, 40),
        "10.2,106.2": _flow_payload(30, 60),
    })
    snapshot = calibration.fetch_corridor_snapshot("fake-key", corridors, session=session)
    assert set(snapshot["corridors"]) == {"A St", "B St", "C St"}
    assert all(snapshot["corridors"][name]["ratio"] == pytest.approx(0.5) for name in corridors)


def test_fetch_corridor_snapshot_propagates_one_corridors_http_failure():
    """A single corridor's HTTP failure must still surface as an exception,
    not be silently swallowed by the concurrent fetch."""
    corridors = {"Good St": (10.0, 106.0), "Bad St": (10.1, 106.1)}

    class FailingSession(FakeSession):
        def get(self, url, params=None, timeout=None):
            if params["point"] == "10.1,106.1":
                return FakeResponse({}, status_code=500)
            return super().get(url, params=params, timeout=timeout)

    session = FailingSession(default_payload=_flow_payload(10, 20))
    with pytest.raises(RuntimeError):
        calibration.fetch_corridor_snapshot("fake-key", corridors, session=session)


def test_fetch_corridor_snapshot_skips_unexpected_shape():
    corridors = {"Test St": (10.0, 106.0)}
    session = FakeSession(default_payload={"flowSegmentData": {}})
    snapshot = calibration.fetch_corridor_snapshot("fake-key", corridors, session=session)
    assert "Test St" not in snapshot["corridors"]


def test_calibrate_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("TOMTOM_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        calibration.calibrate(output_path=str(tmp_path / "calib.json"), api_key=None)


def test_calibrate_appends_to_existing_file(tmp_path):
    output_path = tmp_path / "calib.json"
    output_path.write_text(json.dumps({"snapshots": [{"hour": 1.0, "corridors": {}}]}), encoding="utf-8")

    session = FakeSession(default_payload=_flow_payload(20, 40))
    calibration.calibrate(output_path=str(output_path), api_key="fake-key", session=session)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(saved["snapshots"]) == 2


def test_calibrate_recovers_from_malformed_existing_file(tmp_path):
    output_path = tmp_path / "calib.json"
    output_path.write_text("{not valid json", encoding="utf-8")

    session = FakeSession(default_payload=_flow_payload(20, 40))
    calibration.calibrate(output_path=str(output_path), api_key="fake-key", session=session)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(saved["snapshots"]) == 1


def test_calibrate_recovers_from_structurally_invalid_but_syntactically_valid_json(tmp_path):
    """Valid JSON with the wrong shape (e.g. "snapshots" not a list) must be
    handled like unparsable JSON - start a fresh file - rather than crash
    on the append."""
    output_path = tmp_path / "calib.json"
    output_path.write_text(json.dumps({"snapshots": "not-a-list"}), encoding="utf-8")

    session = FakeSession(default_payload=_flow_payload(20, 40))
    calibration.calibrate(output_path=str(output_path), api_key="fake-key", session=session)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(saved["snapshots"], list)
    assert len(saved["snapshots"]) == 1


@pytest.mark.skipif(not os.path.exists(GRAPH_PATH), reason="cached graph not present")
def test_corridors_keys_match_real_edge_names_in_the_committed_graph():
    """A corridor whose name matches no graph edge fails silently: its
    calibration is fetched and stored but never affects any edge cost. Names
    must match exactly, diacritics included (e.g. "Điện Biên Phủ", not
    "Dien Bien Phu")."""
    graph = nx.read_graphml(GRAPH_PATH)
    edge_names = {d["name"] for _, _, d in graph.edges(data=True) if "name" in d}

    unmatched = [corridor for corridor in calibration.CORRIDORS if corridor not in edge_names]
    assert not unmatched, f"corridors with no matching edge name in the graph: {unmatched}"
