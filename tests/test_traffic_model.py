import json

import pytest

from src import traffic_model


def test_effective_speed_never_exceeds_free_flow():
    for highway in ["motorway", "primary", "residential", "living_street", "unknown_class"]:
        v_free = traffic_model.free_flow_speed_kmh(highway)
        for hour in [0.0, 3.5, 7.0, 8.5, 9.5, 13.0, 17.5, 20.0, 23.9]:
            for bias in [{"Test St": 0.6}, {}]:
                v_eff = traffic_model.effective_speed_kmh(highway, "Test St", hour, bias)
                assert v_eff <= v_free + 1e-9


def test_time_of_day_multiplier_rush_lower_than_night():
    assert traffic_model.time_of_day_multiplier(8.0) < traffic_model.time_of_day_multiplier(2.0)
    assert traffic_model.time_of_day_multiplier(17.5) < traffic_model.time_of_day_multiplier(2.0)


def test_time_of_day_multiplier_night_near_free_flow():
    assert traffic_model.time_of_day_multiplier(2.0) == pytest.approx(0.95)
    assert traffic_model.time_of_day_multiplier(23.5) == pytest.approx(0.95)


def test_time_of_day_multiplier_wraps_at_24h():
    assert traffic_model.time_of_day_multiplier(24.0) == traffic_model.time_of_day_multiplier(0.0)
    assert traffic_model.time_of_day_multiplier(25.0) == traffic_model.time_of_day_multiplier(1.0)


def test_max_time_multiplier_over_window_is_at_least_the_point_value():
    for hour in [0.0, 8.0, 9.0, 17.0, 21.0]:
        point_value = traffic_model.time_of_day_multiplier(hour)
        window_value = traffic_model.max_time_multiplier_over_window(hour)
        assert window_value >= point_value - 1e-9


def test_max_time_multiplier_over_window_captures_upcoming_recovery():
    # Departing at 9h (rush trough, m_time=0.55), a 3h window reaches past
    # the 10h anchor where m_time rises to 0.75 - so the window bound must
    # exceed the point value at 9h.
    assert traffic_model.max_time_multiplier_over_window(9.0) > traffic_model.time_of_day_multiplier(9.0) + 1e-6


def test_cost_positive_and_finite():
    for length_m in [10.0, 500.0, 5000.0]:
        for hour in [0.0, 8.0, 17.5]:
            t = traffic_model.edge_travel_time_hours(length_m, "residential", None, hour, {})
            assert t > 0.0
            assert t < float("inf")


def test_corridor_multiplier_defaults_to_one_for_unknown_street():
    assert traffic_model.corridor_multiplier("Unknown St", {"Dien Bien Phu": 0.7}) == 1.0
    assert traffic_model.corridor_multiplier(None, {"Dien Bien Phu": 0.7}) == 1.0


def test_load_corridor_bias_missing_file_returns_empty(tmp_path):
    bias = traffic_model.load_corridor_bias(str(tmp_path / "does_not_exist.json"))
    assert bias == {}


def test_load_corridor_bias_fits_from_snapshots(tmp_path):
    calibration_file = tmp_path / "calib.json"
    calibration_file.write_text(json.dumps({
        "snapshots": [
            {"hour": 8.0, "corridors": {"Dien Bien Phu": {"ratio": 0.4675}}},   # 0.55 * 0.85
            {"hour": 13.0, "corridors": {"Dien Bien Phu": {"ratio": 0.6375}}},  # 0.75 * 0.85
        ]
    }), encoding="utf-8")

    bias = traffic_model.load_corridor_bias(str(calibration_file))
    assert bias["Dien Bien Phu"] == pytest.approx(0.85, abs=1e-6)


def test_load_corridor_bias_clamped_to_valid_range(tmp_path):
    calibration_file = tmp_path / "calib.json"
    calibration_file.write_text(json.dumps({
        "snapshots": [{"hour": 8.0, "corridors": {"Overload St": {"ratio": 5.0}}}]
    }), encoding="utf-8")
    bias = traffic_model.load_corridor_bias(str(calibration_file))
    assert bias["Overload St"] == 1.0


def test_min_speed_clamp():
    v_eff = traffic_model.effective_speed_kmh("living_street", "Slow St", 17.0, {"Slow St": 0.5})
    assert v_eff == traffic_model.MIN_SPEED_KMH


def test_load_corridor_bias_handles_malformed_structure(tmp_path):
    # Syntactically valid JSON but structurally malformed (snapshots is not a list)
    calibration_file = tmp_path / "bad_structure.json"
    calibration_file.write_text(json.dumps({
        "snapshots": "not-a-list"
    }), encoding="utf-8")
    bias = traffic_model.load_corridor_bias(str(calibration_file))
    assert bias == {}
