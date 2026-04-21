from engine.regression_gate import RegressionGate


def test_approve_when_quality_improves_within_budget():
    gate = RegressionGate(max_cost_increase_pct=50, max_latency_increase_pct=50)
    v1 = {"avg_score": 3.5, "hit_rate": 0.7, "avg_cost_usd": 0.001, "avg_latency_s": 1.0}
    v2 = {"avg_score": 4.0, "hit_rate": 0.85, "avg_cost_usd": 0.0013, "avg_latency_s": 1.2}
    result = gate.evaluate(v1, v2)
    assert result["approved"] is True
    assert result["delta_quality"] > 0


def test_block_when_quality_decreases():
    gate = RegressionGate()
    v1 = {"avg_score": 4.0, "hit_rate": 0.8, "avg_cost_usd": 0.001, "avg_latency_s": 1.0}
    v2 = {"avg_score": 3.0, "hit_rate": 0.6, "avg_cost_usd": 0.001, "avg_latency_s": 1.0}
    result = gate.evaluate(v1, v2)
    assert result["approved"] is False
    assert "Quality giam" in result["reasons"]


def test_block_when_cost_exceeds_threshold():
    gate = RegressionGate(max_cost_increase_pct=50)
    v1 = {"avg_score": 3.5, "hit_rate": 0.7, "avg_cost_usd": 0.001, "avg_latency_s": 1.0}
    v2 = {"avg_score": 4.0, "hit_rate": 0.85, "avg_cost_usd": 0.002, "avg_latency_s": 1.0}
    result = gate.evaluate(v1, v2)
    assert result["approved"] is False
    assert "Cost tang qua 50%" in result["reasons"]


def test_block_when_latency_exceeds_threshold():
    gate = RegressionGate(max_latency_increase_pct=50)
    v1 = {"avg_score": 3.5, "hit_rate": 0.7, "avg_cost_usd": 0.001, "avg_latency_s": 1.0}
    v2 = {"avg_score": 4.0, "hit_rate": 0.85, "avg_cost_usd": 0.001, "avg_latency_s": 2.0}
    result = gate.evaluate(v1, v2)
    assert result["approved"] is False
    assert "Latency tang qua 50%" in result["reasons"]


def test_multiple_block_reasons():
    gate = RegressionGate(max_cost_increase_pct=50, max_latency_increase_pct=50)
    v1 = {"avg_score": 4.0, "hit_rate": 0.8, "avg_cost_usd": 0.001, "avg_latency_s": 1.0}
    v2 = {"avg_score": 3.0, "hit_rate": 0.6, "avg_cost_usd": 0.003, "avg_latency_s": 3.0}
    result = gate.evaluate(v1, v2)
    assert result["approved"] is False
    assert len(result["reasons"]) == 3


def test_deltas_are_correct():
    gate = RegressionGate()
    v1 = {"avg_score": 3.0, "hit_rate": 0.6, "avg_cost_usd": 0.001, "avg_latency_s": 1.0}
    v2 = {"avg_score": 4.0, "hit_rate": 0.8, "avg_cost_usd": 0.0015, "avg_latency_s": 1.5}
    result = gate.evaluate(v1, v2)
    assert result["delta_quality"] == 1.0
    assert result["delta_hit_rate"] == 0.2
    assert abs(result["delta_cost_usd"] - 0.0005) < 0.0001
    assert result["delta_latency_s"] == 0.5
