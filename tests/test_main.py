import json
import os
import pytest


def test_summary_json_format():
    from engine.regression_gate import RegressionGate

    v1_metrics = {"avg_score": 3.5, "hit_rate": 0.7, "agreement_rate": 0.8, "avg_cost_usd": 0.001, "avg_latency_s": 1.0}
    v2_metrics = {"avg_score": 4.0, "hit_rate": 0.85, "agreement_rate": 0.9, "avg_cost_usd": 0.0013, "avg_latency_s": 1.2}

    gate = RegressionGate()
    regression = gate.evaluate(v1_metrics, v2_metrics)

    summary = {
        "metadata": {
            "version": "Agent_V2_Optimized",
            "total": 50,
            "timestamp": "2026-04-21 12:00:00",
        },
        "metrics": {
            "avg_score": v2_metrics["avg_score"],
            "hit_rate": v2_metrics["hit_rate"],
            "agreement_rate": v2_metrics["agreement_rate"],
            "avg_cost_usd": v2_metrics["avg_cost_usd"],
            "avg_latency_s": v2_metrics["avg_latency_s"],
        },
        "regression": regression,
        "v1_metrics": v1_metrics,
    }

    assert "metadata" in summary
    assert "metrics" in summary
    assert "version" in summary["metadata"]
    assert "total" in summary["metadata"]
    assert "hit_rate" in summary["metrics"]
    assert "agreement_rate" in summary["metrics"]
    assert "avg_score" in summary["metrics"]


def test_summary_json_regression_fields():
    from engine.regression_gate import RegressionGate

    v1 = {"avg_score": 3.5, "hit_rate": 0.7, "avg_cost_usd": 0.001, "avg_latency_s": 1.0}
    v2 = {"avg_score": 4.0, "hit_rate": 0.85, "avg_cost_usd": 0.0013, "avg_latency_s": 1.2}

    gate = RegressionGate()
    regression = gate.evaluate(v1, v2)

    assert "approved" in regression
    assert "delta_quality" in regression
    assert "delta_hit_rate" in regression
    assert "delta_cost_usd" in regression
    assert "delta_latency_s" in regression
