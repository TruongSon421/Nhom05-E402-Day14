from typing import Dict


class RegressionGate:
    """
    Release Gate tu dong dua tren 3 chieu: Quality, Cost, Performance.
    Approve neu: quality khong giam VA cost/latency trong nguong cho phep.
    """

    def __init__(self, max_cost_increase_pct: float = 50, max_latency_increase_pct: float = 50):
        self.max_cost_increase_pct = max_cost_increase_pct
        self.max_latency_increase_pct = max_latency_increase_pct

    def evaluate(self, v1_metrics: Dict, v2_metrics: Dict) -> Dict:
        delta_quality = v2_metrics["avg_score"] - v1_metrics["avg_score"]
        delta_hit_rate = v2_metrics["hit_rate"] - v1_metrics["hit_rate"]
        delta_cost = v2_metrics["avg_cost_usd"] - v1_metrics["avg_cost_usd"]
        delta_latency = v2_metrics["avg_latency_s"] - v1_metrics["avg_latency_s"]

        reasons = []

        quality_ok = delta_quality >= 0
        if not quality_ok:
            reasons.append("Quality giam")

        max_cost_delta = v1_metrics["avg_cost_usd"] * (self.max_cost_increase_pct / 100)
        cost_ok = delta_cost <= max_cost_delta
        if not cost_ok:
            reasons.append(f"Cost tang qua {int(self.max_cost_increase_pct)}%")

        max_latency_delta = v1_metrics["avg_latency_s"] * (self.max_latency_increase_pct / 100)
        latency_ok = delta_latency <= max_latency_delta
        if not latency_ok:
            reasons.append(f"Latency tang qua {int(self.max_latency_increase_pct)}%")

        approved = quality_ok and cost_ok and latency_ok

        return {
            "approved": approved,
            "reasons": reasons,
            "delta_quality": round(delta_quality, 3),
            "delta_hit_rate": round(delta_hit_rate, 3),
            "delta_cost_usd": round(delta_cost, 6),
            "delta_latency_s": round(delta_latency, 3),
        }
