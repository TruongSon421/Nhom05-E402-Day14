# main.py
import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from engine.retrieval_eval import RetrievalEvaluator
from engine.regression_gate import RegressionGate
from agent.main_agent import AgentV1, AgentV2, _ChromaStore


class ExpertEvaluator:
    """
    Stub cho RAGAS evaluator.
    Nguoi 2 (Retrieval Engineer) se thay the bang implementation that.
    """
    def __init__(self):
        self.retrieval_eval = RetrievalEvaluator()

    async def score(self, case, resp):
        expected = case.get("ground_truth_ids", [])
        retrieved = resp.get("retrieved_ids", [])
        hit_rate = self.retrieval_eval.calculate_hit_rate(expected, retrieved)
        mrr = self.retrieval_eval.calculate_mrr(expected, retrieved)
        return {
            "faithfulness": 0.0,
            "relevancy": 0.0,
            "retrieval": {"hit_rate": hit_rate, "mrr": mrr}
        }


class MultiModelJudge:
    """
    Stub cho Multi-Judge.
    Nguoi 3 (AI Judge) se thay the bang implementation that.
    """
    async def evaluate_multi_judge(self, q, a, gt):
        return {
            "final_score": 4.0,
            "agreement_rate": 0.8,
            "individual_scores": {"gpt-4o": 4, "claude-3-5": 4},
            "reasoning": "Stub - Nguoi 3 se implement."
        }


async def run_benchmark_with_results(agent, agent_version: str, dataset: list):
    print(f"\n{'='*50}")
    print(f"  Benchmark: {agent_version}")
    print(f"{'='*50}")

    evaluator = ExpertEvaluator()
    judge = MultiModelJudge()

    runner = BenchmarkRunner(agent, evaluator, judge)
    start = time.perf_counter()
    results = await runner.run_all(dataset)
    total_time = time.perf_counter() - start

    total = len(results)

    avg_score = sum(r["judge"]["final_score"] for r in results) / total
    avg_hit_rate = sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total
    avg_mrr = sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total
    avg_agreement = sum(r["judge"]["agreement_rate"] for r in results) / total
    avg_latency = sum(r["latency"] for r in results) / total

    total_cost = 0
    for r in results:
        resp = r.get("agent_response_full", {})
        if isinstance(resp, dict):
            total_cost += resp.get("metadata", {}).get("cost_usd", 0)
    avg_cost = total_cost / total if total else 0

    metrics = {
        "avg_score": round(avg_score, 3),
        "hit_rate": round(avg_hit_rate, 3),
        "mrr": round(avg_mrr, 3),
        "agreement_rate": round(avg_agreement, 3),
        "avg_cost_usd": round(avg_cost, 6),
        "avg_latency_s": round(avg_latency, 3),
    }

    summary = {
        "metadata": {
            "version": agent_version,
            "total": total,
            "total_time_s": round(total_time, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": metrics,
    }

    print(f"  Score:      {metrics['avg_score']}")
    print(f"  Hit Rate:   {metrics['hit_rate']}")
    print(f"  MRR:        {metrics['mrr']}")
    print(f"  Agreement:  {metrics['agreement_rate']}")
    print(f"  Cost:       ${metrics['avg_cost_usd']}")
    print(f"  Latency:    {metrics['avg_latency_s']}s")
    print(f"  Total Time: {summary['metadata']['total_time_s']}s")

    return results, summary


async def main():
    if not os.path.exists("data/golden_set.jsonl"):
        print("Thieu data/golden_set.jsonl. Hay chay 'python data/synthetic_gen.py' truoc.")
        return

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("File data/golden_set.jsonl rong.")
        return

    print(f"Da load {len(dataset)} test cases.")

    # Shared ChromaDB store
    store = _ChromaStore()

    # V1 Benchmark — simple RAG (top-3, basic prompt)
    agent_v1 = AgentV1(store=store)
    v1_results, v1_summary = await run_benchmark_with_results(agent_v1, "Agent_V1_Base", dataset)

    # V2 Benchmark — enhanced RAG (query rewriting, top-5, detailed prompt)
    agent_v2 = AgentV2(store=store)
    v2_results, v2_summary = await run_benchmark_with_results(agent_v2, "Agent_V2_Optimized", dataset)

    # Regression Analysis
    print(f"\n{'='*50}")
    print("  KET QUA SO SANH (REGRESSION)")
    print(f"{'='*50}")

    gate = RegressionGate(max_cost_increase_pct=50, max_latency_increase_pct=50)
    regression = gate.evaluate(v1_summary["metrics"], v2_summary["metrics"])

    print(f"  Quality Delta:  {'+' if regression['delta_quality'] >= 0 else ''}{regression['delta_quality']}")
    print(f"  Hit Rate Delta: {'+' if regression['delta_hit_rate'] >= 0 else ''}{regression['delta_hit_rate']}")
    print(f"  Cost Delta:     {'+' if regression['delta_cost_usd'] >= 0 else ''}${regression['delta_cost_usd']}")
    print(f"  Latency Delta:  {'+' if regression['delta_latency_s'] >= 0 else ''}{regression['delta_latency_s']}s")

    if regression["approved"]:
        print("\n  >>> QUYET DINH: CHAP NHAN BAN CAP NHAT (APPROVE) <<<")
    else:
        print(f"\n  >>> QUYET DINH: TU CHOI (BLOCK RELEASE) - {', '.join(regression['reasons'])} <<<")

    # Save reports
    os.makedirs("reports", exist_ok=True)

    report_summary = {
        "metadata": v2_summary["metadata"],
        "metrics": v2_summary["metrics"],
        "regression": regression,
        "v1_summary": v1_summary,
    }
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(report_summary, f, ensure_ascii=False, indent=2)

    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "v1_results": v1_results,
            "v2_results": v2_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  Reports saved:")
    print(f"    - reports/summary.json")
    print(f"    - reports/benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
