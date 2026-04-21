import asyncio
import json
import os
import time

from engine.runner import BenchmarkRunner
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from agent.main_agent import MainAgent


class ExpertEvaluator:
    """Wrapper RetrievalEvaluator thật → interface evaluator.score() cho BenchmarkRunner."""

    def __init__(self):
        self._retrieval = RetrievalEvaluator()

    async def score(self, case: dict, resp: dict) -> dict:
        question         = case.get("question", "")
        ground_truth_ids = case.get("ground_truth_ids", [])

        retrieval_result = await self._retrieval.evaluate_batch(
            [{"question": question, "ground_truth_ids": ground_truth_ids}]
        )
        per = retrieval_result["per_case"][0] if retrieval_result["per_case"] else {}

        return {
            "faithfulness": 1.0,   # placeholder — cần RAGAS nếu muốn đo thật
            "relevancy":    1.0,   # placeholder
            "retrieval": {
                "hit_rate": per.get("hit_rate") or 0.0,
                "mrr":      per.get("mrr")      or 0.0,
            },
        }


async def run_benchmark_with_results(agent_version: str, agent=None):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    if agent is None:
        agent = MainAgent()

    evaluator = ExpertEvaluator()
    judge     = LLMJudge()

    runner  = BenchmarkRunner(agent, evaluator, judge)
    results = await runner.run_all(dataset)

    total = len(results)
    summary = {
        "metadata": {
            "version":   agent_version,
            "total":     total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score":      sum(r["judge"]["final_score"]           for r in results) / total,
            "hit_rate":       sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"]        for r in results) / total,
            "cohen_kappa":    judge.compute_cohen_kappa(results),
            "cost":           judge.compute_cost(results),
        },
    }
    return results, summary


async def run_benchmark(version: str):
    _, summary = await run_benchmark_with_results(version)
    return summary


async def main():
    # Dùng chung 1 agent instance để tránh khởi tạo LangGraph 2 lần
    shared_agent = MainAgent()

    v1_results, v1_summary = await run_benchmark_with_results("Agent_V1_Base", agent=shared_agent)

    if not v1_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    m = v1_summary["metrics"]
    print("\n📊 --- KẾT QUẢ BENCHMARK ---")
    print(f"  Score trung bình : {m['avg_score']:.2f} / 5")
    print(f"  Hit Rate         : {m['hit_rate']:.3f}")
    print(f"  Agreement Rate   : {m['agreement_rate']:.3f}")
    print(f"  Cohen's Kappa    : {m['cohen_kappa']:.4f}")
    print(f"  Tổng chi phí     : ${m['cost']['total_cost_usd']:.4f} USD")

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v1_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v1_results, f, ensure_ascii=False, indent=2)

    print("\n✅ Đã lưu báo cáo vào reports/summary.json và reports/benchmark_results.json")

    if m["avg_score"] >= 3.5:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")


if __name__ == "__main__":
    asyncio.run(main())
