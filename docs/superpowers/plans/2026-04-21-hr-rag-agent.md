# Người 5 - Regression Gate & main.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement main.py with real V1 vs V2 regression comparison, Auto-Gate logic (Quality + Cost + Performance), and correct report format for check_lab.py.

**Architecture:** main.py orchestrates benchmark runs. Agent/Evaluator/Judge are stub interfaces that teammates plug into later. Regression gate compares 3 dimensions and auto-decides Release/Rollback.

**Tech Stack:** Python async, json, asyncio.

---

## Thiết kế độc lập

Phần này hoạt động **độc lập** với các teammate. Cách tiếp cận:

- `main.py` nhận Agent qua tham số → teammate thay stub bằng agent thật
- ExpertEvaluator/MultiModelJudge là placeholder class → teammate thay bằng implementation thật
- Interface rõ ràng: Agent trả `Dict` với `answer`, `contexts`, `retrieved_ids`, `metadata`
- `summary.json` format khớp chính xác với `check_lab.py` validation

## Ràng buộc từ check_lab.py

File `check_lab.py` kiểm tra `reports/summary.json` cần:
- `data["metadata"]` tồn tại, có `version`, `total`
- `data["metrics"]` tồn tại, có:
  - `hit_rate` (KHÔNG phải `avg_hit_rate`)
  - `agreement_rate`
  - `avg_score`

## File Structure

| File | Responsibility |
|---|---|
| `main.py` | Orchestrate V1 vs V2 benchmark, regression gate, save reports |
| `tests/test_main.py` | Tests for regression logic and report format |
| `tests/test_regression_gate.py` | Tests for release gate decision logic |

---

### Task 1: Extract Regression Gate Logic

**Files:**
- Create: `engine/regression_gate.py`
- Create: `tests/test_regression_gate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_regression_gate.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_regression_gate.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'engine.regression_gate'"

- [ ] **Step 3: Write implementation**

```python
# engine/regression_gate.py
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

        # Quality check
        quality_ok = delta_quality >= 0
        if not quality_ok:
            reasons.append("Quality giam")

        # Cost check
        max_cost_delta = v1_metrics["avg_cost_usd"] * (self.max_cost_increase_pct / 100)
        cost_ok = delta_cost <= max_cost_delta
        if not cost_ok:
            reasons.append(f"Cost tang qua {int(self.max_cost_increase_pct)}%")

        # Latency check
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_regression_gate.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/regression_gate.py tests/test_regression_gate.py
git commit -m "feat: add RegressionGate with 3-dimension release decision"
```

---

### Task 2: Update main.py with Real Regression Logic

**Files:**
- Modify: `main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write test for report format (must match check_lab.py)**

```python
# tests/test_main.py
import json
import os
import pytest


def test_summary_json_format():
    """
    summary.json phai co dung format ma check_lab.py kiem tra:
    - data["metadata"] co "version" va "total"
    - data["metrics"] co "hit_rate", "agreement_rate", "avg_score"
    """
    # Simulate what main.py should produce
    from engine.regression_gate import RegressionGate

    v1_metrics = {"avg_score": 3.5, "hit_rate": 0.7, "agreement_rate": 0.8, "avg_cost_usd": 0.001, "avg_latency_s": 1.0}
    v2_metrics = {"avg_score": 4.0, "hit_rate": 0.85, "agreement_rate": 0.9, "avg_cost_usd": 0.0013, "avg_latency_s": 1.2}

    gate = RegressionGate()
    regression = gate.evaluate(v1_metrics, v2_metrics)

    # Build summary.json as main.py should
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

    # Validate against check_lab.py expectations
    assert "metadata" in summary
    assert "metrics" in summary
    assert "version" in summary["metadata"]
    assert "total" in summary["metadata"]
    assert "hit_rate" in summary["metrics"]
    assert "agreement_rate" in summary["metrics"]
    assert "avg_score" in summary["metrics"]


def test_summary_json_regression_fields():
    """Regression section co du thong tin de phan tich."""
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
```

- [ ] **Step 2: Run test to verify it passes** (tests only validate data structures, not main.py itself)

Run: `python -m pytest tests/test_main.py -v`
Expected: All 2 tests PASS.

- [ ] **Step 3: Rewrite main.py**

Replace entire content of `main.py`:

```python
# main.py
import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from engine.retrieval_eval import RetrievalEvaluator
from engine.regression_gate import RegressionGate
from agent.main_agent import MainAgent


# ============================================================
# STUB CLASSES - Teammates se thay the bang implementation that
# ============================================================

class ExpertEvaluator:
    """
    Stub cho RAGAS evaluator.
    Nguoi 2 (Retrieval Engineer) se thay the bang implementation that.
    
    Interface: async score(case: Dict, resp: Dict) -> Dict
    Phai tra ve: {"faithfulness": float, "relevancy": float, "retrieval": {"hit_rate": float, "mrr": float}}
    """
    def __init__(self):
        self.retrieval_eval = RetrievalEvaluator()

    async def score(self, case, resp):
        expected = case.get("ground_truth_ids", [])
        retrieved = resp.get("retrieved_ids", [])
        hit_rate = self.retrieval_eval.calculate_hit_rate(expected, retrieved)
        mrr = self.retrieval_eval.calculate_mrr(expected, retrieved)
        return {
            "faithfulness": 0.0,  # TODO: Nguoi 2 implement RAGAS faithfulness
            "relevancy": 0.0,    # TODO: Nguoi 2 implement RAGAS relevancy
            "retrieval": {"hit_rate": hit_rate, "mrr": mrr}
        }


class MultiModelJudge:
    """
    Stub cho Multi-Judge.
    Nguoi 3 (AI Judge) se thay the bang implementation that.
    
    Interface: async evaluate_multi_judge(question, answer, ground_truth) -> Dict
    Phai tra ve: {"final_score": float, "agreement_rate": float, ...}
    """
    async def evaluate_multi_judge(self, q, a, gt):
        return {
            "final_score": 4.0,
            "agreement_rate": 0.8,
            "individual_scores": {"gpt-4o": 4, "claude-3-5": 4},
            "reasoning": "Stub - Nguoi 3 se implement."
        }


# ============================================================
# BENCHMARK RUNNER
# ============================================================

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

    # Aggregate metrics
    avg_score = sum(r["judge"]["final_score"] for r in results) / total
    avg_hit_rate = sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total
    avg_mrr = sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total
    avg_agreement = sum(r["judge"]["agreement_rate"] for r in results) / total
    avg_latency = sum(r["latency"] for r in results) / total

    # Cost: lay tu agent metadata neu co, default 0
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


# ============================================================
# MAIN
# ============================================================

async def main():
    # 1. Load golden dataset
    if not os.path.exists("data/golden_set.jsonl"):
        print("Thieu data/golden_set.jsonl. Hay chay 'python data/synthetic_gen.py' truoc.")
        return

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("File data/golden_set.jsonl rong.")
        return

    print(f"Da load {len(dataset)} test cases.")

    # 2. Run V1 Benchmark
    agent_v1 = MainAgent()
    agent_v1.name = "SupportAgent-v1"
    v1_results, v1_summary = await run_benchmark_with_results(agent_v1, "Agent_V1_Base", dataset)

    # 3. Run V2 Benchmark
    # TODO: Khi teammate implement AgentV2, doi MainAgent() thanh AgentV2()
    agent_v2 = MainAgent()
    agent_v2.name = "SupportAgent-v2"
    v2_results, v2_summary = await run_benchmark_with_results(agent_v2, "Agent_V2_Optimized", dataset)

    # 4. Regression Analysis
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

    # 5. Save reports
    os.makedirs("reports", exist_ok=True)

    # summary.json: format khop voi check_lab.py
    # check_lab.py can: metadata.version, metadata.total, metrics.hit_rate, metrics.agreement_rate, metrics.avg_score
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
```

- [ ] **Step 4: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: rewrite main.py with real regression gate and check_lab.py compatible format"
```

---

### Task 3: Update runner.py to Pass Full Response

**Files:**
- Modify: `engine/runner.py`

Runner hien tai chi luu `response["answer"]` vao `agent_response`. Can luu full response de main.py lay duoc `metadata.cost_usd`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_runner.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from engine.runner import BenchmarkRunner


@pytest.mark.asyncio
async def test_run_single_preserves_full_response():
    mock_agent = AsyncMock()
    mock_agent.query.return_value = {
        "answer": "2 thang",
        "contexts": ["context1"],
        "retrieved_ids": ["doc_0"],
        "metadata": {"cost_usd": 0.001, "tokens_used": 150, "model": "gpt-4o-mini"}
    }

    mock_evaluator = AsyncMock()
    mock_evaluator.score.return_value = {
        "faithfulness": 0.9,
        "relevancy": 0.8,
        "retrieval": {"hit_rate": 1.0, "mrr": 1.0}
    }

    mock_judge = AsyncMock()
    mock_judge.evaluate_multi_judge.return_value = {
        "final_score": 4.0,
        "agreement_rate": 0.8,
    }

    runner = BenchmarkRunner(mock_agent, mock_evaluator, mock_judge)
    result = await runner.run_single_test({
        "question": "Thu viec bao lau?",
        "expected_answer": "2 thang"
    })

    assert result["agent_response"] == "2 thang"
    assert result["agent_response_full"]["metadata"]["cost_usd"] == 0.001
    assert result["retrieved_ids"] == ["doc_0"]
    assert "latency" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runner.py -v`
Expected: FAIL — `agent_response_full` not in result.

- [ ] **Step 3: Update runner.py**

Replace content of `engine/runner.py`:

```python
# engine/runner.py
import asyncio
import time
from typing import List, Dict


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()

        # 1. Goi Agent
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start_time

        # 2. Chay RAGAS metrics
        ragas_scores = await self.evaluator.score(test_case, response)

        # 3. Chay Multi-Judge
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case.get("expected_answer", "")
        )

        return {
            "test_case": test_case["question"],
            "agent_response": response["answer"],
            "agent_response_full": response,
            "retrieved_ids": response.get("retrieved_ids", []),
            "latency": latency,
            "ragas": ragas_scores,
            "judge": judge_result,
            "status": "fail" if judge_result["final_score"] < 3 else "pass"
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        """
        Chay song song bang asyncio.gather voi gioi han batch_size de khong bi Rate Limit.
        """
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_runner.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/runner.py tests/test_runner.py
git commit -m "feat: update runner to preserve full agent response for cost tracking"
```

---

### Task 4: Init Files and Gitignore

**Files:**
- Create: `tests/__init__.py`
- Create: `engine/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create init files**

```python
# tests/__init__.py
```

```python
# engine/__init__.py
```

- [ ] **Step 2: Update .gitignore**

Append to `.gitignore`:

```
.env
reports/
__pycache__/
*.pyc
```

- [ ] **Step 3: Commit**

```bash
git add tests/__init__.py engine/__init__.py .gitignore
git commit -m "chore: add init files, update gitignore"
```

---

### Task 5: Smoke Test with Stub Agent

**Files:** (no new files)

- [ ] **Step 1: Tao golden_set nho de test**

```bash
echo '{"question": "Thu viec bao lau?", "expected_answer": "2 thang", "context": "Thoi gian thu viec la 2 thang.", "ground_truth_ids": ["cam_nang_0"], "metadata": {"difficulty": "easy", "type": "fact-check"}}' > data/golden_set.jsonl
echo '{"question": "Bao nhieu ngay phep?", "expected_answer": "12 ngay", "context": "Nhan vien co 12 ngay phep nam.", "ground_truth_ids": ["cam_nang_1"], "metadata": {"difficulty": "easy", "type": "fact-check"}}' >> data/golden_set.jsonl
```

- [ ] **Step 2: Run main.py**

Run: `python main.py`
Expected:
- V1 benchmark chay thanh cong (dung stub agent)
- V2 benchmark chay thanh cong
- Regression comparison hien thi delta 3 chieu
- Release Gate decision hien thi
- reports/summary.json va reports/benchmark_results.json duoc tao

- [ ] **Step 3: Validate with check_lab.py**

Run: `python check_lab.py`
Expected:
- "Tim thay: reports/summary.json" ✅
- "Tim thay: reports/benchmark_results.json" ✅
- "Da tim thay Retrieval Metrics" ✅
- "Da tim thay Multi-Judge Metrics" ✅
- "Da tim thay thong tin phien ban Agent" ✅
- "Bai lab da san sang de cham diem!" ✅

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test: smoke test with stub agent, verify check_lab.py passes"
```

---

## Sau khi teammates hoan thanh

Khi teammates plug code that vao:
1. **Nguoi 1** (Data Lead): Thay `data/golden_set.jsonl` bang 50+ test cases that
2. **Nguoi 2** (Retrieval): Thay `ExpertEvaluator` stub trong `main.py` bang RAGAS evaluator that
3. **Nguoi 3** (AI Judge): Thay `MultiModelJudge` stub trong `main.py` bang real multi-judge
4. **Nguoi 4** (Backend): Co the tinh chinh `runner.py` cho performance
5. **Agent team**: Thay `MainAgent()` bang `AgentV1()` va `AgentV2()` trong `main.py`

Interface khong doi — chi thay class implementation.
