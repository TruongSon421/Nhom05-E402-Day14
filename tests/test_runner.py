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
