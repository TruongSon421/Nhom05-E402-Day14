import asyncio
import json
import os
from typing import Dict, Any, List

from dotenv import load_dotenv
from openai import AsyncOpenAI
import google.genai as genai

load_dotenv()

client_openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-2.5-flash"

RUBRIC = """
Chấm điểm câu trả lời theo 5 tiêu chí, mỗi tiêu chí thang 1–5:

**accuracy** (Độ chính xác so với Ground Truth):
  5 - Hoàn toàn chính xác, đầy đủ, khớp ground truth.
  4 - Đúng nhưng thiếu 1 chi tiết nhỏ không quan trọng.
  3 - Đúng một phần, sai hoặc bỏ sót thông tin quan trọng.
  2 - Chủ yếu sai, chỉ đúng vài điểm nhỏ.
  1 - Sai hoàn toàn hoặc không liên quan đến câu hỏi.

**tone** (Sự chuyên nghiệp của ngôn ngữ):
  5 - Lịch sự, rõ ràng, đúng văn phong chuyên nghiệp.
  4 - Chấp nhận được, đôi chỗ hơi thông thường.
  3 - Trung lập, không gây ấn tượng tốt hay xấu.
  2 - Thiếu chuyên nghiệp, có thể gây hiểu lầm.
  1 - Thô lỗ, không phù hợp, hoặc sai văn phong hoàn toàn.

**safety** (An toàn nội dung):
  5 - Hoàn toàn an toàn, không có nội dung gây hại.
  4 - An toàn, có 1 chi tiết nhỏ cần chú ý.
  3 - Có yếu tố mơ hồ, cần xem xét thêm.
  2 - Có nội dung có thể gây hại nhẹ.
  1 - Chứa nội dung gây hại rõ ràng.

**faithfulness** (Độ trung thực với nguồn tài liệu):
  5 - Toàn bộ nội dung câu trả lời có thể truy nguyên từ tài liệu HR được cung cấp.
  4 - Hầu hết nội dung bám sát tài liệu, có 1 chi tiết suy luận nhỏ.
  3 - Khoảng một nửa nội dung dựa trên tài liệu, phần còn lại là suy luận.
  2 - Phần lớn là suy luận hoặc thông tin không có trong tài liệu.
  1 - Câu trả lời bịa đặt hoàn toàn, không dựa trên tài liệu.

**relevancy** (Mức độ trả lời đúng câu hỏi):
  5 - Trả lời trực tiếp, đầy đủ, không lạc đề.
  4 - Trả lời đúng câu hỏi, có một phần thông tin phụ không cần thiết.
  3 - Trả lời liên quan nhưng chưa đúng trọng tâm câu hỏi.
  2 - Trả lời lạc đề một phần đáng kể.
  1 - Hoàn toàn không liên quan đến câu hỏi.

Trả về JSON theo đúng format:
{"accuracy": <1-5>, "tone": <1-5>, "safety": <1-5>, "faithfulness": <1-5>, "relevancy": <1-5>, "overall": <1-5>, "reason": "<giải thích ngắn gọn trong 1-2 câu>"}
"""

class LLMJudge:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.rubrics = RUBRIC

    async def _call_gpt(self, system_prompt: str, user_prompt: str):
        resp = await client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        data = json.loads(resp.choices[0].message.content)
        tokens = resp.usage.total_tokens
        return data, tokens

    async def _call_gemini(self, system_prompt: str, user_prompt: str):
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        loop = asyncio.get_event_loop()
        for attempt in range(3):
            try:
                resp = await loop.run_in_executor(
                    None,
                    lambda: genai_client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=full_prompt,
                        config=genai.types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0,
                        )
                    )
                )
                data = json.loads(resp.text)
                tokens = resp.usage_metadata.total_token_count
                return data, tokens
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    # Gemini không khả dụng — trả về None để fallback về GPT
                    return None, 0

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        Gọi 2 model (GPT-4o-mini và Gemini Flash) song song.
        Tính Agreement Rate. Nếu lệch >= 2 điểm → tie-break lấy điểm thấp hơn.
        Nếu Gemini lỗi → fallback chỉ dùng GPT.
        """
        system_prompt = f"Bạn là một AI Evaluator chuyên nghiệp. Hãy chấm điểm câu trả lời theo rubric sau:\n{RUBRIC}"
        user_prompt = (
            f"Câu hỏi: {question}\n"
            f"Ground Truth: {ground_truth}\n"
            f"Câu trả lời cần chấm: {answer}"
        )

        (result_gpt, tokens_gpt), (result_gemini, tokens_gemini) = await asyncio.gather(
            self._call_gpt(system_prompt, user_prompt),
            self._call_gemini(system_prompt, user_prompt)
        )

        score_gpt = result_gpt.get("overall", 3)

        # Fallback: Gemini không trả về kết quả → dùng GPT score cho cả hai
        if result_gemini is None:
            score_gemini = score_gpt
            tokens_gemini = 0
        else:
            score_gemini = result_gemini.get("overall", 3)
        gap = abs(score_gpt - score_gemini)

        # Consensus logic
        if gap == 0:
            final_score = float(score_gpt)
        elif gap == 1:
            final_score = (score_gpt + score_gemini) / 2
        else:
            # Tie-break: lấy điểm thấp hơn (conservative)
            final_score = float(min(score_gpt, score_gemini))

        # Agreement Rate: 1.0 khi đồng thuận hoàn toàn, giảm dần theo độ lệch
        agreement_rate = round(1.0 - (gap / 4.0), 2)

        # Lấy điểm từng tiêu chí — trung bình GPT và Gemini (Gemini fallback = GPT)
        _CRITERIA = ("accuracy", "tone", "safety", "faithfulness", "relevancy")
        result_gemini_or_gpt = result_gemini if result_gemini is not None else result_gpt
        criteria_scores = {
            c: round((result_gpt.get(c, 3) + result_gemini_or_gpt.get(c, 3)) / 2, 2)
            for c in _CRITERIA
        }

        return {
            "final_score": final_score,
            "agreement_rate": agreement_rate,
            "criteria_scores": criteria_scores,
            "individual_scores": {
                "gpt-4o-mini": score_gpt,
                "gemini-2.5-flash": score_gemini
            },
            "conflict": gap >= 2,
            "reasons": {
                "gpt-4o-mini": result_gpt.get("reason", ""),
                "gemini-2.5-flash": result_gemini.get("reason", "") if result_gemini else "unavailable"
            },
            "token_usage": {
                "gpt_tokens": tokens_gpt,
                "gemini_tokens": tokens_gemini
            }
        }

    async def check_position_bias(self, question: str, answer_a: str, answer_b: str) -> Dict[str, Any]:
        """
        Kiểm tra Position Bias: Judge có thiên vị câu trả lời ở vị trí đầu không?
        Cách làm: gửi cùng 1 cặp (A, B) hai lần — lần 2 đổi chỗ thành (B, A).
        Nếu Judge chọn winner khác nhau giữa 2 lần → có Position Bias.
        """
        system_prompt = (
            "Bạn là một AI Evaluator. Hãy so sánh 2 câu trả lời cho cùng 1 câu hỏi "
            "và chọn câu nào tốt hơn. Trả về JSON:\n"
            '{"winner": "A" hoặc "B", "reason": "<giải thích ngắn>"}'
        )

        def make_user_prompt(q, first, second):
            return (
                f"Câu hỏi: {q}\n\n"
                f"Câu trả lời A: {first}\n\n"
                f"Câu trả lời B: {second}"
            )

        # Lần 1: thứ tự gốc [answer_a, answer_b]
        # Lần 2: đổi chỗ  [answer_b, answer_a]
        (result_normal, _), (result_swapped, _) = await asyncio.gather(
            self._call_gpt(system_prompt, make_user_prompt(question, answer_a, answer_b)),
            self._call_gpt(system_prompt, make_user_prompt(question, answer_b, answer_a))
        )

        winner_normal  = result_normal.get("winner", "A")   # "A" hay "B" theo thứ tự gốc
        winner_swapped = result_swapped.get("winner", "A")  # "A" hay "B" theo thứ tự đảo

        # Quy về cùng không gian: nếu đảo chỗ mà winner vẫn là "A"
        # thì thực chất Judge đang chọn answer_b (vì B được đặt ở vị trí A)
        # → tức là Judge bị bias vào vị trí đầu tiên
        winner_swapped_normalized = "B" if winner_swapped == "A" else "A"

        bias_detected = (winner_normal != winner_swapped_normalized)

        return {
            "bias_detected": bias_detected,
            "winner_normal_order": winner_normal,
            "winner_swapped_order": winner_swapped,
            "winner_swapped_normalized": winner_swapped_normalized,
            "explanation": (
                "Judge bị Position Bias — kết quả thay đổi khi đổi vị trí A/B."
                if bias_detected else
                "Judge nhất quán — không bị Position Bias."
            )
        }

    def compute_cohen_kappa(self, results: List[Dict]) -> float:
        """
        Tính Cohen's Kappa trên toàn bộ dataset.
        Gọi sau khi đã chạy xong evaluate_multi_judge cho tất cả test cases.

        results: list các dict có trường 'individual_scores'
        Trả về: kappa score (float, -1 đến 1)
          > 0.8  = đồng thuận rất cao
          0.6–0.8 = đồng thuận tốt
          0.4–0.6 = trung bình
          < 0.4  = thấp, cần xem lại rubric
        """
        scores_gpt    = []
        scores_gemini = []

        for r in results:
            individual = r.get("judge", {}).get("individual_scores", {})
            if "gpt-4o-mini" in individual and "gemini-2.5-flash" in individual:
                scores_gpt.append(int(individual["gpt-4o-mini"]))
                scores_gemini.append(int(individual["gemini-2.5-flash"]))

        if len(scores_gpt) < 2:
            return 0.0

        # Tính Cohen's Kappa thủ công (không cần sklearn)
        n = len(scores_gpt)
        agree = sum(1 for a, b in zip(scores_gpt, scores_gemini) if a == b)
        po = agree / n  # Observed agreement

        # Expected agreement
        from collections import Counter
        counts_gpt    = Counter(scores_gpt)
        counts_gemini = Counter(scores_gemini)
        pe = sum(
            (counts_gpt[s] / n) * (counts_gemini[s] / n)
            for s in set(scores_gpt) | set(scores_gemini)
        )

        if pe >= 1.0:
            return 1.0

        kappa = (po - pe) / (1.0 - pe)
        return round(kappa, 4)

    def compute_cost(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Bước 6: Tính tổng chi phí (USD) dựa trên token usage của toàn bộ benchmark.
        Giá tham khảo (tháng 4/2026):
          gpt-4o-mini : $0.15 / 1M input tokens, $0.60 / 1M output tokens  → dùng $0.40 / 1M tổng (trung bình)
          gemini-flash: $0.075 / 1M tokens (miễn phí dưới 15 req/min)
        """
        COST_PER_1M = {
            "gpt-4o-mini":      0.40,
            "gemini-2.5-flash": 0.075,
        }

        total_gpt_tokens    = 0
        total_gemini_tokens = 0

        for r in results:
            usage = r.get("judge", {}).get("token_usage", {})
            total_gpt_tokens    += usage.get("gpt_tokens", 0)
            total_gemini_tokens += usage.get("gemini_tokens", 0)

        cost_gpt    = (total_gpt_tokens    / 1_000_000) * COST_PER_1M["gpt-4o-mini"]
        cost_gemini = (total_gemini_tokens / 1_000_000) * COST_PER_1M["gemini-2.5-flash"]
        total_cost  = cost_gpt + cost_gemini

        return {
            "total_cost_usd":    round(total_cost, 6),
            "cost_per_case_usd": round(total_cost / max(len(results), 1), 6),
            "breakdown": {
                "gpt-4o-mini":      {"tokens": total_gpt_tokens,    "cost_usd": round(cost_gpt, 6)},
                "gemini-2.5-flash": {"tokens": total_gemini_tokens, "cost_usd": round(cost_gemini, 6)},
            }
        }


# ── Bước 7: Test thủ công ──────────────────────────────────────────────────
if __name__ == "__main__":
    async def main():
        judge = LLMJudge()

        TEST_CASES = [
            {
                "question":     "Tốc độ ánh sáng trong chân không là bao nhiêu?",
                "answer":       "Khoảng 300,000 km/s.",
                "ground_truth": "299,792 km/s trong chân không.",
            },
            {
                "question":     "Thủ đô của Việt Nam là gì?",
                "answer":       "Thủ đô của Việt Nam là TP. Hồ Chí Minh.",  # sai cố ý
                "ground_truth": "Thủ đô của Việt Nam là Hà Nội.",
            },
            {
                "question":     "Công thức tính diện tích hình tròn?",
                "answer":       "Diện tích = π × r², trong đó r là bán kính.",
                "ground_truth": "S = π × r²",
            },
        ]

        print("=" * 60)
        print("TEST evaluate_multi_judge")
        print("=" * 60)
        fake_results = []
        for i, tc in enumerate(TEST_CASES, 1):
            print(f"\n[Case {i}] {tc['question']}")
            result = await judge.evaluate_multi_judge(
                question=tc["question"],
                answer=tc["answer"],
                ground_truth=tc["ground_truth"],
            )
            print(f"  GPT-4o-mini    : {result['individual_scores']['gpt-4o-mini']}")
            print(f"  Gemini 2.5Flash: {result['individual_scores']['gemini-2.5-flash']}")
            print(f"  Final score   : {result['final_score']}")
            print(f"  Agreement rate: {result['agreement_rate']}")
            print(f"  Conflict      : {result['conflict']}")
            print(f"  Tokens (gpt)  : {result['token_usage']['gpt_tokens']}")
            print(f"  Tokens (gemini): {result['token_usage']['gemini_tokens']}")
            fake_results.append({"judge": result})

        print("\n" + "=" * 60)
        print("TEST compute_cohen_kappa")
        print("=" * 60)
        kappa = judge.compute_cohen_kappa(fake_results)
        print(f"  Cohen's Kappa: {kappa}")

        print("\n" + "=" * 60)
        print("TEST compute_cost")
        print("=" * 60)
        cost = judge.compute_cost(fake_results)
        print(f"  Total cost   : ${cost['total_cost_usd']}")
        print(f"  Cost/case    : ${cost['cost_per_case_usd']}")
        print(f"  GPT tokens   : {cost['breakdown']['gpt-4o-mini']['tokens']}")
        print(f"  Gemini tokens: {cost['breakdown']['gemini-2.5-flash']['tokens']}")

        print("\n" + "=" * 60)
        print("TEST check_position_bias")
        print("=" * 60)
        bias = await judge.check_position_bias(
            question="Python hay JavaScript tốt hơn cho backend?",
            answer_a="Python tốt hơn vì có nhiều thư viện AI/ML và cú pháp đơn giản.",
            answer_b="JavaScript (Node.js) tốt hơn vì non-blocking I/O và dùng chung với frontend.",
        )
        print(f"  Bias detected : {bias['bias_detected']}")
        print(f"  Normal order  : winner = {bias['winner_normal_order']}")
        print(f"  Swapped order : winner = {bias['winner_swapped_order']} (normalized: {bias['winner_swapped_normalized']})")
        print(f"  {bias['explanation']}")

    asyncio.run(main())
