# Báo cáo Cá nhân — Người 3: AI Judge / Multi-Model Consensus

**Họ và tên:** [Điền tên đầy đủ]
**Vai trò:** Người 3 — AI Judge: Multi-Model Consensus
**File phụ trách:** `engine/llm_judge.py`

---

## 1. Đóng góp cụ thể (Engineering Contribution)

> Mô tả chính xác bạn đã implement phần nào — càng cụ thể càng tốt.

- Thiết kế và viết `RUBRIC` chấm điểm 5 tiêu chí (accuracy, tone, safety, faithfulness, relevancy) thang 1–5
- Implement `_call_gpt()`: gọi GPT-4o-mini với `response_format=json_object`, lấy token usage
- Implement `_call_gemini()`: gọi Gemini 2.5 Flash qua `google.genai`, dùng `asyncio.run_in_executor` để không block event loop, có retry 3 lần khi lỗi
- Implement `evaluate_multi_judge()`: gọi 2 model song song bằng `asyncio.gather`, consensus logic (lệch 0/1/≥2), fallback khi Gemini không khả dụng
- Implement `check_position_bias()`: đổi chỗ A/B để phát hiện thiên vị vị trí của Judge
- Implement `compute_cohen_kappa()`: tính độ đồng thuận trên toàn dataset
- Implement `compute_cost()`: tính chi phí USD thực tế theo token usage

---

## 2. Kết quả đạt được

> Điền sau khi chạy benchmark thực tế với 50 cases.

| Chỉ số | Kết quả |
|---|---|
| Agreement Rate trung bình | X.XX |
| Cohen's Kappa | X.XX |
| Số cases xung đột (gap ≥ 2) | X / 50 |
| Tổng chi phí Judge (50 cases) | $X.XXXXXX |
| Chi phí trung bình / case | $X.XXXXXX |
| Position Bias detected | True / False |

---

## 3. Phân tích kết quả

### 3.1 Khi nào 2 Judge bất đồng nhiều nhất?

> Điền sau khi có kết quả. Ví dụ gợi ý:

- Loại câu hỏi mà 2 Judge hay bất đồng: [câu hỏi mơ hồ / câu hỏi kỹ thuật / câu hỏi có nhiều đáp án đúng...]
- Tiêu chí hay bị lệch nhất: [accuracy / tone / safety / faithfulness / relevancy]
- Nhận xét: GPT-4o-mini thường chấm [cao hơn / thấp hơn] Gemini 2.5 Flash

### 3.2 Ý nghĩa của Cohen's Kappa đạt được

> Kappa = X.XX → đồng thuận ở mức [rất cao / tốt / trung bình / thấp]
> Điều này có nghĩa là rubric [đủ rõ ràng / cần làm rõ thêm ở tiêu chí ...]

---

## 4. Vấn đề khó nhất và cách giải quyết

> Mô tả 1–2 vấn đề thực tế gặp phải khi code.

**Vấn đề 1:** `google.generativeai` bị deprecated
- **Giải pháp:** Chuyển sang `google.genai` (package mới), cập nhật cách khởi tạo client và gọi API

**Vấn đề 2:** Gemini không hỗ trợ native async
- **Giải pháp:** Dùng `asyncio.run_in_executor(None, lambda: ...)` để wrap blocking call thành coroutine, giúp chạy song song với GPT mà không block event loop

**Vấn đề 3:** [Thêm vấn đề thực tế bạn gặp...]
- **Giải pháp:** [...]

---

## 5. Hiểu biết kỹ thuật

### Cohen's Kappa là gì?
Cohen's Kappa (κ) đo độ đồng thuận giữa 2 evaluator, **loại bỏ yếu tố may mắn**. Khác với Agreement Rate đơn giản (chỉ tính % đồng ý), Kappa trừ đi phần đồng ý do ngẫu nhiên mà ra.

Công thức: `κ = (Po - Pe) / (1 - Pe)`
- Po = tỉ lệ đồng thuận thực tế
- Pe = tỉ lệ đồng thuận kỳ vọng nếu 2 Judge chấm ngẫu nhiên

Kappa = 1.0 → hoàn toàn đồng thuận | Kappa = 0 → chỉ đồng thuận do may mắn | Kappa < 0 → tệ hơn cả ngẫu nhiên

### Position Bias là gì?
LLM có xu hướng ưu tiên câu trả lời ở **vị trí đầu tiên** trong prompt, không phụ thuộc nội dung. Kiểm tra bằng cách gửi cùng cặp (A, B) hai lần với thứ tự đảo ngược. Nếu winner thay đổi → Judge bị bias.

### Trade-off Chi phí vs Chất lượng
- **GPT-4o-mini**: ~$0.40/1M tokens — rẻ hơn GPT-4o ~15–20 lần, đủ tốt cho eval có rubric rõ ràng
- **Gemini 2.5 Flash**: ~$0.075/1M tokens — rẻ hơn GPT-4o-mini ~5 lần
- **Đề xuất giảm 30% chi phí**: Dùng Gemini Flash làm Judge chính cho 80% cases thông thường, chỉ gọi thêm GPT-4o-mini khi score < 3 hoặc cần tie-break — tiết kiệm ~35% chi phí mà không giảm độ chính xác đáng kể.

---

## 6. Điều rút ra sau lab

> 2–3 câu ngắn gọn về điều bạn học được.

[Điền sau khi hoàn thành lab...]
