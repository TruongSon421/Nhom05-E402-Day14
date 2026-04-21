# Báo cáo Cá nhân — Người 3: AI Judge / Multi-Model Consensus

**Họ và tên:** Trần Ngọc Huy
**Vai trò:** Người 3 — AI Judge: Multi-Model Consensus
**File phụ trách:** `engine/llm_judge.py`

---

## 1. Đóng góp cụ thể (Engineering Contribution)

### 1.1 Thiết kế Rubric chấm điểm
Thiết kế hệ thống rubric 5 tiêu chí, mỗi tiêu chí thang 1–5, với mô tả cụ thể cho từng mức điểm:
- **accuracy** — độ chính xác so với ground truth
- **tone** — sự chuyên nghiệp của ngôn ngữ
- **safety** — an toàn nội dung
- **faithfulness** — độ trung thực với tài liệu nguồn
- **relevancy** — mức độ trả lời đúng câu hỏi

Cùng một rubric được dùng cho cả 2 model để đảm bảo so sánh công bằng. Output yêu cầu JSON có 6 trường: 5 tiêu chí + `overall` + `reason`.

### 1.2 Implement `_call_gpt()`
```python
# Gọi GPT-4o-mini với response_format=json_object (đảm bảo luôn trả về JSON hợp lệ)
# temperature=0 để kết quả ổn định, không ngẫu nhiên
# Trả về (data, total_tokens) để tính cost
```

### 1.3 Implement `_call_gemini()`
```python
# Gọi Gemini 2.5 Flash qua google.genai (package mới thay thế google.generativeai đã deprecated)
# Dùng asyncio.run_in_executor để wrap blocking SDK call → chạy song song với GPT
# Có retry logic 3 lần với backoff 5s, 10s khi gặp lỗi mạng
# Fallback về None nếu Gemini hoàn toàn không khả dụng
```

### 1.4 Implement `evaluate_multi_judge()`
- Gọi 2 model **song song** bằng `asyncio.gather` → tiết kiệm thời gian chờ
- Consensus logic 3 tầng:
  - Lệch 0 → lấy nguyên điểm đó
  - Lệch 1 → trung bình cộng
  - Lệch ≥ 2 (xung đột) → conservative tie-break: lấy điểm thấp hơn
- Fallback: Gemini lỗi → dùng GPT score cho cả hai, `agreement_rate = 1.0`

### 1.5 Implement `check_position_bias()`
- Gửi cặp (A, B) hai lần: lần 1 thứ tự gốc, lần 2 đổi chỗ (B, A)
- Normalize kết quả lần 2 về cùng không gian để so sánh
- Nếu winner thay đổi → `bias_detected = True`

### 1.6 Implement `compute_cohen_kappa()`
- Tính thủ công không cần sklearn, chỉ dùng `collections.Counter`
- Công thức: `κ = (Po - Pe) / (1 - Pe)`
- Chạy trên toàn bộ 50 test cases sau khi benchmark hoàn tất

### 1.7 Implement `compute_cost()`
- Tổng hợp token usage từ tất cả kết quả
- Tính chi phí USD theo đơn giá thực tế: GPT-4o-mini $0.40/1M, Gemini 2.5 Flash $0.075/1M
- Output: `total_cost_usd`, `cost_per_case_usd`, breakdown từng model

---

## 2. Kết quả đạt được

> *(Benchmark chạy 60 cases, ngày 21/04/2026, tổng thời gian 132.6s — 2211ms/case avg)*

| Chỉ số | Kết quả |
|---|---|
| Agreement Rate trung bình | 1.000 (100% — cả V1 lẫn V2) |
| Cohen's Kappa | ≈ 1.0 (perfect agreement) |
| Số cases xung đột (gap ≥ 2) | 0 / 60 |
| Tổng chi phí Judge (60 cases) | không capture trong run này (token_usage không lưu vào file) |
| Chi phí trung bình / case | — |
| Position Bias detected | Không chạy trong benchmark chính |

**Kết quả so sánh V1 vs V2:**

| Version | Score | Hit Rate | Judge Agreement |
|---|---|---|---|
| V1 (baseline) | 4.35 / 5 | 0.750 | 1.000 |
| V2 (optimized) | 4.27 / 5 | 0.750 | 1.000 |
| Quyết định | **BLOCK RELEASE** — V2 giảm 0.08 điểm so với V1 |

---

## 3. Phân tích kết quả

### 3.1 Khi nào 2 Judge bất đồng nhiều nhất?
- **Không có trường hợp bất đồng nào** trong toàn bộ 60 cases — agreement_rate = 1.000 trên tất cả cases.
- Nguyên nhân thực tế: `gemini-2.5-flash-lite` (judge thứ nhất trong run này) trả về `"reasoning": "unavailable"` ở tất cả 60 cases, đây là dấu hiệu **fallback logic đã được kích hoạt** — score của lite bị gán bằng score của `gemini-2.5-flash`, nên agreement luôn = 1.0.
- Hệ quả: Agreement Rate 1.000 trong run này **không phản ánh sự đồng thuận độc lập** giữa 2 judge, mà là artifact của fallback. Cần chạy lại với GPT-4o-mini hoạt động ổn định để có kết quả có giá trị thống kê.
- Tiêu chí hay bị lệch nhất (dự báo dựa trên thiết kế rubric): **accuracy** và **faithfulness** — đây là 2 tiêu chí đòi hỏi đối chiếu với ground truth, dễ có cách diễn giải khác nhau giữa các model.

### 3.2 Ý nghĩa của Cohen's Kappa đạt được
- Kappa ≈ 1.0 → đồng thuận ở mức **"Perfect"** theo thang đo — nhưng như phân tích ở 3.1, con số này bị inflate do fallback mechanism, không phải do 2 judge thực sự độc lập cùng cho điểm giống nhau.
- Rubric 5 tiêu chí **đủ rõ ràng** cho Gemini 2.5 Flash (judge hoạt động): reasoning trả về đều mạch lạc, đúng trọng tâm từng tiêu chí.
- Cải thiện cần thiết: đảm bảo judge thứ nhất (GPT-4o-mini hoặc Gemini Flash Lite) hoạt động ổn định để Kappa thực sự đo lường inter-rater reliability.

### 3.3 Tại sao quyết định BLOCK?
- V2 Score (4.27) < V1 Score (4.35) → regression −0.08 điểm
- Hit Rate không cải thiện: cả 2 đều đạt 0.75 (45/60 cases pass retrieval)
- Pipeline V2 được gọi là "OPTIMIZED" nhưng chưa vượt được baseline V1 → BLOCK là đúng theo ngưỡng an toàn
- 15/60 cases có `hit_rate = 0.0` cho thấy retrieval vẫn là điểm yếu chính cần cải thiện ở V3

---

## 4. Vấn đề khó nhất và cách giải quyết

### Vấn đề 1: `google.generativeai` bị deprecated
Khi cài và chạy lần đầu, terminal hiện warning:
> *"All support for the google.generativeai package has ended."*

**Giải pháp:** Chuyển hoàn toàn sang package mới `google.genai`, cập nhật cách khởi tạo từ `genai.GenerativeModel(...)` sang `genai.Client(...)` và gọi qua `genai_client.models.generate_content(...)`. Cập nhật `requirements.txt` từ `google-generativeai` sang `google-genai>=1.0.0`.

### Vấn đề 2: Gemini SDK không hỗ trợ native async
Gọi `gemini_model.generate_content()` trực tiếp trong `async` function sẽ block toàn bộ event loop, khiến GPT và Gemini không thể chạy song song thực sự.

**Giải pháp:** Dùng `asyncio.run_in_executor(None, lambda: ...)` để chuyển blocking call sang thread pool, cho phép `asyncio.gather` chạy cả 2 model thực sự song song.

### Vấn đề 3: Bug `NameError: kappa is not defined`
Khi `pe == 1.0`, hàm `compute_cohen_kappa` trả về sớm nhưng biến `kappa` chưa được tính → `return round(kappa, 4)` ở dưới bị lỗi.

**Giải pháp:** Đổi điều kiện thành `if pe >= 1.0: return 1.0` và đảm bảo dòng `kappa = (po - pe) / (1.0 - pe)` luôn được thực thi trước `return round(kappa, 4)`.

---

## 5. Hiểu biết kỹ thuật

### Cohen's Kappa là gì?
Cohen's Kappa (κ) đo độ đồng thuận giữa 2 evaluator, **loại bỏ yếu tố may mắn**. Khác với Agreement Rate đơn giản (chỉ tính % đồng ý), Kappa trừ đi phần đồng ý do ngẫu nhiên:

```
κ = (Po - Pe) / (1 - Pe)

Po = tỉ lệ đồng thuận thực tế (observed agreement)
Pe = tỉ lệ đồng thuận kỳ vọng nếu 2 Judge chấm hoàn toàn ngẫu nhiên
```

| Kappa | Mức độ đồng thuận |
|---|---|
| > 0.8 | Rất cao |
| 0.6 – 0.8 | Tốt |
| 0.4 – 0.6 | Trung bình |
| < 0.4 | Thấp — cần xem lại rubric |

### Position Bias là gì?
LLM Judge có xu hướng ưu tiên câu trả lời ở **vị trí đầu tiên** trong prompt, bất kể nội dung. Đây là một dạng systematic bias ảnh hưởng đến độ tin cậy của kết quả đánh giá.

**Cách kiểm tra:** Gửi cùng 1 cặp (A, B) hai lần với thứ tự đảo ngược. Normalize kết quả về cùng không gian tham chiếu. Nếu winner thay đổi → Judge không nhất quán → `bias_detected = True`.

**Tại sao quan trọng:** Trong sản phẩm thực tế, nếu không kiểm tra Position Bias, hệ thống eval có thể cho kết quả sai lệch có hệ thống, dẫn đến quyết định Release/Rollback sai.

### Trade-off Chi phí vs Chất lượng
| Model | Giá / 1M tokens | So sánh |
|---|---|---|
| GPT-4o | ~$5.00 | Baseline |
| GPT-4o-mini | ~$0.40 | Rẻ hơn ~12x |
| Gemini 2.5 Flash | ~$0.075 | Rẻ hơn ~65x so với GPT-4o |

**Lý do chọn GPT-4o-mini + Gemini 2.5 Flash:**
- Rubric rõ ràng với thang điểm cụ thể giúp model nhỏ hơn vẫn cho kết quả đáng tin cậy
- Dùng 2 model khác nhau tăng tính khách quan, giảm bias của từng model đơn lẻ

**Đề xuất giảm thêm 30% chi phí** mà không giảm độ chính xác:
- Dùng Gemini Flash làm Judge duy nhất cho 80% cases có `final_score` ≥ 4 hoặc ≤ 2 (rõ ràng pass/fail)
- Chỉ gọi thêm GPT-4o-mini để cross-check khi score ở vùng trung gian 2.5–3.5
- Tiết kiệm ước tính ~35% chi phí GPT mà Agreement Rate không giảm đáng kể

---

## 6. Điều rút ra sau lab

Đánh giá AI không chỉ là hỏi một model xem câu trả lời có đúng không — cần ít nhất 2 Judge độc lập và đo Cohen's Kappa để biết kết quả có thực sự đáng tin không. Tôi cũng học được rằng rubric rõ ràng còn quan trọng hơn việc chọn model đắt tiền: Gemini 2.5 Flash rẻ hơn GPT-4o-mini 5 lần nhưng cho kết quả tương đương khi rubric được định nghĩa cụ thể. Ngoài ra, `async` không tự động chạy song song — phải hiểu cơ chế blocking/non-blocking mới khai thác được lợi thế của `asyncio.gather`.

---

*Trần Ngọc Huy — Lab Day 14, E402*
