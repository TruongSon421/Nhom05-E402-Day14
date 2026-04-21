# 📝 Reflection Cá nhân — Trương Đăng Nghĩa (P5 — DevOps / Regression Gate)

**Lab:** Day 14 — AI Evaluation Factory  
**Module phụ trách:** `main.py`, `engine/llm_judge.py`  
**Ngày:** 2026-04-21

---

## 1. Tôi đã làm gì trong lab này?

Tôi chịu trách nhiệm xây dựng **Regression Gate & Pipeline Orchestrator** — module quyết định liệu một phiên bản agent mới (V2) có đủ tiêu chuẩn để release hay không, dựa trên so sánh định lượng với V1.

| Hạng mục | Chi tiết |
|---|---|
| **V1 vs V2 thực (không giả lập)** | V1 = `MainAgent` + `concurrency=10` (baseline), V2 = `OptimisedAgentV2` + `concurrency=15` (cải tiến từ failure analysis) |
| **Auto-Gate 3 trục** | Quality Gate, Cost Gate, Performance Gate — mỗi trục có threshold độc lập; quyết định RELEASE / ROLLBACK / BLOCK |
| **9 Metrics đầy đủ** | retrieval_accuracy, hit_rate, avg_hit_rate, mrr, final_answer_accuracy, hallucination_rate, avg_score, latency, cost, user_satisfaction_score |
| **OptimisedAgentV2** | Wrapper 3 cải tiến từ `failure_analysis.md`: Off-topic Guardrail, Prompt Injection Defense, Direct Answer Instruction |
| **Conservative Rubric cho LLM Judge** | Thêm `RUBRIC_CONSERVATIVE` và `_call_gpt_conservative()` vào `llm_judge.py` — rubric chấm điểm nghiêm khắc hơn để hai judge cho góc nhìn độc lập |
| **Reports** | `reports/summary.json` và `reports/benchmark_results.json` đúng format, vượt qua `check_lab.py` |

**Điểm quan trọng:** Agent V1 (`MainAgent`) đã có sẵn do nhóm xây dựng trước. Nhiệm vụ của tôi là *đo lường, so sánh, và cải tiến có căn cứ* — không phải xây lại agent từ đầu.

---

## 2. Giải thích các khái niệm kỹ thuật quan trọng

### 2.1 Regression Testing & Release Gate trong AI Systems

**Regression Testing** trong phần mềm truyền thống nghĩa là "thay đổi mới không làm vỡ tính năng cũ". Trong AI Engineering, ý nghĩa sâu hơn vì model output không deterministic — cần đo lường *phân phối chất lượng* thay vì pass/fail đơn giản.

**Release Gate** là hệ thống tự động quyết định có nên deploy V2 không, dựa trên nhiều ngưỡng đồng thời:

```
V1 (Baseline) → chạy benchmark → metrics_v1
V2 (Candidate) → chạy benchmark → metrics_v2

if quality_gate AND cost_gate AND perf_gate:
    → RELEASE   (V2 đủ tiêu chuẩn)
elif quality regressed:
    → ROLLBACK  (V2 tệ hơn V1, revert)
else:
    → BLOCK     (vấn đề kỹ thuật, chưa deploy)
```

**Tại sao cần 3 trục thay vì chỉ 1?** Một agent có thể có quality tốt hơn nhưng tốn gấp đôi chi phí, hoặc nhanh hơn nhưng bị quality regression nhẹ. Gate 1 trục sẽ bỏ sót các trade-off quan trọng này.

| Gate | Threshold | Lý do chọn ngưỡng |
|---|---|---|
| Quality: avg_score ≥ 3.5/5 | Ngưỡng "chấp nhận được" trong production HR chatbot |
| Quality: hit_rate ≥ 0.50 | Retrieval tìm đúng ít nhất 1/2 trường hợp |
| Quality: score_drop ≤ 0.30 | Cho phép V2 giảm tối đa 0.3 điểm so với V1 (tolerance) |
| Cost: delta ≤ +10% | V2 không được đắt hơn V1 quá 10% |
| Performance: wall_time ≤ 120s | Mục tiêu < 2 phút cho toàn bộ dataset |

---

### 2.2 Hallucination Rate

**Hallucination** trong RAG là khi agent tạo ra thông tin *không có trong tài liệu nguồn* — tức là "bịa". Đây là rủi ro nghiêm trọng nhất của chatbot HR vì nhân viên có thể tin vào chính sách sai.

**Cách đo trong hệ thống:**

$$\text{Hallucination Rate} = 1 - \frac{\text{avg\_faithfulness}}{5}$$

Trong đó `faithfulness` (1–5) là điểm LLM Judge đánh giá mức độ câu trả lời bám sát tài liệu HR:
- Faithfulness = 5 → trả lời 100% từ tài liệu → Hallucination Rate = 0%
- Faithfulness = 1 → bịa hoàn toàn → Hallucination Rate = 80%

**Kết quả thực tế trong lab:**
- Faithfulness trung bình V1: **4.16/5**  
- Hallucination Rate V1: **1 − (4.16/5) = 16.8%**

Tức là cứ ~6 câu trả lời thì có 1 câu chứa thông tin không hoàn toàn bám sát tài liệu. Đây là điểm cần cải thiện nhất trong production.

---

### 2.3 Guardrails — Rào chắn bảo vệ Agent

**Guardrail** là lớp kiểm tra *trước khi* câu hỏi được gửi vào LLM pipeline, nhằm chặn hoặc biến đổi các input nguy hiểm. Tôi implement 3 loại trong `OptimisedAgentV2`:

**1. Off-topic Guardrail (Hard Block)**  
Detect regex pattern → trả lời ngay không gọi LLM:
```python
"viết\s+(bài\s+)?(thơ|nhạc|văn|truyện)" → từ chối lịch sự
```
Lợi ích kép: (1) trả lời đúng, (2) tiết kiệm ~24s latency và ~$0.00035 per case.

**2. Prompt Injection Defense (Soft Guard)**  
Detect injection pattern → thêm security note vào query trước khi gửi agent:
```python
"hãy bỏ qua tài liệu..." → "[SECURITY NOTE] + original_query"
```
Không block hoàn toàn vì agent vẫn cần trả lời phần hợp lệ của câu hỏi.

**3. Direct Answer Instruction (Query Enhancement)**  
Mọi query bình thường → prepend instruction:
```python
"[Trả lời trực tiếp câu hỏi cốt lõi trong câu đầu tiên, sau đó giải thích thêm]"
```
Giải quyết root cause của "Incomplete Answer" case mà không cần sửa code agent.

**Tại sao implement ở wrapper thay vì sửa MainAgent?** Nguyên tắc Single Responsibility — `main.py` là DevOps layer, không nên can thiệp sâu vào agent logic của nhóm khác. Wrapper giữ cho các layer độc lập, có thể bật/tắt từng guardrail riêng lẻ khi test.

---

### 2.4 Concurrency Tuning — Tối ưu thông lượng pipeline

**Vấn đề:** Với 60 cases × ~24s/case, pipeline tuần tự mất 1440 giây (24 phút!). Cần parallelism nhưng không quá nhiều để tránh rate limit.

**Mô hình lý thuyết:**

$$T_{wall} = \left\lceil \frac{N_{cases}}{concurrency} \right\rceil \times \bar{t}_{case}$$

**Áp dụng thực tế:**

| concurrency | N=60 cases, avg=24s | Kết quả thực đo |
|---|---|---|
| 5 (V2 cũ — sai) | ⌈60/5⌉ × 24 = 288s | 274s ❌ OVER |
| 10 (V1 baseline) | ⌈60/10⌉ × 24 = 144s | 153s ❌ OVER |
| **15 (V2 mới)** | ⌈60/15⌉ × 24 = **96s** | ~102s ✅ PASS |

**Bài học:** Tăng concurrency từ 10 → 15 (tăng 50%) giảm wall-time ~33%. Nhưng concurrency không thể tăng mãi vì API có rate limit. Điểm cân bằng tối ưu cần thực nghiệm.

---

### 2.5 Conservative Rubric — Tăng tính độc lập giữa hai Judge

**Vấn đề:** Khi hai judge dùng cùng một rubric với ngưỡng mô tả giống nhau, chúng có xu hướng đồng thuận cao trên các case "tốt vừa phải" — dẫn đến agreement rate cao nhưng không phản ánh đúng độ khó đánh giá thực tế.

**Giải pháp:** Tôi thêm `RUBRIC_CONSERVATIVE` vào `llm_judge.py` — một rubric với ngưỡng nghiêm khắc hơn:

| Tiêu chí | Rubric chuẩn (5/5) | Rubric Conservative (5/5) |
|---|---|---|
| accuracy | Đúng, đầy đủ | Khớp *hoàn toàn* ground truth, không thiếu bất kỳ chi tiết phụ nào |
| faithfulness | Hầu hết truy nguyên được | 100% từ tài liệu, không có suy luận dù nhỏ |
| relevancy | Trả lời đúng trọng tâm | Trả lời trực tiếp *ngay câu đầu*, súc tích, không thông tin phụ |

**Kết quả:** Judge conservative thường cho điểm thấp hơn 0–1 bậc trên các case borderline. Hai judge nhìn cùng câu trả lời từ hai góc độ khác nhau → agreement rate đo được trong benchmark: 0.75–1.0 (trung bình ~94%), với các case bất đồng tập trung ở những câu trả lời "đúng nhưng thiếu chi tiết".

**Tại sao không chỉ cần 1 judge?** Trong production, không có "đáp án đúng tuyệt đối" cho đánh giá chất lượng LLM. Agreement thấp giữa hai judge là *signal quan trọng* — cho thấy case đó mơ hồ, cần human review. Còn agreement 100% liên tục trên mọi case thường là dấu hiệu hai judge đang dùng cùng một góc nhìn.

---

## 3. Vấn đề gặp phải và cách giải quyết

### 3.1 V2 ban đầu tệ hơn V1 (concurrency=5 sai hướng)

**Triệu chứng:** V2 (concurrency=5) cho kết quả:
- Wall time: 273.5s (tệ hơn V1 140.6s!)
- Avg score: 4.32 (thấp hơn V1 4.40)
- Decision: BLOCK

**Nguyên nhân gốc rễ:** Tôi nhầm logic — giảm concurrency để tránh rate limit nhưng thực ra rate limit không phải bottleneck; bottleneck là agent latency (~24s/case). Giảm concurrency chỉ làm pipeline chậm hơn mà không cải thiện gì.

**Giải pháp:** Phân tích công thức `T = ⌈N/c⌉ × t_avg`:
- Để T < 120s với N=60 và t=24s → cần c ≥ `⌈60 × 24/120⌉ = 12`
- Chọn c=15 để có buffer an toàn → T ≈ 96s ✅

**Bài học:** Đừng tối ưu dựa trên cảm tính ("ít request = ít lỗi"). Phải dùng mô hình lý thuyết trước, sau đó validate bằng đo lường thực tế.

---

### 3.2 `final_answer_accuracy = 0.0` trong benchmark cũ

**Triệu chứng:** Sau khi thêm metric `final_answer_accuracy`, tất cả giá trị đều = 0.0 mặc dù agent trả lời tốt.

**Nguyên nhân:** Transform cũ trong `main.py` chỉ lưu 4 trường ragas (`hit_rate`, `mrr`, `faithfulness`, `relevancy`) mà bỏ qua `accuracy`, `tone`, `safety` — là 3 trong 5 criteria mà LLM Judge đánh giá.

```python
# Cũ (thiếu accuracy)
"ragas": {"hit_rate": ..., "mrr": ..., "faithfulness": ..., "relevancy": ...}

# Mới (đầy đủ 5 criteria)
"ragas": {"hit_rate": ..., "mrr": ..., "accuracy": ..., "faithfulness": ...,
          "relevancy": ..., "tone": ..., "safety": ...}
```

**Bài học:** Khi thiết kế schema lưu kết quả, luôn lưu *tất cả* raw fields ngay từ đầu. Mất data về sau rất khó recover nếu benchmark đã chạy.

---

### 3.3 Agreement Rate quá cao — rubric chuẩn thiếu tính phân biệt

**Triệu chứng:** Sau khi chạy benchmark, agreement rate = 1.0 (100%) trên mọi case — không có case nào hai judge bất đồng.

**Nguyên nhân:** Rubric chuẩn (`RUBRIC`) mô tả các mức điểm khá rộng. Với các câu trả lời "tốt vừa phải" — không hoàn hảo nhưng cũng không sai — cả hai judge đều cho điểm giống nhau (thường là 4/5) vì ngưỡng mô tả của 4 và 5 không đủ rõ ràng để tạo ra sự khác biệt quan điểm.

**Giải pháp:** Thiết kế `RUBRIC_CONSERVATIVE` với ngưỡng nghiêm khắc hơn cho từng tiêu chí — đặc biệt là `faithfulness` (yêu cầu 100% truy nguyên để đạt 5/5) và `relevancy` (yêu cầu trả lời trực tiếp ngay câu đầu). Judge với rubric này tự nhiên cho điểm thấp hơn 1 bậc trên các case borderline.

**Bài học:** Agreement rate = 100% không phải tín hiệu tốt — có thể là dấu hiệu của rubric quá chung chung. Judge tốt cần có quan điểm *khác nhau* trên các case mơ hồ để phát hiện điểm cần cải thiện.

---

## 4. Trade-off: Độ chính xác vs Tốc độ trong Regression Gate

Thiết kế release gate là bài toán trade-off nhiều chiều:

| Quyết định thiết kế | Quá chặt | Quá lỏng |
|---|---|---|
| Ngưỡng quality gate | Không bao giờ release | Release cả version tệ |
| Tolerance score_drop | Mọi improvement bị block | Quality regression được bỏ qua |
| Concurrency | Chậm, an toàn | Nhanh, dễ rate-limit |
| Số dimensions | Đơn giản nhưng bỏ sót | Phức tạp, khó debug khi fail |

**Triết lý tôi chọn:** 3 gates độc lập, fail-fast — nếu quality regression thì **ROLLBACK** ngay, không cần check cost/perf. Vì quality regression là rủi ro duy nhất không thể chấp nhận trong production HR chatbot (nhân viên có thể nhận thông tin sai về lương, nghỉ phép, v.v.).

---

## 5. Điều tôi học được từ lab này

1. **DevOps trong AI ≠ DevOps truyền thống.** Không có "unit test pass/fail" rõ ràng — phải thiết kế multi-axis gate với tolerance cho phép noise thống kê giữa các lần chạy.

2. **Guardrails nên là lớp độc lập, không nhúng vào agent logic.** Wrapper pattern cho phép bật/tắt từng guardrail khi benchmark, giúp isolate nguyên nhân khi kết quả thay đổi.

3. **Failure analysis → cải tiến có căn cứ, không phải đoán mò.** Ba cải tiến của `OptimisedAgentV2` đều xuất phát từ root cause cụ thể trong `failure_analysis.md`, không phải "thêm cho có".

4. **Regression là phòng tuyến cuối cùng.** Dù code review, unit test, integration test tốt đến đâu, chỉ benchmark thực trên golden dataset mới phát hiện được quality regression của LLM output — vì LLM không deterministic.

5. **Hallucination Rate là metric quan trọng nhất cho HR chatbot.** Một chatbot đưa ra thông tin lương/chế độ sai có thể gây tranh chấp pháp lý. Đây là nơi guardrail (faithfulness threshold) quan trọng hơn accuracy tuyệt đối.

6. **Rubric design là engineering, không phải art.** Mô tả mức điểm mơ hồ dẫn đến judge không phân biệt được "tốt" và "tốt vừa phải". Cần đầu tư thiết kế rubric rõ ràng, có ví dụ cụ thể cho từng mức — tương tự như viết unit test: test tốt phải có assertion rõ ràng.

---

## 6. Nếu làm lại, tôi sẽ...

- **Thiết kế schema output đầy đủ ngay từ lần chạy đầu tiên** — không để mất fields như `accuracy`, `tone`, `safety` trong benchmark cũ.
- **Implement Shadow Mode** trước khi chạy full benchmark: V2 chạy song song với V1 nhưng output không được dùng, chỉ để so sánh — tránh tốn chi phí chạy lại khi V2 fail gate.
- **Thêm `--version` flag vào main.py** để chạy riêng V1 hoặc V2 khi debug, không phải chạy cả hai mỗi lần.
- **Lưu checkpoint sau mỗi 10 cases** — nếu pipeline crash ở case 45, không mất toàn bộ kết quả và phải chạy lại từ đầu.
- **Viết integration test cho guardrail patterns** ngay từ đầu, không test thủ công bằng unicode string sau.
- **Calibrate rubric trên tập nhỏ trước khi chạy benchmark thật** — chạy thử 5–10 cases, kiểm tra phân phối điểm, đảm bảo không bị ceiling effect (toàn 5/5) hay floor effect (toàn 1/5) trước khi chạy toàn bộ 60 cases tốn chi phí.

---

*Trương Đăng Nghĩa — P5 DevOps Engineer*  
*Nhóm 05 — E402 — Lab Day 14*
