# 📝 Reflection Cá nhân — Nông Trung Kiên (P4 — Backend / Async Runner)

**Lab:** Day 14 — AI Evaluation Factory  
**Module phụ trách:** `engine/runner.py`  
**Ngày:** 2026-04-21

---

## 1. Tôi đã làm gì trong lab này?

Tôi chịu trách nhiệm xây dựng **Async Runner & Performance Engine** — module trung tâm điều phối toàn bộ pipeline đánh giá AI. Công việc cụ thể:

| Hạng mục | Chi tiết |
|---|---|
| **Async Pipeline** | Dùng `asyncio.Semaphore(10)` để chạy 50 cases song song, tránh Rate Limit |
| **Cost & Token Tracking** | Theo dõi prompt/completion tokens, tính USD/case theo bảng giá OpenAI/Claude |
| **Cost Report** | Tổng hợp chi phí theo model, theo bước (agent/judge), throughput P95 |
| **30% Cost Reduction** | Đề xuất 3 chiến lược: Tiered Judging, Prompt Caching, Batch API |
| **Progress Bar** | Tích hợp `tqdm.asyncio` để hiển thị tiến độ real-time |

---

## 2. Giải thích các khái niệm kỹ thuật quan trọng

### 2.1 MRR — Mean Reciprocal Rank

MRR đo lường **vị trí trung bình** của chunk đúng đầu tiên trong danh sách kết quả retrieval.

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

**Ví dụ thực tế:**
- Query A: chunk đúng ở vị trí 1 → Reciprocal Rank = 1/1 = **1.0**
- Query B: chunk đúng ở vị trí 3 → Reciprocal Rank = 1/3 = **0.33**
- Query C: không tìm thấy → Reciprocal Rank = **0.0**
- MRR = (1.0 + 0.33 + 0.0) / 3 = **0.44**

**Tại sao quan trọng?** MRR phạt nặng hơn Hit Rate khi chunk đúng nằm ở vị trí thấp. Nếu hệ thống trả về đúng nhưng ở rank 10, Hit Rate = 1.0 nhưng MRR chỉ = 0.1. MRR phản ánh chính xác hơn trải nghiệm người dùng thực tế.

**Bài học từ lab:** Hit Rate của chúng tôi khá cao nhưng MRR thấp hơn đáng kể — cho thấy retrieval tìm đúng tài liệu nhưng không đặt nó lên đầu. Nguyên nhân: chunking quá lớn khiến embedding bị "pha loãng" thông tin quan trọng.

---

### 2.2 Cohen's Kappa — Hệ số đồng thuận

Cohen's Kappa (κ) đo mức độ đồng ý giữa hai judge **vượt ra ngoài ngẫu nhiên**:

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

Trong đó:
- $P_o$ = tỉ lệ đồng thuận thực tế (Observed Agreement)
- $P_e$ = tỉ lệ đồng thuận kỳ vọng nếu hai judge chọn ngẫu nhiên (Expected Agreement)

**Thang đánh giá:**
| κ | Mức độ đồng thuận |
|---|---|
| < 0.20 | Kém (Poor) |
| 0.21–0.40 | Nhẹ (Fair) |
| 0.41–0.60 | Vừa phải (Moderate) |
| 0.61–0.80 | Tốt (Substantial) |
| > 0.80 | Gần hoàn hảo (Almost Perfect) |

**Tại sao không chỉ dùng Agreement Rate?** Agreement Rate = số case đồng ý / tổng cases. Nhưng nếu cả hai judge đều có xu hướng cho điểm cao (5/5 thường xuyên), Agreement Rate cao nhưng κ thấp — vì sự đồng ý đó chỉ là do bias, không phải vì judge thực sự tin cậy.

**Trong hệ thống của chúng tôi:** κ > 0.6 là mục tiêu để hệ thống đáng tin. Khi κ < 0.4, cần xem lại rubric hoặc nhiệt độ (temperature) của model judge.

---

### 2.3 Position Bias — Lỗi thiên vị vị trí

Position Bias xảy ra khi LLM judge **cho điểm cao hơn câu trả lời được đặt ở vị trí đầu** (A trước B) so với khi đặt ở cuối (B trước A), mặc dù nội dung không thay đổi.

**Cách kiểm tra (implemented trong llm_judge.py):**
1. Đánh giá cặp (response_A, response_B) → score A = 4, B = 3
2. Đổi thứ tự: (response_B, response_A) → nếu bây giờ B = 4, A = 3 thì có bias
3. Tính Position Bias Rate = số lần điểm thay đổi khi đổi vị trí / tổng cases

**Giải pháp đề xuất:**
- Chạy mỗi case **2 lần** với thứ tự đảo ngược, lấy điểm trung bình
- Thêm instruction vào system prompt: *"Hãy đánh giá câu trả lời độc lập với thứ tự trình bày"*
- Tăng cost ~2x nhưng độ tin cậy tăng đáng kể

---

## 3. Trade-off: Chi phí vs Chất lượng

Đây là bài toán cốt lõi của AI Engineering. Qua lab này tôi rút ra:

### Ma trận quyết định

| Tình huống | Model tốt nhất | Lý do |
|---|---|---|
| Đánh giá nhanh / CI pipeline | `gpt-4o-mini` | Nhanh, rẻ, đủ tốt cho smoke test |
| Phán quyết cuối (release gate) | `gpt-4o` hoặc `Claude` | Cần độ chính xác cao nhất |
| Production monitoring (1M+ queries) | Tiered: mini → full | Cân bằng cost/quality |
| Offline analysis / weekly report | OpenAI Batch API | 50% discount, latency không quan trọng |

### Chiến lược tiết kiệm 30% tôi đề xuất

**Chiến lược A — Tiered Judging (~40% giảm chi phí judge):**
- Tất cả cases qua `gpt-4o-mini` trước (rẻ: $0.15/1M input)
- Chỉ escalate lên `gpt-4o` ($2.50/1M input) khi score ở vùng 2.5–4.5 (mơ hồ)
- ~60% cases có thể giải quyết ở tầng mini → tiết kiệm ~40% chi phí judge

**Chiến lược B — Prompt Caching (zero-cost):**
- System prompt judge (~300 tokens) giống nhau cho 50 cases
- OpenAI tự động cache prefix → giảm 50% cost phần đó
- Không cần code thêm, chỉ cần đặt system prompt ở đầu message list

**Kết hợp A + B: tiết kiệm ≥ 30% — đạt mục tiêu đề ra.**

---

## 4. Vấn đề gặp phải khi code async pipeline

### 4.1 Rate Limit khi chạy song song quá nhiều

**Vấn đề:** `asyncio.gather()` mặc định spawn tất cả 50 coroutine cùng lúc → OpenAI API trả về `RateLimitError 429`.

**Giải pháp:** Dùng `asyncio.Semaphore(10)` — giới hạn tối đa 10 request đồng thời. Như một "cửa vào" hẹp, coroutine phải chờ khi đã đủ 10 người trong phòng.

```python
semaphore = asyncio.Semaphore(10)
async with semaphore:
    result = await run_single_test(case)
```

### 4.2 Blocking calls trong async context

**Vấn đề:** Một số thư viện (e.g. chromadb sync query) là blocking → làm nghẽn event loop khi gọi từ `async def`.

**Giải pháp:** Wrap trong `asyncio.get_event_loop().run_in_executor()` để đẩy sang thread pool, tránh block event loop.

### 4.3 Tính tổng cost chính xác khi kết quả về không theo thứ tự

**Vấn đề:** `asyncio.gather()` trả kết quả theo thứ tự *submit* (không phải thứ tự hoàn thành), nhưng khi dùng `tqdm` + gather kết hợp cần cẩn thận.

**Giải pháp:** Luôn dùng `asyncio.gather(*tasks)` (giữ thứ tự). Tránh `as_completed()` khi cần correlate với dataset gốc.

---

## 5. Điều tôi học được từ lab này

1. **Evaluation là engineering, không phải nghệ thuật.** Đo đạc chính xác (MRR, κ, cost/token) quan trọng hơn cảm nhận chủ quan về "agent có vẻ tốt".

2. **Cost tracking không phải optional.** Trong production, biết chính xác mỗi eval tốn bao nhiêu USD giúp team quyết định được có nên chạy eval mỗi commit hay chỉ mỗi release.

3. **Semaphore là công cụ cơ bản nhất của async rate management.** Không có nó, async trở thành DDoS chính mình.

4. **Tiered approach là tư duy đúng cho bài toán cost vs quality.** Không phải lúc nào cũng cần model xịn nhất — cần biết khi nào thì cần.

5. **P95 latency quan trọng hơn average.** Average che giấu outliers. Nếu 5% cases mất 30 giây, cả pipeline "trung bình 2 phút" thực ra bao gồm những trường hợp người dùng chờ rất lâu.

---

## 6. Nếu làm lại, tôi sẽ...

- **Implement retry với exponential backoff** ngay từ đầu, không để pipeline fail khi API timeout
- **Viết unit test cho `CostBreakdown.cost_usd`** trước khi tích hợp — lỗi tính tiền là lỗi nghiêm trọng
- **Thêm `--dry-run` flag** để simulate pipeline không tốn API call khi debug
- **Dùng `aiohttp` session với connection pooling** thay vì để mỗi coroutine tự tạo connection

---

*Nông Trung Kiên — P4 Backend Engineer*  
*Nhóm 05 — E402 — Lab Day 14*
