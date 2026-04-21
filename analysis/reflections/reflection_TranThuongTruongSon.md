# Báo cáo Cá nhân — Người 2: Retrieval Engineer: Hit Rate & MRR

**Họ và tên:** Trần Thương Trường Sơn
**Vai trò:** Người 2 — Retrieval Engineer: Hit Rate & MRR
**File phụ trách:** `engine/retrieval_eval.py`

---

## 1. Đóng góp cụ thể (Engineering Contribution)

> Mô tả chính xác bạn đã implement phần nào — càng cụ thể càng tốt.

- Implement `_load_chunks_from_dataset()`: đọc và parse toàn bộ subsection + FAQ chunks từ `data/documents/hr_rag_dataset.json`, chuẩn hóa thành list dict với `chunk_id`, `text`, `doc_id`, `title`
- Implement `build_vector_db()`: khởi tạo ChromaDB `PersistentClient`, tạo/nạp collection `hr_chunks` với embedding function `text-embedding-3-small` (cosine similarity), upsert theo batch 100 để tránh vượt giới hạn API, chỉ index lại khi collection rỗng
- Implement `query_vector_db()`: truy vấn ChromaDB với `query_texts`, trả về danh sách `chunk_id` theo thứ tự độ tương đồng giảm dần
- Implement `RetrievalEvaluator.calculate_hit_rate()`: tính Hit Rate@K — trả về 1.0 nếu ít nhất 1 `expected_id` xuất hiện trong top-K kết quả, ngược lại 0.0
- Implement `RetrievalEvaluator.calculate_mrr()`: tính Reciprocal Rank — trả về `1/rank` của expected_id đầu tiên tìm thấy (1-indexed), trả về 0.0 nếu không có hit
- Implement `RetrievalEvaluator.evaluate_batch()`: chạy toàn bộ golden dataset, bỏ qua out-of-scope cases (không có `ground_truth_ids`), tổng hợp avg Hit Rate và MRR, trả về kết quả chi tiết từng case
- Thiết kế lazy-load property `collection` để chỉ kết nối VectorDB khi thực sự cần truy vấn

---

## 2. Kết quả đạt được

> Benchmark thực tế với 60 cases trên `reports/summary.json` (timestamp: 2026-04-21 19:21:31)

| Chỉ số | Kết quả |
|---|---|
| Hit Rate@3 trung bình | **0.75** (75%) |
| MRR trung bình | **0.6306** |
| Số cases có hit | **45 / 60** |
| Retrieval accuracy | 0.75 |
| Top-K sử dụng | 3 |

---

## 3. Phân tích kết quả

### 3.1 Khi nào retrieval bị miss?

Dựa trên phân tích benchmark_results.json (60 cases), 15 cases có hit_rate = 0.0:

- **Out-of-scope questions** (viết thơ, câu hỏi về sản phẩm công ty, lịch nghỉ Tết cụ thể...): hit_rate = 0.0 là **đúng** — không có ground truth chunk nào trong HR database. Đây không phải lỗi retrieval mà là câu hỏi ngoài phạm vi.
- **Câu hỏi hỏi thông tin rất cụ thể theo ngày/năm** (ví dụ: "lịch nghỉ Tết Nguyên Đán năm 2025"): retrieval không tìm thấy vì chunk không có dữ liệu thời gian cụ thể này.
- **Câu hỏi mơ hồ hoặc mang tính triết học về HR** (cân bằng đổi mới sáng tạo và trách nhiệm cá nhân): embedding tìm được chunk liên quan nhưng không đủ cụ thể → MRR = 0.5 thay vì 1.0.
- Tiêu chí hay bị ảnh hưởng nhất: **faithfulness** — các case hit_rate = 0.0 mà agent vẫn trả lời thường có faithfulness thấp (2.5–3.0) vì agent không có chunk để bám vào.

### 3.2 Ý nghĩa của MRR đạt được

MRR = **0.6306** → chunk đúng trung bình xuất hiện ở vị trí rank ~1.59 trong top-K (tức là thường ở rank 1 hoặc 2).

- Hit Rate = 0.75 nhưng MRR = 0.63 → khoảng cách ~0.12 cho thấy trong ~12% cases, retrieval **tìm thấy chunk đúng nhưng không xếp nó ở vị trí #1** (rank 2 hoặc 3). LLM vẫn nhận được context đúng nhưng không ở vị trí ưu tiên trong prompt.
- Đây là dấu hiệu chunking có thể quá lớn (embedding bị "pha loãng") — nhận xét này cũng được Kiên (P4) ghi nhận trong báo cáo của anh ấy.
- Để cải thiện MRR: có thể thử **smaller chunk size** hoặc **re-ranking** sau bước retrieval ban đầu.

---

## 4. Vấn đề khó nhất và cách giải quyết

> Mô tả 1–2 vấn đề thực tế gặp phải khi code.

**Vấn đề 1:** ChromaDB index lại toàn bộ mỗi lần chạy → tốn thời gian và API cost embedding
- **Giải pháp:** Dùng `PersistentClient` lưu data xuống đĩa tại `.chromadb/`, kiểm tra `collection.count() == 0` trước khi upsert — chỉ index lần đầu, những lần sau load lại ngay lập tức

**Vấn đề 2:** Upsert quá nhiều documents một lúc dễ bị timeout hoặc lỗi rate limit từ OpenAI Embeddings API
- **Giải pháp:** Chia thành batch 100 documents mỗi lần upsert bằng vòng lặp `range(0, len(ids), batch_size)`

**Vấn đề 3:** Out-of-scope cases trong golden dataset không có `ground_truth_ids` → nếu tính vào sẽ kéo thấp Hit Rate một cách không công bằng
- **Giải pháp:** Skip các case có `ground_truth_ids = []`, ghi rõ `"skipped": True` trong `per_case` để debug, chỉ tính average trên các case thực sự có ground truth

---

## 5. Hiểu biết kỹ thuật

### Hit Rate@K là gì?
Hit Rate@K đo xem retrieval stage có tìm được ít nhất 1 chunk đúng trong top-K kết quả hay không.

- **Hit Rate = 1.0**: ít nhất 1 chunk đúng xuất hiện trong K kết quả đầu tiên
- **Hit Rate = 0.0**: không có chunk đúng nào trong top-K
- **Avg Hit Rate** trên toàn dataset = % câu hỏi mà retrieval "không bỏ sót" hoàn toàn

Hit Rate@K là điều kiện cần tối thiểu cho RAG pipeline — nếu retrieval miss thì LLM không thể trả lời đúng dù prompt có tốt đến đâu.

### MRR (Mean Reciprocal Rank) là gì?
MRR đo **vị trí trung bình** của chunk đúng đầu tiên trong danh sách kết quả.

Công thức: `MRR = (1/N) * Σ (1 / rank_i)`
- `rank_i` = vị trí (1-indexed) của chunk đúng đầu tiên trong kết quả query thứ i
- Nếu chunk đúng không xuất hiện → `1/rank = 0`

| MRR | Ý nghĩa |
|-----|---------|
| 1.0 | Chunk đúng luôn ở vị trí #1 |
| 0.5 | Chunk đúng trung bình ở vị trí #2 |
| 0.33 | Chunk đúng trung bình ở vị trí #3 |
| < 0.33 | Retrieval kém, chunk đúng thường ở cuối hoặc không có |

### Tại sao dùng cosine similarity thay vì L2?
Cosine similarity đo **góc** giữa 2 vector embedding, không bị ảnh hưởng bởi độ dài văn bản. Phù hợp hơn với text embeddings vì câu ngắn và câu dài nói về cùng chủ đề vẫn có cosine cao. L2 (Euclidean) bị ảnh hưởng bởi magnitude → chunk dài sẽ có lợi thế không công bằng.

### Trade-off Top-K: Hit Rate vs Precision
- **Top-K lớn hơn** → Hit Rate cao hơn (dễ tìm được chunk đúng hơn) nhưng LLM phải xử lý nhiều context hơn → tốn token, tăng noise
- **Top-K = 3** là sweet spot thực tế: đủ để bắt được chunk đúng mà không làm prompt quá dài
- Nên đánh giá cả Hit Rate@1, @3, @5 để hiểu rõ hơn về chất lượng ranking

---

## 6. Điều rút ra sau lab

Retrieval là bottleneck thầm lặng của RAG — Hit Rate 75% đồng nghĩa với 25% câu hỏi (15/60 cases) retrieval không tìm được chunk liên quan, và LLM không thể cứu vãn nếu không có context. MRR = 0.63 cho thấy chunk đúng thường ở rank 1–2, nhưng vẫn còn ~12% cases cần cải thiện ranking. Việc tách đánh giá retrieval stage riêng (Hit Rate, MRR) khỏi generation stage giúp debug chính xác: khi answer sai, biết ngay là do retrieval miss hay do LLM hallucinate từ context đúng. Sử dụng Vector DB persistent (không index lại mỗi lần) là điều bắt buộc trong thực tế — toàn bộ 60 cases chạy được trong 102 giây một phần là nhờ ChromaDB không phải re-embed từ đầu.
