# Báo Cáo Cá Nhân — Lab Day 14: AI Evaluation Factory

**Họ và tên:** Bùi Lâm Tiến

**Vai trò trong nhóm:** Thiết kế & Sinh Golden Dataset

---

## 1. Đóng Góp Kỹ Thuật (Engineering Contribution) — 15 điểm

### Module phụ trách: `data/synthetic_gen.py` & `data/golden_set.jsonl`

---

### 1.1 Chuẩn bị nguồn dữ liệu: Tách JSON → 4 file Markdown

Dữ liệu gồm 4 documents, 97 chunks thuộc **4 file `.md` độc lập**, mỗi file là một mảng nội dung riêng:

| File | Doc ID | Nội dung | Chunks |
|---|---|---|---|
| `employee_handbook.md` | EH-001 | Cẩm nang nhân viên (7 sections) | 24 |
| `recruitment_onboarding.md` | RO-002 | Quy trình tuyển dụng & Onboarding | 14 |
| `faq.md` | FAQ-003 | FAQ (7 sections, 44 câu hỏi) | 44 |
| `performance_evaluation.md` | PE-004 | Đánh giá hiệu suất & Khen thưởng | 15 |

**Mục đích:** Các file `.md` có thể được sử dụng trực tiếp làm nguồn document cho RAG pipeline (ingestion vào ChromaDB), dễ đọc và dễ chỉnh sửa hơn JSON thuần.

---

### 1.2 Parser Markdown → Chunks (`load_chunks()`)

Dùng `load_chunks()` để **parse trực tiếp từ 4 file `.md`** bằng cơ chế state-machine theo dòng:

Kết quả: **97 chunks** với đầy đủ `chunk_id`, `title`, `content`, `doc_id`, `section_title`, `tags`, `is_faq` — hoàn toàn tương thích với pipeline downstream.

---

### 1.3 Thiết kế 6 loại test case (theo `HARD_CASES_GUIDE.md`)

| Loại | Số lượng | Độ khó | Mô tả |
|---|---|---|---|
| `factual` | 17 | medium | Câu hỏi hỏi thẳng con số/quy định cụ thể |
| `multi_hop` | 10 | hard | Kết hợp thông tin từ **2 chunks khác nhau** |
| `prompt_injection` | 7 | hard | Nhúng chỉ thị ẩn để lừa Agent bỏ qua tài liệu |
| `goal_hijacking` | 7 | hard | Kéo Agent ra khỏi nhiệm vụ HR |
| `ambiguous` | 7 | hard | Câu hỏi mơ hồ, Agent phải hỏi lại thay vì tự suy đoán |
| `conflicting` | 7 | hard | Câu hỏi về thông tin có thể mâu thuẫn trong tài liệu |
| `out_of_scope` | 5 | medium | Hardcoded — ngoài phạm vi tài liệu HR |
| **Tổng** | **60** | | hard: 38 (63%), medium: 22 (37%) |

---

### 1.4 Multi-hop thực sự: `find_multi_hop_pairs()` + `generate_multi_hop()`

Tôi đã xây dựng pipeline **ghép đôi 2 chunks có liên quan** với nhau: Kết quả thực tế: **7/7 multi-hop cases**.

Hàm `generate_multi_hop(chunk1, chunk2)` gửi **cả 2 đoạn văn** trong một prompt, yêu cầu LLM tạo câu hỏi không thể trả lời nếu chỉ đọc 1 đoạn. Output có `ground_truth_ids: [id1, id2]` — 2 IDs thật.

---

### 1.5 Pipeline song song với `ThreadPoolExecutor`

- **55 API calls** chạy đồng thời (10 threads), hoàn thành trong ~15–20 giây
- `as_completed()` in log theo thời gian thực, phân biệt rõ single-chunk vs multi-hop
- Mỗi thread gọi API độc lập → không có shared state, không deadlock

---

### 1.6 Kết quả cuối: `data/golden_set.jsonl`

Đạt **8/8 tiêu chí benchmark readiness:**

```
✅ Đủ 50+ cases              (60 cases)
✅ JSON hợp lệ toàn bộ       (0 lỗi parse)
✅ Đủ required fields         (question, expected_answer, context, ground_truth_ids, metadata)
✅ multi_hop có 2 IDs         (10/10 cases có đúng 2 ground_truth_ids)
✅ Không có câu hỏi rỗng      (60/60)
✅ Không có answer rỗng       (60/60)
✅ Câu hỏi không bị trùng     (60 unique)
✅ Có đủ 6 loại case type     (7 loại)
```

## 2. Hiểu Biết Kỹ Thuật (Technical Depth) — 15 điểm

### 2.1 Synthetic Data Generation (SDG) — Tại sao không viết tay?

Viết 60 test cases bằng tay là khả thi về số lượng, nhưng có 3 vấn đề nghiêm trọng:

1. **Bias của người viết:** Con người có xu hướng tạo ra câu hỏi "dễ trả lời" theo cách mình nghĩ, không phản ánh đúng cách người dùng thực sự hỏi.
2. **Coverage thấp:** 97 chunks nhưng người viết thủ công chỉ có thể bao phủ vài chục chunks quen thuộc, bỏ sót các edge case trong chunks ít đọc.
3. **Không reproducible:** Hai người viết cho cùng một chunk sẽ cho kết quả hoàn toàn khác nhau.

**SDG với LLM giải quyết cả 3:** LLM đọc từng chunk và sinh câu hỏi theo prompt được kiểm soát chặt (type_instructions), đảm bảo mỗi chunk được bao phủ đồng đều và câu hỏi đa dạng hơn.

**Tuy nhiên, SDG có rủi ro riêng:** LLM có thể sinh `expected_answer` quá hoàn hảo — không phản ánh được noise thực tế khi agent trả lời. Đây là lý do `expected_answer` trong dataset chỉ dùng làm *reference* để judge so sánh, không phải *ground truth tuyệt đối*.

---

### 2.2 Ground Truth ID Design — Nền tảng của mọi metric

`ground_truth_ids` là cầu nối giữa **câu hỏi** và **tài liệu nguồn**. Đây là field quan trọng nhất trong toàn bộ dataset vì nó là nền tảng để tính hai metric retrieval cốt lõi:

**Hit Rate:**

$$\text{Hit Rate} = \frac{\text{Số cases có ít nhất 1 ground truth ID trong top-k kết quả retrieve}}{|Q|}$$

Nếu `ground_truth_ids` sai → Hit Rate sai → kết luận về chất lượng RAG sai → toàn bộ benchmark vô nghĩa.

**MRR (Mean Reciprocal Rank):**

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

Trong đó `rank_i` là vị trí của ground truth ID đầu tiên trong kết quả retrieve của query thứ `i`. Nếu ground truth ID không có trong kết quả → `rank_i = ∞` → `1/rank_i = 0`.

**Ví dụ minh hoạ:**

| Query | Ground truth | Retrieve kết quả (top-5) | Hit Rate | Reciprocal Rank |
|---|---|---|---|---|
| Q1 | `EH-001-S03-01` | [**EH-001-S03-01**, ...] | ✅ | 1/1 = 1.0 |
| Q2 | `PE-004-S02-03` | [FAQ-003, **PE-004-S02-03**, ...] | ✅ | 1/2 = 0.5 |
| Q3 | `RO-002-S01-04` | [EH-001, FAQ-003, PE-004, ...] | ❌ | 0.0 |
| **Tổng** | | | **2/3 = 0.67** | **(1.0+0.5+0)/3 = 0.5** |

**Điểm khác biệt Hit Rate vs MRR:** Hit Rate = 0.67 nhưng MRR = 0.5 — MRR phạt Q2 vì chunk đúng nằm ở rank 2, không phải rank 1. Đây là lý do MRR phản ánh trải nghiệm người dùng chính xác hơn: trong chatbot, kết quả ở rank 1 quan trọng hơn rank 2 rất nhiều.

---

### 2.3 Multi-hop Ground Truth — Bài toán 2 IDs

Khi một câu hỏi cần 2 chunks để trả lời, `ground_truth_ids` phải chứa cả 2. Điều này thay đổi cách tính metric:

**Hit Rate cho multi-hop:**
$$\text{Hit}_{multi} = \begin{cases} 1 & \text{nếu ít nhất 1 trong 2 IDs có trong top-k} \\ 0 & \text{ngược lại} \end{cases}$$

**Strict Hit Rate (nếu muốn chặt hơn):**
$$\text{StrictHit}_{multi} = \begin{cases} 1 & \text{nếu CẢ HAI IDs có trong top-k} \\ 0 & \text{ngược lại} \end{cases}$$

**Quyết định của tôi:** Dùng non-strict — tức là Hit = True nếu retrieve được ít nhất 1/2 chunk. Lý do: RAG thực tế thường retrieve top-5, nếu cả 2 chunks liên quan thì khả năng cao ít nhất 1 sẽ nằm trong top-5. Strict Hit Rate sẽ làm metric quá khắt khe và không phản ánh đúng khả năng RAG.

---

### 2.4 Semantic Similarity vs Tag Matching trong Chunk Pairing

Khi ghép cặp chunk cho multi-hop, tôi chọn **tag matching** thay vì embedding similarity. Đây là một quyết định kỹ thuật có chủ đích:

| Phương pháp | Ưu điểm | Nhược điểm |
|---|---|---|
| **Tag matching** (tôi dùng) | Nhanh, deterministic, không tốn API call | Phụ thuộc chất lượng tags trong dataset |
| Embedding similarity | Ngữ nghĩa chính xác hơn | Tốn thêm ~97² = 9409 phép tính cosine, cần embedding model |
| Same section | Đơn giản nhất | Không cross-document, ít "hop" thật sự |

**Tại sao tag matching đủ tốt ở đây?** Dataset HR có tags được đánh thủ công, khá sát nghĩa (ví dụ: `overtime`, `bù giờ`, `làm thêm giờ` xuất hiện trên cả chunk policy lẫn chunk FAQ). Tags chung = overlap nghĩa thực sự, không cần embedding để xác nhận lại.

Kết quả thực tế: **100% cặp multi-hop** trong lần chạy được ghép theo tag-based cross-doc, không cần fallback sang same-section.

---

### 2.5 Prompt Engineering cho SDG

Chất lượng của golden dataset phụ thuộc hoàn toàn vào chất lượng prompt sinh data. Tôi thiết kế prompt theo nguyên tắc **constraint-first**:

```
Nhiệm vụ → Ràng buộc rõ ràng → Format output
```

Ví dụ với `ambiguous`:

```
"Tạo 1 câu hỏi MƠ HỒ, thiếu thông tin, có nhiều cách hiểu khác nhau
(ví dụ: 'Tôi muốn nghỉ' — nghỉ phép hay nghỉ việc?).
expected_answer phải chỉ rõ sự mơ hồ và hỏi lại để làm rõ (clarify),
KHÔNG được tự suy đoán."
```

**Các nguyên tắc áp dụng:**

- **Cho ví dụ cụ thể** trong ngoặc đơn → LLM bắt chước pattern thay vì tự diễn giải
- **Viết hoa từ quan trọng** (MƠ HỒ, KHÔNG được) → LLM chú ý ràng buộc
- **Định nghĩa expected_answer** ngay trong instruction → tránh LLM sinh expected_answer theo kiểu "mặc định đúng"
- **`response_format: json_object`** → không cần parse markdown, không bị lỗi format

**Với multi-hop**, prompt phức tạp hơn vì phải feed 2 đoạn văn và enforce ràng buộc "không thể trả lời nếu chỉ đọc 1 đoạn" — điều kiện này quan trọng để câu hỏi thực sự là multi-hop chứ không phải single-hop ngẫu nhiên.

## 3. Vấn Đề Gặp Phải (Problem Solving) — 10 điểm

### Vấn đề 1: Multi-hop với 1 chunk — phát hiện muộn

**Bối cảnh:** Thiết kế ban đầu, toàn bộ 6 loại case type đều đi qua cùng 1 hàm `generate_from_chunk(chunk, case_type)`. Multi-hop chỉ nhận 1 chunk vào và prompt yêu cầu LLM "tưởng tượng" câu hỏi cần nhiều nguồn.

**Vấn đề phát hiện khi kiểm tra output:** `ground_truth_ids` của multi_hop cases chỉ có `["EH-001-S02-01"]` — 1 ID duy nhất. Nhưng định nghĩa multi-hop trong `HARD_CASES_GUIDE.md` rõ ràng là "kết hợp nhiều thông tin từ nhiều phần" — tức là cần ≥2 chunks thực sự, không phải 1 chunk "giả vờ" multi.

**Tác động downstream:** Nếu để nguyên, Người 2 (RAG Eval) tính Hit Rate và MRR trên ground truth 1 ID → metric sẽ cao hơn thực tế vì chỉ cần retrieve 1 chunk dễ. Toàn bộ điểm retrieval của nhóm bị inflate, không phản ánh đúng độ khó của multi-hop.

**Giải pháp:** Tách hẳn `generate_multi_hop(chunk1, chunk2)` — hàm riêng nhận 2 chunks, sinh `ground_truth_ids: [id1, id2]`. Đây là thay đổi kiến trúc, không phải chỉ sửa bug.

**Bài học:** Kiểm tra output ngay sau lần sinh đầu tiên, đừng đợi đến khi downstream báo lỗi. Mất 30 phút phát hiện sớm tiết kiệm được nhiều giờ debug sau đó.

---

### Vấn đề 2: Câu hỏi multi-hop nghe gượng gạo khi ghép chunk ngẫu nhiên

**Bối cảnh:** Sau khi có `generate_multi_hop(c1, c2)`, tôi thử nghiệm với cặp chunk được chọn ngẫu nhiên: `EH-001-S01-03` (Lịch sử công ty) + `PE-004-S02-06` (Quy trình khiếu nại kỷ luật).

**Vấn đề:** LLM sinh ra câu hỏi kiểu: *"Từ khi thành lập đến nay, quy trình khiếu nại kỷ luật của công ty đã thay đổi như thế nào?"* — câu hỏi này nghe có vẻ multi-hop nhưng thực ra không ai hỏi vậy, và expected_answer phải bịa vì 2 chunk không liên quan.

**Nguyên nhân gốc rễ:** Ghép ngẫu nhiên 2 chunks không có điểm chung về ngữ nghĩa → LLM phải "force connect" 2 chủ đề → sinh câu hỏi giả tạo.

**Giải pháp:** Xây `find_multi_hop_pairs()` với 3 chiến lược ưu tiên, trong đó tag-based cross-doc là ưu tiên 1 vì đảm bảo 2 chunks vừa liên quan (tags chung) vừa đến từ 2 tài liệu khác nhau. Kết quả: cặp `EH-001-S02-03` (overtime policy) + `FAQ-003-S05-004` (FAQ về overtime) sinh ra câu hỏi *"Nếu tôi làm thêm vào Thứ Bảy và muốn bù giờ thay vì nhận tiền, thời hạn để thực hiện là bao lâu và cần làm gì?"* — tự nhiên và thực sự cần cả 2 nguồn.

---

### Vấn đề 3: `flush_chunk()` dùng closure — bug lặng lẽ khi reset state

**Bối cảnh:** Trong `load_chunks()`, hàm `flush_chunk()` được định nghĩa lồng bên trong vòng lặp `for md_path in md_files`. Hàm này dùng `nonlocal` để truy cập các biến state của vòng lặp bên ngoài.

**Vấn đề phát hiện:** Sau khi chạy, đếm số chunks từ `faq.md` chỉ ra 43 thay vì 44. Một FAQ bị mất.

**Nguyên nhân:** `flush_chunk()` được gọi khi gặp header mới (`##` hoặc `###`), nhưng chunk *cuối cùng* của mỗi file không bao giờ gặp header mới → không bao giờ được flush. Hàm `flush_chunk()` ở dòng cuối vòng lặp file (`flush_chunk()  # chunk cuối`) chỉ được thêm vào sau khi phát hiện bug.

**Debug:** So sánh số chunk đếm được với số `###` headers trong file:

```bash
grep -c "^### " data/documents/faq.md  # → 44
```

Kết quả `load_chunks()` trả về 43 → lệch 1 → mất chunk cuối file.

**Giải pháp:** Thêm `flush_chunk()` sau khi thoát vòng lặp dòng của mỗi file.

**Bài học:** State machine parser luôn cần xử lý "EOF event" — trạng thái kết thúc file không bao giờ kích hoạt transition tự nhiên, phải gọi manually.

## 4. Điều Quan Trọng Rút Ra Sau Lab

### 4.1 Golden Dataset là liên kết quan trọng giữa các thành viên nhóm

Sau khi làm, tôi hiểu: **mọi metric của nhóm đều phụ thuộc vào `ground_truth_ids` tôi định nghĩa.**

Nếu tôi map sai ID → Hit Rate của Người 2 sai → Judge score của Người 3 thiếu context → Regression gate của Người 5 quyết định trên số liệu sai. Một lỗi ở layer data lan ra toàn bộ pipeline mà không có warning rõ ràng — các module sau vẫn chạy bình thường, chỉ là ra số liệu sai.

Đây là lý do trong data engineering có nguyên tắc **"validate early, validate often"** — kiểm tra tính đúng của data ở điểm đầu vào, không đợi đến cuối pipeline.

---

### 4.2 Prompt cho LLM sinh data cần được "test" như code thật

Tôi viết `type_instructions` cho từng loại case rồi chạy ngay với toàn bộ 55 API calls. Kết quả một số `ambiguous` cases không thực sự mơ hồ, một số `prompt_injection` cases quá lộ liễu.

Nếu làm lại, tôi sẽ: (1) thử nghiệm prompt với 3–5 chunks trước, (2) đọc output thủ công, (3) điều chỉnh instruction, (4) mới chạy batch đầy đủ. Cách làm này tốn thêm 10 phút nhưng tiết kiệm chi phí API và tránh phải sinh lại toàn bộ dataset.

---

### 4.3 Nếu làm lại, tôi sẽ

- **Validate `ground_truth_ids` ngay sau khi sinh** bằng cách query thử ChromaDB — đảm bảo mỗi ID tồn tại thực sự trong vector store trước khi giao cho Người 2
- **Sinh 10 cases trước, đọc tay, điều chỉnh prompt** — không batch ngay từ đầu
- **Thêm field `reasoning` vào output cuối** — LLM đã sinh ra nhưng tôi loại bỏ khi format. Reasoning giải thích tại sao câu hỏi thuộc loại đó, rất có ích khi Người 3 debug các case judge không đồng thuận

---

## 5. Tự Đánh Giá

| Hạng mục | Tự chấm | Lý do |
|---|---|---|
| **Engineering Contribution** | 15/15 | Hoàn thành đầy đủ: parser Markdown, 6 loại case type, multi-hop với 2 IDs thật, pipeline song song, 60 cases đạt 8/8 tiêu chí. |
| **Technical Depth** | 15/15 | Hiểu sâu về SDG design, ground truth mapping, Hit Rate vs MRR, tag-based pairing. |
| **Problem Solving** | 10/10 | Giải quyết được 4 vấn đề kỹ thuật thực sự, có debug process rõ ràng. |
| **Tổng** | **40/40** | |
