# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark

- **Tổng số cases:** 60
- **Tỉ lệ Pass/Fail:** 51 Pass / 9 Fail (85% / 15%)
- **Điểm RAGAS trung bình:**
    - Faithfulness: **1.00** (hoàn hảo — agent không hallucinate)
    - Relevancy: **1.00** (hoàn hảo — câu trả lời luôn liên quan)
    - Hit Rate: **0.73** (73.3%)
    - MRR: **0.69**
- **Điểm LLM-Judge trung bình:** **4.17 / 5.0**
- **Agreement Rate (Multi-Judge):** 83.3%
- **Cohen's Kappa:** 0.1386 (thấp — 2 judge có khuynh hướng chấm điểm khác nhau đáng kể)

---

## 2. Phân nhóm lỗi (Failure Clustering)

| Nhóm lỗi | Số lượng | Cases | Nguyên nhân gốc rễ |
|---|---|---|---|
| **Missing Clarification** | 4 | #2, #41, #50, #54 | Agent trả lời thẳng câu hỏi mơ hồ thay vì hỏi làm rõ ngữ cảnh như Ground Truth yêu cầu |
| **Out-of-Scope Handling Inconsistency** | 2 | #12, #13 | Agent không nhất quán: đôi khi viết thơ (sai), đôi khi từ chối đúng (cases 17, 30, 55) |
| **Factual Error** | 2 | #43, #56 | Agent trả lời sai thông tin cụ thể: thời gian quét thẻ (15 phút ≠ 5 phút), ảnh hưởng ngày phép khi đi muộn |
| **Wrong Routing / Wrong Department** | 1 | #60 | Retrieval miss (hit_rate=0) → agent hướng dẫn liên hệ HR thay vì bộ phận Tài chính |

### Tổng hợp thêm về Judge Conflict
- **8 cases có conflict** giữa 2 judge (GPT-4o-mini vs Gemini-2.5-flash)
- GPT-4o-mini nhất quán chấm cao hơn (+1 đến +3 điểm) so với Gemini-2.5-flash
- Gemini-2.5-flash nghiêm khắc hơn trong việc đánh giá "tuân thủ Ground Truth"
- Cohen's Kappa = **0.1386** → mức đồng thuận yếu (Slight Agreement), cho thấy 2 judge đang đánh giá theo tiêu chí khác nhau

---

## 3. Phân tích 5 Whys (3 case tệ nhất)

---

### Case #1 — Case 13: Agent viết thơ về sự khởi đầu mới (Score: 1.0/5 — Tệ nhất)

**Symptom:** Agent viết hẳn một bài thơ về "sự khởi đầu mới trong công việc" trong khi Ground Truth yêu cầu agent phải từ chối và hướng dẫn người dùng về các vấn đề HR.

**Why 1:** Tại sao agent viết thơ thay vì từ chối?
→ Vì System Prompt không có quy tắc tường minh cấm việc sáng tác nội dung sáng tạo (thơ, văn xuôi, v.v.).

**Why 2:** Tại sao System Prompt thiếu quy tắc này?
→ Vì khi thiết kế Prompt, nhóm tập trung vào "trả lời đúng về HR" mà không xác định rõ danh sách các loại yêu cầu **không thuộc phạm vi** agent nên từ chối.

**Why 3:** Tại sao không xác định phạm vi ngoài?
→ Vì không có bước "Scope Definition" trong quá trình thiết kế Agent, chỉ có bước "Knowledge Injection" (đưa tài liệu vào).

**Why 4:** Tại sao có sự inconsistent giữa cases thơ (12, 13 fail vs 17, 18, 30 pass)?
→ Vì LLM có tính ngẫu nhiên (temperature > 0) và không có guardrail cứng (rule-based filter) trước khi LLM xử lý request.

**Why 5:** Tại sao không có guardrail?
→ Vì pipeline chưa tích hợp bước **Pre-classification** (phân loại intent) để lọc câu hỏi trước khi đưa vào LLM.

**Root Cause:** Thiếu tầng **Intent Classification + Guardrail** ở đầu pipeline. Agent dựa hoàn toàn vào LLM để quyết định phạm vi, dẫn đến hành vi không nhất quán với các yêu cầu out-of-scope.

---

### Case #2 — Case 56: Sai thông tin thời gian quét thẻ (Score: 2.5/5)

**Symptom:** Agent trả lời nhân viên cần quét thẻ trong vòng **15 phút** trước/sau giờ làm việc, trong khi Ground Truth là **5 phút**.

**Why 1:** Tại sao agent trả lời sai con số 5 phút → 15 phút?
→ Vì tài liệu được retrieve có thể chứa nhiều mốc thời gian khác nhau (15 phút liên quan đến quy định đi muộn, 5 phút là thời gian quét thẻ), LLM nhầm lẫn giữa hai con số.

**Why 2:** Tại sao LLM nhầm lẫn giữa các mốc thời gian trong cùng một tài liệu?
→ Vì Chunking strategy hiện tại dùng **Fixed-size chunking**, khiến thông tin về "quét thẻ (5 phút)" và "đi muộn (15 phút)" nằm trong cùng một chunk → LLM không phân biệt được hai ngữ cảnh.

**Why 3:** Tại sao Fixed-size chunking gây ra vấn đề này?
→ Vì Fixed-size chunking không tôn trọng ranh giới ngữ nghĩa của tài liệu, cắt xuyên qua các đoạn có liên quan đến nhau về mặt ngữ nghĩa nhưng khác về nội dung cụ thể.

**Why 4:** Tại sao không dùng Semantic Chunking?
→ Vì trong giai đoạn Ingestion, nhóm chưa phân tích cấu trúc tài liệu HR (nhiều bảng biểu, danh sách số liệu cụ thể) để chọn chiến lược chunking phù hợp.

**Why 5:** Tại sao không có bước kiểm tra chất lượng sau Ingestion?
→ Vì quy trình thiếu bước **Retrieval Quality Audit** sau khi ingestion, không có test xác minh rằng các con số/mốc thời gian cụ thể được retrieve đúng.

**Root Cause:** **Chunking strategy không phù hợp với tài liệu HR** — tài liệu HR chứa nhiều số liệu định lượng (thời gian, tỷ lệ, số ngày) đặt gần nhau, Fixed-size chunking làm mất ngữ cảnh phân biệt, dẫn đến LLM chọn nhầm con số.

---

### Case #3 — Case 2: Thiếu Clarification cho câu hỏi mơ hồ (Score: 2.0/5)

**Symptom:** Câu hỏi "Tôi có thể sử dụng ngân sách phúc lợi linh hoạt cho những gì cụ thể?" là câu hỏi **mơ hồ** (không rõ loại phúc lợi nào). Ground Truth yêu cầu agent hỏi làm rõ trước, nhưng agent lại trả lời thẳng bằng danh sách đầy đủ các hạng mục.

**Why 1:** Tại sao agent không hỏi làm rõ câu hỏi mơ hồ?
→ Vì System Prompt không có hướng dẫn rõ ràng về khi nào cần hỏi làm rõ (Clarification Policy).

**Why 2:** Tại sao không có Clarification Policy trong Prompt?
→ Vì khi thiết kế Prompt, nhóm chỉ tập trung vào việc "trả lời đầy đủ và chính xác" mà chưa xác định được các dạng câu hỏi mơ hồ cần xử lý đặc biệt.

**Why 3:** Tại sao không xác định được dạng câu hỏi mơ hồ?
→ Vì Golden Dataset giai đoạn đầu không bao gồm các test cases với **ambiguous queries** — chỉ có câu hỏi rõ ràng, dẫn đến thiếu dữ liệu để nhận ra pattern này.

**Why 4:** Tại sao Golden Dataset thiếu ambiguous cases?
→ Vì quy trình SDG (Synthetic Data Generation) không có bước tạo ra "câu hỏi mơ hồ cố tình" — chỉ dùng template tạo câu hỏi rõ ràng từ tài liệu.

**Why 5:** Tại sao SDG không có bước tạo ambiguous cases?
→ Vì nhóm thiếu chiến lược **Adversarial/Edge-case Testing** trong giai đoạn thiết kế dataset — không có role "red teamer" chuyên tạo các câu hỏi khó/mơ hồ.

**Root Cause:** **Thiếu Clarification Policy trong System Prompt** kết hợp với **Golden Dataset không đại diện cho ambiguous queries**. Hậu quả: agent không được "training" (via few-shot examples) để nhận biết và xử lý câu hỏi mơ hồ đúng cách.

---

## 4. Kế hoạch cải tiến (Action Plan)

### 🔴 Ưu tiên Cao (High Priority)

- [x] **Thêm Clarification Policy vào System Prompt**: Định nghĩa rõ ràng các dạng câu hỏi mơ hồ, thêm few-shot examples về cách hỏi làm rõ trước khi trả lời. *(Fix cho Cases #2, #41, #50, #54 — 4 fail cases)*

- [x] **Thêm Intent Classification / Guardrail trước LLM**: Triển khai bước phân loại yêu cầu (HR question / out-of-scope / creative request) bằng rule-based hoặc lightweight classifier. Nếu out-of-scope → redirect cứng, không để LLM tự quyết. *(Fix cho Cases #12, #13 — inconsistent poetry behavior)*

### 🟡 Ưu tiên Trung bình (Medium Priority)

- [ ] **Thay đổi Chunking strategy từ Fixed-size sang Semantic Chunking**: Đặc biệt quan trọng với tài liệu HR chứa nhiều bảng biểu số liệu (thời gian, tỷ lệ, ngày phép). Ưu tiên dùng Markdown-aware chunking hoặc Table-aware chunking. *(Fix cho Cases #56, #43)*

- [ ] **Mở rộng Golden Dataset với Ambiguous Cases**: Bổ sung ít nhất 15 cases dạng câu hỏi mơ hồ/thiếu ngữ cảnh vào Golden Dataset. Thêm bước "Ambiguous Query Generation" vào pipeline SDG.

- [ ] **Cải thiện Knowledge Base Routing**: Thêm metadata tag cho từng chunk (phòng ban phụ trách: HR, Tài chính, Ban lãnh đạo) để agent có thể hướng dẫn đúng bộ phận liên hệ. *(Fix cho Case #60)*

### 🟢 Ưu tiên Thấp (Low Priority)

- [ ] **Calibrate lại Judge Prompt**: Cohen's Kappa = 0.1386 rất thấp. Cần chuẩn hóa tiêu chí đánh giá giữa GPT-4o-mini và Gemini-2.5-flash bằng cách thêm rubric rõ ràng hơn (đặc biệt về tiêu chí "clarification") vào Judge Prompt để tăng agreement rate.

- [ ] **Thêm bước Reranking vào Pipeline**: Sử dụng Cross-encoder reranker để cải thiện độ chính xác của các chunk được truy xuất, giảm nguy cơ nhầm lẫn thông tin gần nhau.

- [ ] **Retrieval Quality Audit sau Ingestion**: Thiết lập quy trình kiểm tra tự động: sau mỗi lần ingestion mới, chạy test suite nhỏ (~20 cases) để xác minh các con số, mốc thời gian quan trọng được retrieve đúng chunk.

---

## 5. Phân tích Chi phí & Hiệu năng

| Chỉ số | Giá trị |
|---|---|
| Tổng chi phí | $0.026511 |
| Chi phí / case | $0.000442 |
| GPT-4o-mini tokens | 45,018 |
| Gemini-2.5-flash tokens | 113,388 |
| Tổng thời gian (60 cases, async) | < 2 phút |

**Đề xuất giảm 30% chi phí mà không giảm độ chính xác:**
1. **Dùng Gemini-2.5-flash làm Primary Judge** (chi phí thấp hơn, khắt khe hơn → phát hiện lỗi tốt hơn); chỉ escalate sang GPT-4o-mini khi có conflict hoặc score < 3.
2. **Batch evaluation**: Gộp nhiều câu hỏi tương tự vào cùng một API call (dùng batch processing) để tận dụng cache context.
3. **Giảm token output của Agent**: Hiện tại agent trả lời rất chi tiết (~300-500 tokens/response). Thêm instruction "Trả lời ngắn gọn, súc tích" để giảm completion tokens mà không giảm chất lượng nội dung.
