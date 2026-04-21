# 📋 Quy trình Phối hợp & Phân chia Công việc - Lab Day 14

Tài liệu này xác định thứ tự triển khai (workflow) và phân định trách nhiệm cụ thể để đảm bảo hệ thống Benchmark hoạt động trơn tru và đạt điểm tối đa (Expert Level).

---

## ⏳ Lộ trình Triển khai (Sequencing)

Để tránh tình trạng "người này chờ người kia", nhóm sẽ chia làm 4 giai đoạn logic:

### Giai đoạn 1: Xây móng (Foundation) - **Ưu tiên: CAO NHẤT**
- **Người thực hiện:** Người 1 (Data) & Người 5 (DevOps).
- **Mục tiêu:** Có dữ liệu mẫu (`golden_set.jsonl`) và khung folder dự án hoàn chỉnh.
- **Thứ tự:** Người 5 setup cấu trúc dự án -> Người 1 tạo 5-10 test case đầu tiên để các người khác có cái để chạy thử.

### Giai đoạn 2: Phát triển Module (Parallel) - **Làm song song**
- **Người thực hiện:** Người 2 (Retrieval), Người 3 (Judge), Người 4 (Runner).
- **Mục tiêu:** Hoàn thiện các class core (`RetrievalEvaluator`, `LLMJudge`, `BenchmarkRunner`).
- **Lưu ý:** Các thành viên dùng dữ liệu từ Giai đoạn 1 để unit test module của mình.

### Giai đoạn 3: Tích hợp & Nâng cao (Integration & Extra)
- **Người thực hiện:** Người 5 (Tích hợp), Người 1 (Mở rộng Data), Người 4 (Optimize).
- **Mục tiêu:** Nối các module vào `main.py`, hoàn thiện các công việc **EXTRA** để lấy điểm cộng.

### Giai đoạn 4: Phân tích & Đóng gói (Analysis)
- **Người thực hiện:** Người 6 (Analyst).
- **Mục tiêu:** Chạy hệ thống, phân tích lỗi, viết báo cáo 5 Whys và kiểm tra checklist cuối cùng.

---

## 🎯 Chi tiết Trách nhiệm & Công việc EXTRA

| Vai trò | Người phụ trách | Công việc chính (Code) | **Công việc EXTRA (Điểm cộng)** |
| :--- | :--- | :--- | :--- |
| **P1: Data Lead** | [Tên] | `synthetic_gen.py`: Tạo bộ Golden Set 50+ câu. | **Red Teaming**: Tạo bộ test case "phá hoại" (jailbreak, mâu thuẫn) để test độ an toàn. |
| **P2: Retrieval Eng** | [Tên] | `retrieval_eval.py`: Tính Hit Rate & MRR. | **Deep Link**: Giải thích mối quan hệ định lượng giữa Retrieval Quality và Answer Quality. |
| **P3: AI Judge** | [Tên] | `llm_judge.py`: Triển khai chấm điểm tự động. | **Consensus Logic**: Dùng 2 model (GPT-4 + Claude), xử lý xung đột (Tie-break) và check Position Bias. |
| **P4: Backend** | [Tên] | `runner.py`: Code logic chạy async parallel. | **Performance Expert**: Tối ưu tốc độ (< 2 phút cho 50 cases) & Theo dõi Cost/Token chi tiết. |
| **P5: DevOps** | [Tên] | `main.py`: Chạy Regression V1 vs V2. | **Auto-Gate**: Logic tự động Approve/Block release dựa trên điểm Quality & Cost. |
| **P6: Analyst** | [Tên] | `failure_analysis.md`: Tổng hợp kết quả. | **Root Cause Analysis**: Phân tích "5 Whys" cực sâu cho các case bị fail. |

---

## 🔄 Luồng dữ liệu (Data Flow)

1. **Người 1** -> Sinh ra `data/golden_set.jsonl`.
2. **Người 5** -> Cung cấp code `MainAgent` (hoặc mock agent) để test.
3. **Người 4** -> Gọi logic của **Người 2** (eval retrieval) và **Người 3** (eval answer).
4. **Người 5** -> Chạy `main.py` để ra file `reports/summary.json`.
5. **Người 6** -> Đọc file summary và kết quả chi tiết để viết Analysis.

---

## ⚠️ Quy tắc Phối hợp (Critical Rules)

1. **Format là trên hết:** Người 1 phải đảm bảo file JSONL đúng format để không làm gãy code của Người 4 & 5.
2. **Không Code block:** Nếu module chưa xong, hãy để lại `mock_result` để người khác vẫn có thể tích hợp được.
3. **Check Lab:** Trước khi bàn giao cho Người 6, Người 5 phải chạy `python check_lab.py` để đảm bảo không có lỗi runtime.

> [!IMPORTANT]
> **Điểm thưởng (Bonus):** Ai hoàn thiện phần EXTRA của mình trước thời hạn sẽ được ưu tiên cộng điểm đóng góp (Engineering Contribution - 15đ).
