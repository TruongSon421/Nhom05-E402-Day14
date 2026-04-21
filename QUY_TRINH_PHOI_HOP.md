# 📑 Hướng dẫn Triển khai Hợp nhất (Strictly Source-Accurate) - Lab Day 14

Tài liệu này được tổng hợp 100% nguyên văn từ các tài liệu chuẩn của Lab 14. Không có chi tiết tự suy luận.

---

## 🧪 Lộ trình Phối hợp Từng bước (Theo README.md)

| Bước | Thời gian | Người thực hiện | Công việc & Mục tiêu | File chính |
| :--- | :--- | :--- | :--- | :--- |
| **Bước 1** | **45 Phút** | **Tiến (P1)** & **Nghĩa (P5)** | **Thiết kế Golden Dataset & SDG**: Nghĩa setup khung code. Tiến tạo 10-20 case đầu tiên làm dữ liệu mẫu. | `data/synthetic_gen.py`<br>`main.py` |
| **Bước 2** | **90 Phút** | **Sơn (P2)**, **Huy (P3)**, **Kiên (P4)** | **Phát triển Eval Engine & Runner**: Sơn (Retrieval), Huy (Judge), Kiên (Async Runner). | `engine/retrieval_eval.py`<br>`engine/llm_judge.py`<br>`engine/runner.py` |
| **Bước 3** | **60 Phút** | **Nghĩa (P5)**, **Tiến (P1)**, **Kiên (P4)** | **Tích hợp & Benchmark**: Chạy Benchmark. Tiến hoàn thiện 50+ case (Hard Cases). Kiên tối ưu tốc độ & báo giá. | `main.py`<br>`data/synthetic_gen.py`<br>`engine/runner.py` |
| **Bước 4** | **45 Phút** | **Công (P6)** & **Cả nhóm** | **Phân tích lỗi & Tối ưu**: Phân cụm lỗi, 5 Whys. Viết Reflection cá nhân và chạy `check_lab.py`. | `analysis/failure_analysis.md`<br>`check_lab.py` |

---

## 👥 Phân nhiệm & Expert Level (Theo Rubric & README)

| Người phụ trách | Vai trò | File chính | **Expert Level (Bonus Tasks)** |
| :--- | :--- | :--- | :--- |
| **Bùi Lâm Tiến** | P1: Data Lead | `synthetic_gen.py` | Tạo 50+ cases kèm Ground Truth IDs. Thiết kế **Red Teaming** phá vỡ hệ thống thành công và các **Hard Cases** (Adversarial, Edge, Multi-turn). |
| **Trần Trường Sơn** | P2: Retrieval Eng | `retrieval_eval.py` | Giải thích mối liên hệ giữa **Retrieval Quality** và **Answer Quality**. Chỉ ra chính xác chunk nào gây Hallucination. |
| **Trần Ngọc Huy** | P3: AI Judge | `llm_judge.py` | Triển khai 2+ model Judge. Tính toán **Cohen's Kappa** (Hệ số đồng thuận). Kiểm tra lỗi **Position Bias**. |
| **Nông Trung Kiên** | P4: Backend | `runner.py` | Chạy song song cực nhanh (< 2 phút cho 50 case). Báo cáo **Cost & Token usage**. Đề xuất giảm 30% chi phí eval. |
| **Trường Đăng Nghĩa** | P5: DevOps | `main.py` | So sánh **V1 vs V2**. Logic **Auto-Gate Release/Rollback** dựa trên chỉ số Chất lượng/Chi phí/Hiệu năng. |
| **Bùi Thế Công** | P6: Analyst | `failure_analysis.md` | Chỉ ra lỗi nằm ở đâu: **Ingestion pipeline, Chunking strategy, Retrieval, hay Prompting** thông qua phân tích **5 Whys**. |

---

## 💎 Đặc tả Kỹ thuật (Chi tiết Phụ lục)

### 1. Hard Case Design (Theo `HARD_CASES_GUIDE.md`)
Tiến (P1) thực hiện thiết kế các loại case sau:
- **Adversarial**: Prompt Injection, Goal Hijacking.
- **Edge Cases**: Out of Context, Ambiguous, Conflicting Info.
- **Multi-turn Complexity**: Context carry-over, Correction.
- **Technical Constraints**: Latency Stress, Cost Efficiency.

### 2. Multi-Judge Calibration (Theo `README.md` & `GRADING_RUBRIC.md`)
Huy (P3) đảm bảo tính khách quan của hệ thống:
- Chứng minh hệ thống khách quan bằng cách so sánh nhiều Judge model.
- Tính toán độ tin cậy bằng **Cohen's Kappa**.
- Có logic xử lý xung đột điểm số tự động.

### 3. Failure Analysis (Theo `README.md` & `failure_analysis.md`)
Công (P6) phân tích sâu nguyên nhân gốc rễ:
- Thực hiện **Failure Clustering** (Phân cụm lỗi).
- Báo cáo **5 Whys** phải chỉ rõ lỗi thuộc về tầng nào của hệ thống (Ingestion, Chunking, Retrieval, hay Prompting).

---

## 📥 Điều kiện nộp bài (Theo `check_lab.py`)
Hệ thống chỉ đạt chuẩn khi chạy `python check_lab.py` và báo ✅ xanh cho:
- Tồn tại đủ `summary.json`, `benchmark_results.json`, và `failure_analysis.md`.
- `metrics` chứa đủ trường `hit_rate` và `agreement_rate`.
- Toàn bộ Benchmark chạy song song (Async) hoàn thành dưới 2 phút.
