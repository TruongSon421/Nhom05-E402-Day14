Bùi Thế Công - 2A202600008
Bùi Lâm Tiến - 2A202600004
Trần Trường Sơn - 2A202600313
Trần Ngọc Huy - 2A202600298
Trường Đăng Nghĩa - 2A202600407
Nông Trung Kiên - 2A202600414
 
  👤 Người 1 — Data Lead: Golden Dataset & SDG (Bùi Lâm Tiến)

  File chính: data/synthetic_gen.py

- Tạo 50+ test cases chất lượng với ground_truth_ids
- Bao gồm bộ Red Teaming (câu hỏi đánh lừa, edge cases)
- Đảm bảo đúng format JSONL cho data/golden_set.jsonl
- Chạy python data/synthetic_gen.py và verify output

  ---
  👤 Người 2 — Retrieval Engineer: Hit Rate & MRR (Trần Trường Sơn)

  File chính: data/retrieval_eval.py

- Implement tính Hit Rate (chunk có trong top-K kết quả?)
- Implement tính MRR (Mean Reciprocal Rank)
- Kết nối với Vector DB, chứng minh retrieval stage hoạt động
- Viết report liên hệ Retrieval Quality → Answer Quality

  ---
  👤 Người 3 — AI Judge: Multi-Model Consensus (Trần Ngọc Huy)

  File chính: engine/llm_judge.py

- Triển khai 2 model judge (ví dụ: GPT-4o + Claude)
- Tính Agreement Rate (Cohen's Kappa hoặc đơn giản hơn)
- Logic xử lý xung đột điểm số tự động (ví dụ: average, tie-break)
- Tránh Position Bias khi judge

  ---
  👤 Người 4 — Backend: Async Runner & Performance (Nông Trung Kiên)

  File chính: engine/runner.py

- Đảm bảo toàn bộ pipeline chạy async/parallel
- Target: < 2 phút cho 50 cases
- Thêm Cost & Token usage tracking vào mỗi kết quả
- Báo cáo giá tiền mỗi lần eval

  ---
  👤 Người 5 — DevOps: Regression Gate & main.py (Trường Đăng Nghĩa)

  File chính: main.py

- Implement so sánh V1 vs V2 có ý nghĩa (không phải giả lập)
- Logic Auto-Gate: Release/Rollback dựa trên ngưỡng Quality + Cost + Performance
- Tạo reports/summary.json và reports/benchmark_results.json đúng format
- Chạy python check_lab.py trước khi nộp

  ---
  👤 Người 6 — Analyst: Failure Analysis & Tích hợp (Bùi Thế Công)

  File chính: analysis/failure_analysis.md

- Phân tích 5 Whys sau khi có kết quả benchmark
- Phân cụm lỗi (Failure Clustering): Chunking? Ingestion? Retrieval? Prompting?
- Viết reflection cá nhân cho toàn nhóm (analysis/reflections/reflection_[Tên].md)
- Tích hợp toàn bộ module, đảm bảo pipeline chạy end-to-end
