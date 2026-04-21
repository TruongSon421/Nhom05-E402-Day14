"""
Synthetic Data Generation (SDG) - Lab Day 14
Output: data/golden_set.jsonl (55+ test cases)
"""
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_WORKERS = 10  # số thread song song gọi API


def load_chunks() -> List[Dict]:
    with open("data/documents/hr_rag_dataset.json", encoding="utf-8") as f:
        data = json.load(f)
    chunks = []
    for doc in data["documents"]:
        for section in doc.get("sections", []):
            for sub in section.get("subsections", []):
                chunks.append({
                    "chunk_id": sub["subsection_id"],
                    "title": sub["title"],
                    "content": sub.get("content", sub.get("chunk_text", "")),
                    "chunk_text": sub.get("chunk_text", ""),
                    "doc_id": doc["doc_id"],
                    "section_title": section["title"],
                    "tags": sub.get("tags", []),
                })
            for faq in section.get("faqs", []):
                chunks.append({
                    "chunk_id": faq["faq_id"],
                    "title": faq["question"],
                    "content": faq["answer"],
                    "chunk_text": faq["answer"],
                    "doc_id": doc["doc_id"],
                    "section_title": section["title"],
                    "tags": faq.get("tags", []),
                })
    print(f"[INFO] Đã tải {len(chunks)} chunks từ dataset")
    return chunks


def generate_from_chunk(chunk: Dict, case_type: str) -> Dict | None:
    type_instructions = {
        "factual":     "Tạo 1 câu hỏi factual hỏi thẳng một sự kiện/con số/quy định cụ thể mà câu trả lời nằm rõ ràng trong đoạn văn. Độ khó: dễ-trung bình.",
        "multi_hop":   "Tạo 1 câu hỏi đòi hỏi suy luận nhiều bước hoặc kết hợp thông tin từ nhiều phần trong đoạn văn. Độ khó cao.",
        "adversarial": "Tạo 1 câu hỏi bẫy: câu hỏi TRÔNG có vẻ liên quan nhưng câu trả lời KHÔNG có hoặc sai lệch so với đoạn văn. expected_answer phải là 'Thông tin này không có trong tài liệu.' hoặc giải thích câu hỏi bị sai.",
        "negation":    "Tạo 1 câu hỏi dùng từ phủ định hoặc ngoại lệ (ví dụ: Trường hợp nào KHÔNG được, Điều gì bị cấm). Câu trả lời phải có trong đoạn văn.",
    }
    prompt = (
        "Bạn là chuyên gia tạo dữ liệu đánh giá AI.\n\n"
        f"Đoạn văn nguồn:\nTiêu đề: {chunk['title']}\nNội dung: {chunk['content'][:800]}\n\n"
        f"Nhiệm vụ: {type_instructions[case_type]}\n\n"
        "Trả về JSON (không markdown) theo format:\n"
        '{"question":"...","expected_answer":"...","reasoning":"..."}'
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        r = json.loads(resp.choices[0].message.content)
        return {
            "question": r["question"],
            "expected_answer": r["expected_answer"],
            "context": chunk["chunk_text"] or chunk["content"][:400],
            "ground_truth_ids": [chunk["chunk_id"]],
            "metadata": {
                "difficulty": "hard" if case_type in ("multi_hop", "adversarial") else "medium",
                "type": case_type,
                "topic": chunk["section_title"],
                "source_doc": chunk["doc_id"],
                "chunk_title": chunk["title"],
                "tags": chunk["tags"],
            },
        }
    except Exception as e:
        print(f"  [WARN] Lỗi {chunk['chunk_id']}: {e}")
        return None


def build_out_of_scope() -> List[Dict]:
    items = [
        ("Giá cổ phiếu của công ty hôm nay là bao nhiêu?",
         "Thông tin này không có trong tài liệu HR của công ty."),
        ("Công ty có kế hoạch niêm yết trên sàn chứng khoán không?",
         "Thông tin này không thuộc phạm vi Cẩm nang Nhân viên."),
        ("Sản phẩm chủ lực và doanh thu năm ngoái của công ty là bao nhiêu?",
         "Thông tin về doanh thu và sản phẩm không được đề cập trong tài liệu HR."),
        ("Quy trình mua cổ phần nội bộ cho nhân viên như thế nào?",
         "Không có trong tài liệu nhân sự. Vui lòng liên hệ bộ phận Tài chính."),
        ("Lịch nghỉ Tết Nguyên Đán năm 2025 của công ty là mấy ngày?",
         "Lịch nghỉ lễ theo từng năm không có trong Cẩm nang. Theo dõi thông báo chính thức từ HR."),
    ]
    return [{
        "question": q, "expected_answer": a, "context": "",
        "ground_truth_ids": [],
        "metadata": {"difficulty": "medium", "type": "out_of_scope",
                     "topic": "out_of_scope", "source_doc": "none",
                     "chunk_title": "N/A", "tags": ["out_of_scope", "hallucination_test"]},
    } for q, a in items]


def main():
    chunks = load_chunks()

    plan = [("factual", 20), ("multi_hop", 10), ("adversarial", 10), ("negation", 10)]

    jobs: List[tuple] = []
    used: set = set()
    for case_type, count in plan:
        available = [c for c in chunks if c["chunk_id"] not in used] or chunks
        selected = random.sample(available, min(count, len(available)))
        for chunk in selected:
            used.add(chunk["chunk_id"])
            jobs.append((chunk, case_type))

    print(f"[INFO] Bắt đầu sinh {len(jobs)} test cases.")

    cases: List[Dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(generate_from_chunk, chunk, ct): (chunk["chunk_id"], ct)
            for chunk, ct in jobs
        }
        done = 0
        for future in as_completed(future_map):
            done += 1
            chunk_id, ct = future_map[future]
            result = future.result()
            status = "OK " if result else "ERR"
            print(f"  [{done:2d}/{len(jobs)}] {status} {ct:12s} <- {chunk_id}")
            if result:
                cases.append(result)

    cases.extend(build_out_of_scope())
    random.shuffle(cases)

    print(f"\n[RESULT] Tổng số cases: {len(cases)}")
    for t, n in sorted(Counter(c["metadata"]["type"] for c in cases).items()):
        print(f"   {t:15s}: {n}")

    os.makedirs("data", exist_ok=True)
    with open("data/golden_set.jsonl", "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"\n[OK] Đã lưu {len(cases)} cases vào data/golden_set.jsonl")
    if len(cases) < 50:
        print("[WARN] Chưa đủ 50 cases!")
    else:
        print("[OK] Đạt yêu cầu 50+ cases.")


if __name__ == "__main__":
    main()
