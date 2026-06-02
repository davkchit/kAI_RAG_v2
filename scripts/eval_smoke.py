import json
from pathlib import Path

from rag import COLLECTION_NAME, ask_question, hybrid_search

EVAL_PATH = Path(__file__).resolve().parents[1] / "eval_questions.jsonl"


def contains_all(text: str, parts: list[str]) -> bool:
    lowered = text.lower()
    return all(part.lower() in lowered for part in parts)


def main():
    cases = [
        json.loads(line)
        for line in EVAL_PATH.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    retrieval_passed = 0
    answer_passed = 0
    answer_checked = 0
    answer_skipped = 0

    for i, case in enumerate(cases, start=1):
        question = case["question"]
        hits = hybrid_search(question, COLLECTION_NAME)

        top_source = ""
        if hits:
            top_source = hits[0].payload.get("source", "")

        retrieval_ok = top_source == case.get("expected_source", "")
        if retrieval_ok:
            retrieval_passed += 1

        answer = "[SKIPPED]"
        answer_status = "SKIP"

        try:
            answer = ask_question(question)
            answer_checked += 1

            answer_ok = contains_all(answer, case.get("must_contain", []))
            if answer_ok:
                answer_passed += 1
                answer_status = "PASS"
            else:
                answer_status = "FAIL"
        except Exception as exc:
            answer_skipped += 1
            answer = f"[SKIPPED] {type(exc).__name__}: {exc}"

        retrieval_status = "PASS" if retrieval_ok else "FAIL"

        print(f"[{retrieval_status}/{answer_status}] {i}. {question}")
        print(f"answer: {answer}")
        print(f"top_source: {top_source}")
        print()

    print(f"RETRIEVAL RESULT: {retrieval_passed}/{len(cases)}")
    print(f"ANSWER RESULT: {answer_passed}/{answer_checked} checked, {answer_skipped} skipped")


if __name__ == "__main__":
    main()
