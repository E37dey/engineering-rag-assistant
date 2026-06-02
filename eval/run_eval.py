"""Evaluate the RAG system against eval/test_questions.json.

Run: python eval/run_eval.py

For each question:
  * in_corpus    -> answer must contain all required keywords AND the
                    expected source must be cited (or appear as top-1
                    retrieval) AND the model must not have refused.
  * out_of_corpus -> answer must contain the exact refusal phrase.

Prints per-question status, an aggregate accuracy + refusal precision
table, and a failure breakdown.

The matching convention is intentionally simple: each item in
`expected_answer_contains` may include `|` to specify alternatives
(any one matches), and items are then ANDed together. So
`["two|dual", "op"]` is read as "(two OR dual) AND op".
"""
import json
import sys
import time
from pathlib import Path

# Make project root importable when running this script directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.generate import generate_answer

QUESTIONS_PATH = Path(__file__).parent / "test_questions.json"
REFUSAL_PHRASE = "i don't have information on this in the provided documents"


def keyword_matches(answer_lower: str, keyword_spec: str) -> bool:
    """Each `|`-separated alternative within a spec satisfies the slot."""
    alternatives = [a.strip().lower() for a in keyword_spec.split("|") if a.strip()]
    return any(alt in answer_lower for alt in alternatives)


def evaluate_in_corpus(q: dict, result: dict) -> dict:
    answer_lower = result["answer"].lower()
    expected_keywords = q["expected_answer_contains"]
    expected_source = q["expected_source"]

    keywords_missing = [
        k for k in expected_keywords if not keyword_matches(answer_lower, k)
    ]
    keywords_ok = not keywords_missing

    source_cited_in_answer = expected_source.lower() in answer_lower
    top_source = result["sources"][0]["filename"] if result["sources"] else None
    source_top1 = (top_source == expected_source)
    source_ok = source_cited_in_answer or source_top1

    refused_wrongly = REFUSAL_PHRASE in answer_lower

    return {
        "passed": keywords_ok and source_ok and not refused_wrongly,
        "keywords_ok": keywords_ok,
        "keywords_missing": keywords_missing,
        "source_ok": source_ok,
        "source_cited_in_answer": source_cited_in_answer,
        "source_top1_correct": source_top1,
        "top_source": top_source,
        "refused_wrongly": refused_wrongly,
    }


def evaluate_out_of_corpus(q: dict, result: dict) -> dict:
    answer_lower = result["answer"].lower()
    refused = REFUSAL_PHRASE in answer_lower
    return {"passed": refused, "refused": refused}


def main() -> int:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    print(f"Running {len(questions)} eval questions...")
    print()

    results = []
    t_start = time.perf_counter()
    for i, q in enumerate(questions, start=1):
        print(
            f"[{i:>2}/{len(questions)}] {q['type']:<14} {q['id']:<30}",
            end="",
            flush=True,
        )
        t0 = time.perf_counter()
        result = generate_answer(q["question"])
        elapsed = time.perf_counter() - t0

        if q["type"] == "in_corpus":
            ev = evaluate_in_corpus(q, result)
        else:
            ev = evaluate_out_of_corpus(q, result)

        status = "PASS" if ev["passed"] else "FAIL"
        print(f"  {status}  ({elapsed:4.1f}s)")
        results.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "answer": result["answer"],
            "passed": ev["passed"],
            "eval": ev,
        })

    total_elapsed = time.perf_counter() - t_start

    in_corpus = [r for r in results if r["type"] == "in_corpus"]
    out_corpus = [r for r in results if r["type"] == "out_of_corpus"]
    n_total = len(results)
    n_passed = sum(1 for r in results if r["passed"])
    n_in_passed = sum(1 for r in in_corpus if r["passed"])
    n_out_passed = sum(1 for r in out_corpus if r["passed"])

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"Overall accuracy:       {n_passed}/{n_total}    "
          f"({100 * n_passed / n_total:5.1f}%)")
    print(f"In-corpus accuracy:     {n_in_passed}/{len(in_corpus)}    "
          f"({100 * n_in_passed / len(in_corpus):5.1f}%)")
    print(f"Refusal precision:      {n_out_passed}/{len(out_corpus)}    "
          f"({100 * n_out_passed / len(out_corpus):5.1f}%)")
    print(f"Total elapsed:          {total_elapsed:.0f}s    "
          f"({total_elapsed / n_total:.1f}s avg)")

    failures = [r for r in results if not r["passed"]]
    if failures:
        print()
        print("=" * 72)
        print(f"FAILURES ({len(failures)})")
        print("=" * 72)
        for r in failures:
            print(f"  [{r['id']}]  ({r['type']})")
            print(f"    Q: {r['question']}")
            if r["type"] == "in_corpus":
                ev = r["eval"]
                if ev["keywords_missing"]:
                    print(f"    -> missing keyword(s): {ev['keywords_missing']}")
                if not ev["source_ok"]:
                    print(
                        f"    -> wrong source: top-1 was '{ev['top_source']}', "
                        f"citation not in answer"
                    )
                if ev["refused_wrongly"]:
                    print(f"    -> refused incorrectly")
            else:
                snippet = " ".join(r["answer"].split())[:160]
                print(f"    -> did not refuse: {snippet}...")
            print()

    return 0 if n_passed == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
