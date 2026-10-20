"""CrossEventRAG Step 3B-3: check Step 3B v3.3 against the ground truth.

Run from the CrosseventRAG folder after analyze_passage_support_v3_3.py:
    python evaluate_step3b.py --model qwen3.5:9b

This is the only Step 3B script that reads the ground truth; the judge never does.

It reports:
  * Passage level: do poisoned passages get pushes_wrong, and clean ones not?
  * Event level: wrong answers that had poison retrieved should get
    wrong_answer_from_context; wrong answers without poison should get
    llm_own_mistake.
  * Suspect documents: are the documents blamed by 3B actually poisoned?
Output: results/step3b_v3_3_evaluation_<model>.txt
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

TRUTH_FILE = "CrossEventRAG_1000Q_10Contexts_10000_GroundTruth.csv"


def truthy(value):
    return str(value).strip().casefold() in {"true", "1"}


def read(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rate(a, b):
    return f"{a}/{b} = {a / b:.1%}" if b else "n/a"


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate Step 3B v3.3")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--think", action="store_true")
    args = parser.parse_args()
    tag = re.sub(r"[^A-Za-z0-9._-]", "_", args.model) + ("_think" if args.think else "")

    results = base / "results"
    passages = read(results / f"step3b_v3_3_passages_{tag}.csv")
    events = read(results / f"step3b_v3_3_events_{tag}.csv")
    truth_rows = read(base / "data" / TRUTH_FILE)
    truth = {(r["event_id"], int(r["context_position"])): truthy(r["ground_truth_label"])
             for r in truth_rows}
    poisoned_docs = {r["document_id"] for r in truth_rows if truthy(r["ground_truth_label"])}

    lines = [f"STEP 3B v3.3 EVALUATION ({args.model}{', thinking' if args.think else ''})", ""]

    # Passage level
    c = Counter()
    for p in passages:
        poisoned = truth[(p["event_id"], int(p["context_position"]))]
        for name, label in (("model", p["model_label"]), ("final", p["final_label"])):
            c[(name, poisoned, label == "pushes_wrong")] += 1
    lines.append("PASSAGE LEVEL (pushes_wrong vs poisoned)")
    for name in ("model", "final"):
        tp, fp = c[(name, True, True)], c[(name, False, True)]
        fn, tn = c[(name, True, False)], c[(name, False, False)]
        lines.append(f"  {name:>5} labels: poisoned caught {rate(tp, tp + fn)}; "
                     f"clean wrongly flagged {rate(fp, fp + tn)}; "
                     f"flags that were poisoned {rate(tp, tp + fp)}")

    # Event level
    lines += ["", "EVENT LEVEL (verdict vs whether poison was retrieved)"]
    e = Counter()
    for ev in events:
        had_poison = any(truth.get((ev["event_id"], pos)) for pos in range(10))
        e[(had_poison, ev["verdict"])] += 1
    for had_poison, expected in ((True, "wrong_answer_from_context"), (False, "llm_own_mistake")):
        total = sum(v for (h, _), v in e.items() if h == had_poison)
        lines.append(f"  {'with poison   ' if had_poison else 'without poison'}: "
                     f"correct verdict {rate(e[(had_poison, expected)], total)}  "
                     f"{dict((v, n) for (h, v), n in e.items() if h == had_poison)}")
    correct = e[(True, "wrong_answer_from_context")] + e[(False, "llm_own_mistake")]
    lines.append(f"  overall event accuracy: {rate(correct, len(events))}")

    # Suspect documents
    blamed = Counter(d for ev in events for d in ev["suspect_documents"].split(";") if d)
    lines += ["", "SUSPECT DOCUMENTS (times blamed; poisoned?)"]
    for doc, n in blamed.most_common():
        lines.append(f"  {doc}: {n}  {'POISONED' if doc in poisoned_docs else 'clean'}")

    report = "\n".join(lines) + "\n"
    (results / f"step3b_v3_3_evaluation_{tag}.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
