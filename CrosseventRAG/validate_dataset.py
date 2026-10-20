
"""Step 1: Validate CrossEventRAG's 1,000-event x 10-context prototype dataset.

No third-party packages needed. This script never changes the CSV inputs.

From CrosseventRAG/, run:
    python validate_dataset.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

MASTER_NAME = "CrossEventRAG_1000Q_10Contexts_10000_Master.csv"
DETECTOR_NAME = "CrossEventRAG_1000Q_10Contexts_10000_Detector.csv"
TRUTH_NAME = "CrossEventRAG_1000Q_10Contexts_10000_GroundTruth.csv"
MANIFEST_NAME = "CrossEventRAG_20Documents_Source_Manifest.csv"

MASTER_COLUMNS = [
    "event_id", "question", "correct_answer", "RAG_response",
    "context_position", "context_text", "retrieval_score",
    "ground_truth_label", "document_id",
]

DETECTOR_COLUMNS = [
    name for name in MASTER_COLUMNS
    if name not in {"correct_answer", "ground_truth_label"}
]

TRUTH_COLUMNS = [
    "event_id", "context_position", "document_id",
    "ground_truth_label", "correct_answer", "RAG_response",
]


def load_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        rows = list(reader)

    if not columns:
        raise ValueError(f"CSV has no header: {path}")

    return columns, rows


def normalized(text: str) -> str:
    """Formatting-only comparison, not semantic answer checking."""
    return " ".join(text.strip().casefold().split())


def label_value(text: str):
    value = text.strip().casefold()

    if value in {"true", "1"}:
        return True

    if value in {"false", "0"}:
        return False

    return None


def main():

    parser = argparse.ArgumentParser(
        description="Validate Step 1 CrossEventRAG CSV files"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Folder containing the four CSV files",
    )

    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()

    report_path = (
        Path(__file__).resolve().parent
        / "results"
        / "step1_validation_report.txt"
    )

    results = []
    errors = []
    warnings = []

    def check(condition, title, detail=""):
        marker = "PASS" if condition else "FAIL"

        results.append(
            f"[{marker}] {title}"
            + (f" - {detail}" if detail else "")
        )

        if not condition:
            errors.append(
                title + (f": {detail}" if detail else "")
            )

    # -----------------------------------
    # CHECK 1: Load all dataset files
    # -----------------------------------

    try:
        master_cols, master = load_csv(
            data_dir / MASTER_NAME
        )

        detector_cols, detector = load_csv(
            data_dir / DETECTOR_NAME
        )

        truth_cols, truth = load_csv(
            data_dir / TRUTH_NAME
        )

        manifest_cols, manifest = load_csv(
            data_dir / MANIFEST_NAME
        )

    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[FAIL] {exc}")
        print(
            "Place all four CSV files inside CrosseventRAG\\data"
        )
        raise SystemExit(1)

    # -----------------------------------
    # CHECK 2: Validate columns and rows
    # -----------------------------------

    check(
        master_cols == MASTER_COLUMNS,
        "Master has exactly the nine required columns"
    )

    check(
        detector_cols == DETECTOR_COLUMNS,
        "Detector excludes correct answers and poison labels"
    )

    check(
        truth_cols == TRUTH_COLUMNS,
        "Ground truth contains expected evaluation fields"
    )

    check(
        "document_id" in manifest_cols,
        "Manifest has document_id column"
    )

    check(
        len(master) == 10000,
        "Master has exactly 10,000 rows",
        f"found {len(master)}"
    )

    check(
        len(detector) == len(master) == len(truth),
        "All three main files have the same row count"
    )

    missing = [
        (idx, col)
        for idx, row in enumerate(master, start=2)
        for col in MASTER_COLUMNS
        if not row.get(col, "").strip()
    ]

    check(
        not missing,
        "No missing values in Master",
        f"missing cells {len(missing)}"
    )

    # -----------------------------------
    # CHECK 3: Validate context positions,
    # retrieval scores and labels
    # -----------------------------------

    by_event = defaultdict(list)
    by_doc = defaultdict(list)

    bad_scores = []
    bad_positions = []
    bad_labels = []

    for line_number, row in enumerate(master, start=2):

        by_event[row["event_id"]].append(row)
        by_doc[row["document_id"]].append(row)

        try:
            position = int(row["context_position"])

            if str(position) != row["context_position"].strip():
                bad_positions.append(line_number)

        except ValueError:
            bad_positions.append(line_number)

        try:
            score = float(row["retrieval_score"])

            if not 0.0 <= score <= 1.0:
                bad_scores.append(line_number)

        except ValueError:
            bad_scores.append(line_number)

        if label_value(row["ground_truth_label"]) is None:
            bad_labels.append(line_number)

    check(
        not bad_positions,
        "Context positions are valid integers",
        f"invalid rows {len(bad_positions)}"
    )

    check(
        not bad_scores,
        "Retrieval scores are between 0 and 1",
        f"invalid rows {len(bad_scores)}"
    )

    check(
        not bad_labels,
        "Poison labels are valid True/False values",
        f"invalid rows {len(bad_labels)}"
    )

    # -----------------------------------
    # CHECK 4: Validate 1000 questions
    # and 10 contexts per question
    # -----------------------------------

    check(
        len(by_event) == 1000,
        "Exactly 1,000 event IDs",
        f"found {len(by_event)}"
    )

    bad_size = [
        event
        for event, rows in by_event.items()
        if len(rows) != 10
    ]

    check(
        not bad_size,
        "Exactly 10 context rows per event",
        f"bad events {len(bad_size)}"
    )

    bad_event_positions = []
    bad_event_metadata = []

    for event, rows in by_event.items():

        try:
            positions = [
                int(r["context_position"]) for r in rows
            ]

        except ValueError:
            continue

        if sorted(positions) != list(range(10)):
            bad_event_positions.append(event)

        metadata = {
            (
                r["question"],
                r["correct_answer"],
                r["RAG_response"]
            )
            for r in rows
        }

        if len(metadata) != 1:
            bad_event_metadata.append(event)

    check(
        not bad_event_positions,
        "Each event has positions 0 through 9 once",
        f"bad events {len(bad_event_positions)}"
    )

    check(
        not bad_event_metadata,
        "Each event has one consistent question and answer pair",
        f"bad events {len(bad_event_metadata)}"
    )

    question_texts = {
        normalized(rows[0]["question"])
        for rows in by_event.values()
    }

    check(
        len(question_texts) == 1000,
        "1,000 distinct question strings",
        f"found {len(question_texts)}"
    )

    # -----------------------------------
    # CHECK 5: Validate document IDs
    # -----------------------------------

    check(
        len(by_doc) == 20,
        "Exactly 20 distinct source-document IDs",
        f"found {len(by_doc)}"
    )

    mixed_label_ids = [
        doc for doc, rows in by_doc.items()
        if len({
            label_value(r["ground_truth_label"])
            for r in rows
        }) != 1
    ]

    check(
        not mixed_label_ids,
        "Each document has a consistent poison label",
        f"mixed-label docs {len(mixed_label_ids)}"
    )

    different_chunks = sum(
        len({
            normalized(r["context_text"])
            for r in rows
        }) > 1
        for rows in by_doc.values()
    )

    check(
        different_chunks == 20,
        "All 20 documents have multiple different context texts",
        f"found {different_chunks}"
    )

    cross_event_docs = sum(
        len({
            r["event_id"]
            for r in rows
        }) > 1
        for rows in by_doc.values()
    )

    check(
        cross_event_docs == 20,
        "All document IDs recur across different events",
        f"found {cross_event_docs}"
    )

    # -----------------------------------
    # CHECK 6: Validate poisoning labels
    # -----------------------------------

    count_poison = sum(
        label_value(r["ground_truth_label"]) is True
        for r in master
    )

    count_clean = sum(
        label_value(r["ground_truth_label"]) is False
        for r in master
    )

    check(
        count_poison == 500,
        "Exactly 500 poisoned context rows",
        f"found {count_poison}"
    )

    check(
        count_clean == 9500,
        "Exactly 9,500 clean context rows",
        f"found {count_clean}"
    )

    poisoned_docs = sum(
        all(
            label_value(r["ground_truth_label"]) is True
            for r in rows
        )
        for rows in by_doc.values()
    )

    check(
        poisoned_docs == 5,
        "Exactly five injected-source document IDs",
        f"found {poisoned_docs}"
    )

    # -----------------------------------
    # CHECK 7: Validate RAG answers
    # -----------------------------------

    correct_events = 0
    groups = Counter()

    for event, rows in by_event.items():

        sample = rows[0]

        is_correct = (
            normalized(sample["correct_answer"])
            == normalized(sample["RAG_response"])
        )

        if is_correct:
            correct_events += 1

        poisoned_retrieved = any(
            label_value(r["ground_truth_label"]) is True
            for r in rows
        )

        groups[(is_correct, poisoned_retrieved)] += 1

    check(
        correct_events == 680,
        "Exactly 680 correct-answer events (68%)",
        f"found {correct_events}"
    )

    check(
        len(by_event) - correct_events == 320,
        "Exactly 320 incorrect-answer events (32%)"
    )

    expected_groups = {
        (True, False): 480,
        (True, True): 200,
        (False, False): 120,
        (False, True): 200,
    }

    check(
        groups == Counter(expected_groups),
        "Correct/error outcomes include clean and poisoned events",
        ", ".join(
            f"{key}:{groups[key]}"
            for key in expected_groups
        )
    )

    # -----------------------------------
    # CHECK 8: Compare all three CSVs
    # -----------------------------------

    def indexed(rows, name):

        result = {}
        duplicate = 0

        for row in rows:

            key = (
                row.get("event_id", ""),
                row.get("context_position", "")
            )

            if key in result:
                duplicate += 1

            result[key] = row

        check(
            duplicate == 0,
            f"No duplicate event/position pairs in {name}",
            f"duplicates {duplicate}"
        )

        return result

    master_map = indexed(master, "Master")
    detector_map = indexed(detector, "Detector")
    truth_map = indexed(truth, "GroundTruth")

    check(
        set(master_map)
        == set(detector_map)
        == set(truth_map),
        "Master/Detector/GroundTruth have identical keys"
    )

    mismatch_detector = 0
    mismatch_truth = 0

    for key, row in master_map.items():

        det = detector_map.get(key)
        tru = truth_map.get(key)

        if det is None or any(
            row.get(field) != det.get(field)
            for field in DETECTOR_COLUMNS
        ):
            mismatch_detector += 1

        if tru is None or any(
            row.get(field) != tru.get(field)
            for field in TRUTH_COLUMNS
        ):
            mismatch_truth += 1

    check(
        mismatch_detector == 0,
        "Detector values match the Master",
        f"mismatched rows {mismatch_detector}"
    )

    check(
        mismatch_truth == 0,
        "GroundTruth values match the Master",
        f"mismatched rows {mismatch_truth}"
    )

    # -----------------------------------
    # CHECK 9: Validate source manifest
    # -----------------------------------

    manifest_ids = [
        r.get("document_id", "")
        for r in manifest
    ]

    check(
        len(manifest_ids) == len(set(manifest_ids)) == 20,
        "Manifest has 20 unique document IDs"
    )

    check(
        set(manifest_ids) == set(by_doc),
        "Every document ID is listed in source manifest"
    )

    # -----------------------------------
    # CHECK 10: Save validation report
    # -----------------------------------

    warnings.append(
        "Answers and retrieval scores are SIMULATED, "
        "not actual RAG/LLM outputs."
    )

    warnings.append(
        "Questions have 1,000 distinct strings "
        "but about 100 underlying fact targets."
    )

    warnings.append(
        "Structure checks do not prove passage relevance "
        "or causal poisoning responsibility."
    )

    warnings.append(
        "Keep ground-truth CSV and source manifest "
        "away from detector inference."
    )

    headline = (
        "STEP 1 VALIDATION: "
        + ("PASS" if not errors else "FAIL")
    )

    lines = [
        headline,
        f"Data folder: {data_dir}",
        "",
        "RESULTS",
        *results,
        "",
        "SUMMARY",
        (
            f"Events: {len(by_event)}; "
            f"context rows: {len(master)}; "
            f"documents: {len(by_doc)}"
        ),
        (
            f"Clean rows: {count_clean}; "
            f"poisoned rows: {count_poison}; "
            f"poisoned sources: {poisoned_docs}"
        ),
        (
            f"Correct events: {correct_events}; "
            f"incorrect events: {len(by_event) - correct_events}"
        ),
        f"Correct without poison: {groups[(True, False)]}",
        f"Correct with poison: {groups[(True, True)]}",
        f"Incorrect without poison: {groups[(False, False)]}",
        f"Incorrect with poison: {groups[(False, True)]}",
        "",
        "RESEARCH LIMITATIONS",
        *["- " + message for message in warnings],
        "",
        f"Errors: {len(errors)}",
    ]

    report = "\n".join(lines) + "\n"

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path.write_text(
        report,
        encoding="utf-8"
    )

    print(report)
    print(f"Report saved to: {report_path}")

    raise SystemExit(
        0 if not errors else 1
    )


if __name__ == "__main__":
    main()
