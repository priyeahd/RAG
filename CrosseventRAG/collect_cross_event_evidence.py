
"""Step 2: Collect cross-event document evidence from Detector.csv.

This script does NOT read poisoning labels or correct answers.
It groups retrieved contexts by document ID and question event.
"""

import argparse
import csv
import json

from collections import defaultdict
from pathlib import Path
from statistics import mean


DETECTOR_FILENAME = (
    "CrossEventRAG_1000Q_10Contexts_10000_Detector.csv"
)

REQUIRED_COLUMNS = [
    "event_id",
    "question",
    "RAG_response",
    "context_position",
    "context_text",
    "retrieval_score",
    "document_id",
]


# -----------------------------------
# STEP A: Read the Detector dataset
# -----------------------------------

def read_detector(path):

    if not path.is_file():
        raise FileNotFoundError(
            f"Detector CSV not found: {path}"
        )

    records = []
    seen_positions = set()
    event_metadata = {}

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:

        reader = csv.DictReader(file)

        missing_columns = (
            set(REQUIRED_COLUMNS)
            - set(reader.fieldnames or [])
        )

        if missing_columns:
            raise ValueError(
                f"Missing columns: {sorted(missing_columns)}"
            )

        for line_number, row in enumerate(
            reader, start=2
        ):

            if any(
                not row[col].strip()
                for col in REQUIRED_COLUMNS
            ):
                raise ValueError(
                    f"Missing value on line {line_number}"
                )

            event_id = row["event_id"]
            position = int(row["context_position"])
            score = float(row["retrieval_score"])

            if not 0 <= score <= 1:
                raise ValueError(
                    f"Invalid retrieval score: line {line_number}"
                )

            row_key = (event_id, position)

            if row_key in seen_positions:
                raise ValueError(
                    f"Duplicate event/position: {row_key}"
                )

            seen_positions.add(row_key)

            metadata = (
                row["question"],
                row["RAG_response"]
            )

            if (
                event_id in event_metadata
                and event_metadata[event_id] != metadata
            ):
                raise ValueError(
                    f"Inconsistent question or answer: {event_id}"
                )

            event_metadata[event_id] = metadata

            records.append({
                **{
                    col: row[col]
                    for col in REQUIRED_COLUMNS
                },
                "context_position": position,
                "retrieval_score": score,
            })

    if not records:
        raise ValueError("Detector dataset is empty")

    return records


# -----------------------------------
# STEP B: Save output CSV
# -----------------------------------

def save_csv(path, columns, rows):

    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------
# STEP C: Main program
# -----------------------------------

def main():

    BASE = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="CrossEventRAG Step 2"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BASE / "data"
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=BASE / "results"
    )

    args = parser.parse_args()

    detector_path = (
        args.data_dir / DETECTOR_FILENAME
    )

    print("Reading Detector dataset...")

    records = read_detector(detector_path)

    # -----------------------------------
    # STEP D: Group contexts by document
    # and by event-document pair
    # -----------------------------------

    by_event_document = defaultdict(list)
    by_document = defaultdict(list)

    for record in records:

        event_id = record["event_id"]
        document_id = record["document_id"]

        key = (event_id, document_id)

        by_event_document[key].append(record)
        by_document[document_id].append(record)

    # -----------------------------------
    # STEP E: Collect event-document
    # evidence
    # -----------------------------------

    event_document_rows = []

    for (event_id, document_id), contexts in sorted(
        by_event_document.items()
    ):

        contexts = sorted(
            contexts,
            key=lambda r: r["context_position"]
        )

        scores = [
            r["retrieval_score"]
            for r in contexts
        ]

        event_document_rows.append({

            "event_id": event_id,

            "document_id": document_id,

            "question": contexts[0]["question"],

            "RAG_response": contexts[0]["RAG_response"],

            "context_count": len(contexts),

            "context_positions": json.dumps([
                r["context_position"]
                for r in contexts
            ]),

            "context_texts": json.dumps(
                [
                    r["context_text"]
                    for r in contexts
                ],
                ensure_ascii=False
            ),

            "mean_retrieval_score": round(
                mean(scores), 6
            ),

            "max_retrieval_score": round(
                max(scores), 6
            ),
        })

    # -----------------------------------
    # STEP F: Aggregate evidence for
    # each document across all events
    # -----------------------------------

    document_rows = []

    for document_id, contexts in sorted(
        by_document.items()
    ):

        event_ids = sorted({
            r["event_id"]
            for r in contexts
        })

        scores = [
            r["retrieval_score"]
            for r in contexts
        ]

        unique_texts = {
            " ".join(r["context_text"].split())
            for r in contexts
        }

        document_rows.append({

            "document_id": document_id,

            "distinct_event_count": len(event_ids),

            "total_context_rows": len(contexts),

            "distinct_context_texts": len(unique_texts),

            "mean_retrieval_score": round(
                mean(scores), 6
            ),

            "max_retrieval_score": round(
                max(scores), 6
            ),

            "event_ids": ";".join(event_ids),
        })

    # -----------------------------------
    # STEP G: Save results
    # -----------------------------------

    results_dir = args.results_dir

    detail_path = (
        results_dir
        / "step2_event_document_evidence.csv"
    )

    summary_path = (
        results_dir
        / "step2_document_evidence.csv"
    )

    save_csv(
        detail_path,
        [
            "event_id",
            "document_id",
            "question",
            "RAG_response",
            "context_count",
            "context_positions",
            "context_texts",
            "mean_retrieval_score",
            "max_retrieval_score",
        ],
        event_document_rows
    )

    save_csv(
        summary_path,
        [
            "document_id",
            "distinct_event_count",
            "total_context_rows",
            "distinct_context_texts",
            "mean_retrieval_score",
            "max_retrieval_score",
            "event_ids",
        ],
        document_rows
    )

    # -----------------------------------
    # STEP H: Print the final output
    # -----------------------------------

    print("\n====================================")
    print("STEP 2: CROSS-EVENT DOCUMENT EVIDENCE")
    print("====================================")

    print("Context rows processed:", len(records))

    print(
        "Different question events:",
        len({
            r["event_id"]
            for r in records
        })
    )

    print(
        "Different source documents:",
        len(by_document)
    )

    print(
        "Event-document pairs:",
        len(by_event_document)
    )

    print("\nSample document evidence:")

    for row in document_rows[:5]:

        print(
            f"{row['document_id']}: "
            f"{row['distinct_event_count']} events, "
            f"{row['total_context_rows']} context rows, "
            f"mean retrieval score "
            f"{row['mean_retrieval_score']:.3f}"
        )

    print("\nOutput 1:", summary_path)
    print("Output 2:", detail_path)

    print(
        "\nNo poison labels or correct answers "
        "were used."
    )

    print(
        "Document frequency alone does not "
        "prove poisoning responsibility."
    )


if __name__ == "__main__":
    main()
