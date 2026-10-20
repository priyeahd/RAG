
"""CrossEventRAG Step 3A: Analyze every incorrect-answer event.

Run inside CrosseventRAG:
    python analyze_event_evidence.py

Inputs (inside data/):
    CrossEventRAG_1000Q_10Contexts_10000_Master.csv
    CrossEventRAG_1000Q_10Contexts_10000_Detector.csv

The Master file supplies correct_answer ONLY for identifying failed events
and comparing passage text with the expected answer. All passage evidence
comes from Detector.csv. No poisoning labels or source manifest are used.

IMPORTANT: Passage-text matching is not proof of poisoning or causal
responsibility. RAG responses and retrieval scores here are simulated.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

MASTER_FILE = "CrossEventRAG_1000Q_10Contexts_10000_Master.csv"
DETECTOR_FILE = "CrossEventRAG_1000Q_10Contexts_10000_Detector.csv"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "the", "to", "was", "were", "with",
}


# -----------------------------------
# STEP A: Text comparison functions
# -----------------------------------

def words(text):
    """Convert text into lowercase words and numbers."""
    return re.findall(r"[a-z0-9]+", text.casefold())


def normalized(text):
    """Compare answers using Step 1's basic normalization."""
    return " ".join(text.strip().casefold().split())


def exact_mention(answer, passage):
    """Check whether the passage contains the entire answer phrase."""

    answer_words = words(answer)
    passage_words = words(passage)

    if not answer_words:
        return False

    if len(answer_words) == 1 and answer_words[0] in STOPWORDS:
        return False

    length = len(answer_words)

    return any(
        passage_words[i:i + length] == answer_words
        for i in range(len(passage_words) - length + 1)
    )


def token_coverage(answer, passage):
    """Calculate lexical overlap, NOT semantic support."""

    answer_terms = set(words(answer)) - STOPWORDS
    passage_terms = set(words(passage))

    if not answer_terms:
        return 0.0

    overlap = len(answer_terms & passage_terms)

    return round(overlap / len(answer_terms), 4)


# -----------------------------------
# STEP B: Read and write CSV files
# -----------------------------------

def read_csv(path, required):

    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as fh:

        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []

        missing = set(required) - set(headers)

        if missing:
            raise ValueError(
                f"Missing columns in {path.name}: {sorted(missing)}"
            )

        return headers, list(reader)


def save_csv(path, fieldnames, rows):

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as fh:

        writer = csv.DictWriter(fh, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------
# STEP C: Main program
# -----------------------------------

def main():

    base = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="CrossEventRAG Step 3A"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=base / "data"
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=base / "results"
    )

    args = parser.parse_args()

    print("Reading dataset...", flush=True)

    # Read Master and Detector.
    # Master is used only for question/answer metadata.
    # Poisoning labels are never accessed.

    _, master_rows = read_csv(
        args.data_dir / MASTER_FILE,
        ["event_id", "question", "correct_answer", "RAG_response"]
    )

    detector_headers, detector_rows = read_csv(
        args.data_dir / DETECTOR_FILE,
        [
            "event_id",
            "question",
            "RAG_response",
            "context_position",
            "context_text",
            "retrieval_score",
            "document_id"
        ]
    )

    forbidden = {
        "correct_answer",
        "ground_truth_label"
    } & set(detector_headers)

    if forbidden:
        raise ValueError(
            f"Detector CSV leaks hidden fields: {sorted(forbidden)}"
        )

    # -----------------------------------
    # STEP D: Find all incorrect answers
    # -----------------------------------

    events = {}
    master_counts = defaultdict(int)

    for row in master_rows:

        event = row["event_id"].strip()

        metadata = (
            row["question"],
            row["correct_answer"],
            row["RAG_response"]
        )

        if not event or any(not s.strip() for s in metadata):
            raise ValueError("Missing event/question/answer in Master")

        if event in events and events[event] != metadata:
            raise ValueError(
                f"Conflicting answers for {event} in Master"
            )

        events[event] = metadata

        master_counts[event] += 1

    if any(n != 10 for n in master_counts.values()):
        raise ValueError(
            "Expected exactly 10 Master rows per event"
        )

    # -----------------------------------
    # STEP E: Load every retrieved context
    # -----------------------------------

    contexts_by_event = defaultdict(list)

    for row in detector_rows:

        event = row["event_id"].strip()

        if event not in events:
            raise ValueError(
                f"Unknown detector event ID: {event}"
            )

        question, _correct, response = events[event]

        if (
            row["question"] != question
            or row["RAG_response"] != response
        ):
            raise ValueError(
                f"Detector/Master mismatch for {event}"
            )

        if (
            not row["document_id"].strip()
            or not row["context_text"].strip()
        ):
            raise ValueError(
                f"Empty document ID or context text in {event}"
            )

        position = int(row["context_position"])
        score = float(row["retrieval_score"])

        if not 0 <= score <= 1:
            raise ValueError(
                f"Out-of-range retrieval score for {event}"
            )

        contexts_by_event[event].append({
            "document_id": row["document_id"],
            "context_position": position,
            "context_text": row["context_text"],
            "retrieval_score": score,
        })

    if set(contexts_by_event) != set(events):
        raise ValueError(
            "Master and Detector do not contain identical events"
        )

    for event, rows in contexts_by_event.items():

        positions = sorted(
            r["context_position"] for r in rows
        )

        if positions != list(range(10)):
            raise ValueError(
                f"Expected context positions 0 to 9 in {event}"
            )

    incorrect_events = sorted(
        event
        for event, (_question, correct, response) in events.items()
        if normalized(correct) != normalized(response)
    )

    # -----------------------------------
    # STEP F: Analyze the 10 contexts
    # for every incorrect-answer event
    # -----------------------------------

    event_report_rows = []
    context_evidence_rows = []
    event_document_rows = []

    for event in incorrect_events:

        question, correct, wrong_response = events[event]

        contexts = sorted(
            contexts_by_event[event],
            key=lambda r: r["context_position"]
        )

        docs = defaultdict(list)
        event_evidence = []

        for context in contexts:

            text = context["context_text"]

            wrong_exact = int(
                exact_mention(wrong_response, text)
            )

            correct_exact = int(
                exact_mention(correct, text)
            )

            wrong_overlap = token_coverage(
                wrong_response, text
            )

            correct_overlap = token_coverage(
                correct, text
            )

            question_overlap = token_coverage(
                question, text
            )

            # These tags indicate text mentions only.
            # They DO NOT prove that a passage caused an error.

            if wrong_exact and correct_exact:
                tag = "both_answers_mentioned"

            elif wrong_exact:
                tag = "wrong_answer_mentioned_only"

            elif correct_exact:
                tag = "correct_answer_mentioned_only"

            else:
                tag = "neither_exact_answer_mentioned"

            evidence = {
                "event_id": event,
                "question": question,
                "correct_answer": correct,
                "RAG_response": wrong_response,
                "document_id": context["document_id"],
                "context_position": context["context_position"],
                "context_text": text,
                "retrieval_score": context["retrieval_score"],
                "wrong_answer_exact_mention": wrong_exact,
                "correct_answer_exact_mention": correct_exact,
                "wrong_answer_token_coverage": wrong_overlap,
                "correct_answer_token_coverage": correct_overlap,
                "question_token_coverage": question_overlap,
                "evidence_tag": tag,
            }

            context_evidence_rows.append(evidence)
            event_evidence.append(evidence)

            docs[context["document_id"]].append(evidence)

        # -----------------------------------
        # STEP G: Group evidence by document
        # within this particular event
        # -----------------------------------

        for document_id, rows in sorted(docs.items()):

            event_document_rows.append({
                "event_id": event,
                "document_id": document_id,
                "context_count": len(rows),

                "context_positions": ";".join(
                    str(r["context_position"])
                    for r in rows
                ),

                "wrong_answer_exact_mentions": sum(
                    r["wrong_answer_exact_mention"]
                    for r in rows
                ),

                "correct_answer_exact_mentions": sum(
                    r["correct_answer_exact_mention"]
                    for r in rows
                ),

                "max_wrong_answer_token_coverage": max(
                    r["wrong_answer_token_coverage"]
                    for r in rows
                ),

                "mean_wrong_answer_token_coverage": round(
                    mean(
                        r["wrong_answer_token_coverage"]
                        for r in rows
                    ), 4
                ),

                "max_correct_answer_token_coverage": max(
                    r["correct_answer_token_coverage"]
                    for r in rows
                ),

                "max_retrieval_score": max(
                    r["retrieval_score"]
                    for r in rows
                ),
            })

        # -----------------------------------
        # STEP H: Prepare event-level report
        # -----------------------------------

        event_report_rows.append({
            "event_id": event,
            "question": question,
            "correct_answer": correct,
            "RAG_response": wrong_response,

            "retrieved_context_count": len(contexts),
            "distinct_document_count": len(docs),

            "document_ids": ";".join(sorted(docs)),

            "contexts_mentioning_wrong_answer": sum(
                r["wrong_answer_exact_mention"]
                for r in event_evidence
            ),

            "contexts_mentioning_correct_answer": sum(
                r["correct_answer_exact_mention"]
                for r in event_evidence
            ),

            "max_wrong_answer_token_coverage": max(
                r["wrong_answer_token_coverage"]
                for r in event_evidence
            ),
        })

    # -----------------------------------
    # STEP I: Save the three output files
    # -----------------------------------

    output_files = [
        (
            "step3_misgeneration_reports.csv",
            event_report_rows
        ),
        (
            "step3_event_context_evidence.csv",
            context_evidence_rows
        ),
        (
            "step3_event_document_evidence.csv",
            event_document_rows
        ),
    ]

    for filename, data in output_files:

        if not data:
            raise ValueError(
                f"No data generated for {filename}"
            )

        save_csv(
            args.results_dir / filename,
            list(data[0].keys()),
            data
        )

    # -----------------------------------
    # STEP J: Display results
    # -----------------------------------

    print("\n====== STEP 3A: EVENT-LEVEL EVIDENCE ======")

    print("All events checked:", len(events))

    print(
        "Incorrect-answer events:",
        len(incorrect_events)
    )

    print(
        "Retrieved contexts in incorrect events:",
        len(context_evidence_rows)
    )

    print(
        "Event-document connections in incorrect events:",
        len(event_document_rows)
    )

    print(
        "Exact wrong-answer mentions:",
        sum(
            r["wrong_answer_exact_mention"]
            for r in context_evidence_rows
        )
    )

    print(
        "Exact correct-answer mentions:",
        sum(
            r["correct_answer_exact_mention"]
            for r in context_evidence_rows
        )
    )

    print("\nFiles created:")

    for filename, _ in output_files:
        print("  ", args.results_dir / filename)

    print(
        "\nIMPORTANT: This is lexical evidence, "
        "NOT a poisoning detection result."
    )

    print(
        "Poisoning labels were not accessed or used; "
        "no manifest was read."
    )

    print(
        "Responses and retrieval scores in this "
        "dataset are simulated."
    )


if __name__ == "__main__":
    main()
