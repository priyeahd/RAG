"""Test ONE stuck passage on its own, to see whether Ollama or the passage is the problem.

Run from the CrosseventRAG folder:
    python test_one_passage.py
    python test_one_passage.py --event evt_0475 --position 2
"""

import argparse
import time
from pathlib import Path

from analyze_passage_support_v3_3 import INPUT_FILE, judge_passage, read_events

parser = argparse.ArgumentParser()
parser.add_argument("--event", default="evt_0475")
parser.add_argument("--position", type=int, default=2)
parser.add_argument("--model", default="qwen3.5:9b")
parser.add_argument("--timeout", type=int, default=120)
args = parser.parse_args()

events = read_events(Path(__file__).resolve().parent / "results" / INPUT_FILE)
rows = events[args.event]

print(f"{args.event}: question = {rows[0]['question']}")
print(f"wrong answer = {rows[0]['RAG_response']}\n")
for r in rows:
    print(f"  position {r['context_position']}: {r['document_id']}  {len(r['context_text'])} characters")

row = next(r for r in rows if int(r["context_position"]) == args.position)
print(f"\nPassage at position {args.position}:\n{row['context_text'][:1500]}\n")

print(f"Sending it to {args.model} (timeout {args.timeout} s)...")
start = time.time()
try:
    result = judge_passage("http://127.0.0.1:11434", args.model, row, args.timeout, False)
    print(f"Answered in {time.time() - start:.1f} s:\n{result}")
except Exception as exc:
    print(f"FAILED after {time.time() - start:.1f} s: {exc}")
