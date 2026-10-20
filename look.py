import json
data = json.load(open("attack_feedback/NQ/PRAGB/k5_m5_e5_gpt-4o-mini.json", encoding="utf-8"))
print("Number of complaint events:", len(data))
e = data[0]
print("Question:      ", e["question"])
print("Correct answer:", e["correct_answer"])
print("Target (lie):  ", e["target_answer"])
print("RAG said:      ", e["RAG_response"])
print("Passages:      ", len(e["context_texts"]))
print("Poisoned among top 10:", e["context_labels"][:10])
print("First passage: ", e["context_texts"][0][:200])