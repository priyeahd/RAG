Let me also read the README files for more context:



Now let me get diagnostics to check for any issues:



## Code Base Breakdown: **CS224N / RAG-Responsibility-Attribution**

This is an experimental framework for detecting and analyzing poisoning attacks in **Retrieval-Augmented Generation (RAG)** systems. The project consists of two main modules: **`RAGOrigin`** (for identifying poisoned documents) and **`RAGForensics`** (for evaluating poison attribution).

---

## 📦 **Overall Structure**

```
CS224N/RAG-Responsibility-Attribution/
├── README.md                          # Project documentation, citations
├── requirements.txt                    # Dependencies: numpy, openai, scikit_learn, torch, tqdm, transformers
├── RAGOrigin/
│   ├── main.py                        # Main entry point for responsibility attribution (RAGOrigin method)
│   └── src/
│       ├── OpenAI_API.py              # Async wrapper for OpenAI API calls
│       ├── determine_threshold.py     # Dynamic thresholding with K-means clustering
│       ├── measure_responsibility.py  # Compute trace scores using proxy model (Llama-3.1-8B)
│       └── narrow_scope.py            # Scope narrowing via consistency checking
├── RAGForensics/
│   ├── main.py                        # Main entry point for attack evaluation
│   └── OpenAI_API.py                  # (Identical copy of src/OpenAI_API.py)
└── attack_feedback/PRAGB/
    └── k5_m5_e5_gpt-4o-mini.json       # Feedback data: misgeneration events from attacks
```

---

## 🎯 **Core Pipeline**

### **Phase 1: Feedback Collection (`attack_feedback`)**
- **Input Format**: `k5_m5_e5_gpt-4o-mini.json`
  - Contains ~30,000 records (question + answer pairs) from poisoned RAG runs
  - Fields: `question`, `RAG_response`, `context_texts`, `context_labels`, `retrieval_scores`

### **Phase 2: Scope Narrowing (`narrow_scope.py`)**
- Uses **LLM-based consistency checking** to group contexts and find "inconsistent" ones
- **How it works**:
  1. Split contexts into groups of `top_K` (e.g., 5)
  2. Generate answers for each group using a test LLM
  3. Ask a judge LLM if the new answer is consistent with the original RAG response
  4. Continue until balanced number of "True" vs "False" judgments
- **Output**: Saved to `attack_feedback_scope/*.json`

### **Phase 3: Responsibility Measurement (`measure_responsibility.py`)**
- Uses a **proxy model** (Llama-3.1-8B) to compute "trace scores" for each context chunk.
- **Score types** (all normalized):
  - `answer_scores`: loss of generating the RAG response from `(context + question)`
  - `question_scores`: loss of generating the question from `context` alone
  - `retrieval_scores`: original retrieval ranking scores
- **Trace score variants** (`variant_0` through `variant_6`):
  - `variant_0`: average of all three normalized scores
  - `variant_1–3`: individual score types
- **Output**: `scores_iter_result.json`, `metric_result.json`, `context_result.json`

### **Phase 4: Dynamic Thresholding (`determine_threshold.py`)**
- Applies **K-means clustering** to separate "poisoned" (high scores) vs "clean" contexts.
- Produces per-query metrics:
  - **DACC**: Dynamic Accuracy
  - **FPR**: False Positive Rate
  - **FNR**: False Negative Rate
- Used by both `RAGOrigin` and `RAGForensics` to determine which documents are "poisoned."

### **Phase 5: Attack Evaluation (`RAGForensics/main.py`)**
- Reads poisoned feedback from `attack_feedback/PRAGB/*.json`
- For each context chunk, asks a judge LLM whether the context tries to "induce" a consistent answer with the poisoned response.
- Metrics:
  - **TP**: Context correctly flagged as poison when ground truth says so.
  - **FP**: Context incorrectly flagged as poison.
  - **TN**: Context correctly flagged as clean.
  - **FN**: Poisoned context missed by judge.
- Aggregates DACC, FPR, FNR per query and overall.

---

## 🔧 **Implementation Highlights**

### **1. Async API Wrapper (`OpenAI_API.py`)**
```python
class OpenAIGenerator:
    - Batch requests to OpenAI using `asyncio.gather`.
    - Supports custom models via environment variables.
    ```

### **2. Proxy-Based Trace Scoring (`measure_responsibility.py`)**
- Loads Llama-3.1-8B and uses masked token losses to measure how much each context contributed to the RAG response.
- Lower loss = stronger evidence the context supports the answer.

### **3. Scope Narrowing (`narrow_scope.py`)**
- Uses parallel execution (ThreadPoolExecutor) for efficiency.
- Groups contexts into `top_K` chunks and iteratively probes them until finding enough inconsistencies.

### **4. Dynamic Thresholding (`determine_threshold.py`)**
- Clusters trace scores into 2 groups per query using K-means.
- Automatically picks the higher cluster as "positive" (poisoned).

---

## 🧪 **Test Data Structure**

The JSON file `k5_m5_e5_gpt-4o-mini.json` has this schema per record:

```python
{
  "question_id": int,              # Query ID
  "question": str,                 # The user query
  "correct_answer": str,           # Ground truth answer
  "target_answer": str,            # Incorrect (poisoned) answer
  "RAG_response": str,             # Response from poisoned RAG system
  "context_texts": List[str],      # Retrieved context chunks
  "context_labels": List[int],     # Binary: 0=clean, 1=poisoned
  "retrieval_scores": List[float]  # Original retrieval scores
}
```

---

## 🚀 **Quick Start Commands**

### **Run RAGOrigin (Trace and Attribute)**
```bash
python RAGOrigin/main.py \
    --dataset "NQ" \
    --attack_retriever "e5" \
    --attack_LLM "gpt-4o-mini" \
    --judge_LLM "gpt-4o-mini" \
    --attack_method "PRAGB" \
    --attack_M 5 \
    --top_K 5 \
    --trace_method "RAGOrigin" \
    --proxy_model "meta-llama/Llama-3.1-8B" \
    --variant 0 \
    --normalize_method "z_score_normalize" \
    --feedback_root_dir "attack_feedback" \
    --feedback_scope_dir "attack_feedback_scope" \
    --result_root_dir "result" \
    --test_version "v1" \
    --cuda_device 0
```

### **Run RAGForensics (Evaluate Attribution)**
```bash
python RAGForensics/main.py \
    --dataset "NQ" \
    --attack_retriever "e5" \
    --attack_LLM "gpt-4o-mini" \
    --trace_LLM "gpt-4o-mini" \
    --attack_method "PRAGB" \
    --attack_M 5 \
    --top_K 5 \
    --feedback_root_dir "attack_feedback" \
    --result_root_dir "result" \
    --test_version "v1"
```

---

## ⚠️ **Minor Code Quality Issues (Diagnostics)**

| File | Issue | Suggestion |
|------|-------|------------|
| `RAGOrigin/main.py:4` | `transformers` not imported | Add import at top |
| `RAGOrigin/main.py:18` | Suboptimal slicing loop | Use `min(end_index, len(y_pred))` |
| `RAGOrigin/main.py:117` | Unused variable | Prefix with `_` |
| `RAGOrigin/main.py:144/155` | Index-based enumerate | Use direct variable names |
| `RAGOrigin/src/OpenAI_API.py:1` | Unsorted imports | Reorder |

---

## 📊 **Summary by Module**

| Module | Purpose | Key Dependencies |
|--------|---------|------------------|
| `RAGOrigin/main.py` | Orchestrates full attribution pipeline | `measure_responsibility`, `determine_threshold`, `narrow_scope` |
| `RAGOrigin/src/measure_responsibility.py` | Compute trace scores with Llama proxy | `torch`, `transformers` |
| `RAGOrigin/src/determine_threshold.py` | Dynamic thresholding per query | `sklearn.cluster.KMeans`, `sklearn.metrics.confusion_matrix` |
| `RAGOrigin/src/narrow_scope.py` | Scope narrowing via LLM consistency checks | Async `OpenAI_API` |
| `RAGOrigin/src/OpenAI_API.py` | Async batched API calls | `openai.AsyncOpenAI` |
| `RAGForensics/main.py` | Evaluate poisoned context detection | Same `OpenAI_API`, custom judge prompt |

---

Let me know if you want to explore a specific module deeper, add new variants, or expand the scoring methods!