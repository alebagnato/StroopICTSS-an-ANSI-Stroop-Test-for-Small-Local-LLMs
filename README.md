# StroopICTSS-an-ANSI-Stroop-Test-for-Small-Local-LLMs
 This repository implements a novel adaptation of the classical Stroop color-word paradigm for local LLMs. Instead of colored ink (impossible in a text-only context), ANSI terminal escape codes are injected into prompts as the "ink color" signal, while the written color name acts as the competing word.  


> *Probing inhibitory control in small local language models via ANSI terminal color codes*

Companion code for the paper Probing inhibitory control in small local language models via ANSI terminal color code submitted to **ICTSS 2026 (38th International Conference on Testing Software and Systems)**.

Inspired by: Patel, Wang & Fan — *"Deficient executive control in transformer attention"* — PNAS Nexus, June 2026  
DOI: [10.1093/pnasnexus/pgag149](https://doi.org/10.1093/pnasnexus/pgag149) · Preprint: [bioRxiv 10.1101/2025.01.22.634394v2](https://www.biorxiv.org/content/10.1101/2025.01.22.634394v2)

---

## Overview

This repository implements a novel adaptation of the classical Stroop color-word paradigm for local LLMs. Instead of colored ink (impossible in a text-only context), ANSI terminal escape codes are injected into prompts as the "ink color" signal, while the written color name acts as the competing word.

**All experiments were conducted on a consumer laptop, CPU-only, no GPU required.**

### Four experimental variants

| Variant | Description | Key question |
|---|---|---|
| **A** — Raw ANSI | `\x1b[34m` tokens + reference table | Do models read ANSI codes at all? |
| **B** — echo -e | `echo -e "\e[34mWORD\e[0m"` bash style | Does syntactic context activate ANSI semantics? |
| **Liguria** | Variant B + rare 256-color palette | Is failure driven by word frequency in corpus? |
| **C3** | Variant B + named interference + chain-of-thought | Can language compensate for missing architecture? |

### Models tested

| Alias | Model | Size | Format |
|---|---|---|---|
| gemma | google/gemma-3n-e4b | 6.9B | GGUF Q4_K_M |
| llama | llama-3.2-3b-instruct | 3B | GGUF Q4_K_M |
| mistral | mistralai/mistral-7b-instruct-v0.3 | 7B | GGUF Q4_K_M |

---

## Requirements

### Python

```
Python >= 3.10
matplotlib >= 3.7
```

No other dependencies — all other modules (`json`, `csv`, `urllib`, `argparse`, `random`, `itertools`, `collections`) are from the Python standard library.

```bash
pip install matplotlib
```

### LM Studio

Download and install [LM Studio](https://lmstudio.ai/) (free, cross-platform).

1. Download the three models via the LM Studio model browser:
   - `google/gemma-3n-e4b` (GGUF Q4_K_M)
   - `llama-3.2-3b-instruct` (GGUF Q4_K_M)
   - `mistralai/mistral-7b-instruct-v0.3` (GGUF Q4_K_M)

2. Start the local server: **Developer tab → Start Server** (default port: 1234)

3. Load one model at a time before running experiments.

> **Note:** The experiment runs models sequentially. You must manually switch the loaded model in LM Studio between the three model blocks, or run each model separately using `--output` with resumption (see Usage below).

---

## Repository Structure

```
StroopICTSS/
├── stroop_ansi_llm.py          # Main experiment script
├── stroop_plot.py              # Result visualization (3 figures)
├── stroop_analyze_reasonings.py # Qualitative analysis of C3 reasonings
├── data/
│   ├── stroop_results_20260625_114157.csv   # Variant A results
│   ├── stroop_echo.csv                       # Variant B results
│   ├── stroop_ligurian.csv                   # Liguria variant results
│   ├── stroop_c3-R.csv                       # C3 variant results
│   └── stroop_c3-R_reasonings.jsonl          # 189 archived reasonings
├── figures/
│   ├── stroop_1_accuracy.png
│   ├── stroop_2_interference.png
│   └── stroop_3_heatmap.png
└── README.md
```

---

## Usage

### Main experiment script

```bash
# Variant A — raw ANSI (default)
python stroop_ansi_llm.py --output stroop_results.csv

# Variant B — bash echo -e style
python stroop_ansi_llm.py --variant echo --output stroop_echo.csv

# Liguria variant — rare 256-color palette
# (edit ANSI_COLORS and ANSI_ECHO in the script first — see Configuration)
python stroop_ansi_llm.py --variant echo --output stroop_ligurian.csv

# Variant C3 — named interference + chain-of-thought reasoning
python stroop_ansi_llm.py --variant c3 --output stroop_c3.csv

# With debug output (prints prompt + raw model response for each call)
python stroop_ansi_llm.py --variant c3 --debug --output stroop_c3_debug.csv

# With extended timeout (recommended for C3 on CPU)
python stroop_ansi_llm.py --variant c3 --timeout 600 --output stroop_c3.csv
```

### All CLI options

```
--host      LM Studio host        (default: localhost)
--port      LM Studio port        (default: 1234)
--output    Output CSV file       (default: stroop_results_TIMESTAMP.csv)
--variant   ansi | echo | c3      (default: ansi)
--timeout   Timeout per call (s)  (default: 600)
--debug     Print prompt + raw response for each call
```

### Resumption

The script saves results after **every API call**. If interrupted, simply rerun with the same `--output` file — already completed runs are skipped automatically.

```bash
# Resume an interrupted run
python stroop_ansi_llm.py --variant c3 --output stroop_c3.csv --timeout 600
```

> **Note:** Rows with `parse_failed=True` or `accuracy=-1.0` (timeouts) are considered complete and will **not** be retried automatically. Delete those rows from the CSV before resuming if you want them recalculated.

### Visualization

```bash
# Generate 3 figures from a CSV
python stroop_plot.py stroop_results.csv

# Specify output directory
python stroop_plot.py stroop_results.csv ./figures/
```

Produces:
- `stroop_1_accuracy.png` — accuracy curves by condition and length
- `stroop_2_interference.png` — word_read_errors bar charts
- `stroop_3_heatmap.png` — cognitive profile heatmap (red→green)

### Reasoning analysis (C3 only)

```bash
python stroop_analyze_reasonings.py stroop_c3_reasonings.jsonl
```

Outputs per-color TP/FP/FN/TN, precision/recall/F1, top confusions, and reasoning pattern statistics ("However" marker, code-first vs word-first reasoning).

---

## Configuration

### Switching to the Ligurian palette

In `stroop_ansi_llm.py`, replace the `ANSI_COLORS` and `ANSI_ECHO` dictionaries:

```python
ANSI_COLORS = {
    "sea-green":    "\033[38;5;74m",
    "sky-teal":     "\033[38;5;75m",
    "azure-green":  "\033[38;5;76m",
    "aqua-cyan":    "\033[38;5;77m",
    "emerald-cyan": "\033[38;5;78m",
    "lagoon-teal":  "\033[38;5;73m",
    "mint-green":   "\033[38;5;79m",
}

ANSI_ECHO = {
    "sea-green":    r"\e[38;5;74m",
    "sky-teal":     r"\e[38;5;75m",
    "azure-green":  r"\e[38;5;76m",
    "aqua-cyan":    r"\e[38;5;77m",
    "emerald-cyan": r"\e[38;5;78m",
    "lagoon-teal":  r"\e[38;5;73m",
    "mint-green":   r"\e[38;5;79m",
}
```

Also update the reference table strings in `_build_prompt_ansi()`, `_build_prompt_echo()`, and `_build_prompt_c3()`, and the final `Use only these color names:` line.

### Adding a model

In `LM_STUDIO_MODELS`:

```python
LM_STUDIO_MODELS = {
    "gemma":   "google/gemma-3n-e4b",
    "llama":   "llama-3.2-3b-instruct",
    "mistral": "mistralai/mistral-7b-instruct-v0.3",
    "mymodel": "my-model-identifier",   # add here
}
```

---

## Output Format

### CSV columns

| Column | Description |
|---|---|
| timestamp | Run timestamp (YYYYMMDD_HHMMSS) |
| variant | ansi / echo / c3 |
| model_key | gemma / llama / mistral |
| model_id | Full LM Studio model identifier |
| length | List length (1, 5, 10, 20) |
| condition | congruent / incongruent / mixed |
| accuracy | % correct (0–100) |
| correct | Number of correct answers |
| total | Total items |
| word_read_errors | Model named the word instead of ANSI color |
| invalid_errors | Response outside color palette |
| parse_failed | True if JSON parsing failed |
| elapsed_s | Wall time in seconds (-1.0 = timeout) |

### JSONL reasoning records (C3 only)

One JSON object per line:

```json
{
  "timestamp": "20260626_134650",
  "model": "gemma",
  "length": 10,
  "condition": "incongruent",
  "item": 3,
  "word": "VERT",
  "correct_answer": "chartreuse",
  "reasoning": "The code is \\e[38;5;76m, which maps to chartreuse. However...",
  "ink_color": "chartreuse",
  "correct": true
}
```

---

## Reproducibility

- **temperature=0.0** — deterministic outputs
- **random.seed(42)** — identical list construction across runs
- All experiments run on a single consumer laptop (Dell Latitude), CPU-only

> **Limitation:** A single random seed was used throughout. Robustness across multiple seeds was not evaluated and constitutes a known limitation.

---

## Key Findings

1. **Stroop failure is not uniform** — three distinct cognitive signatures across architectures
2. **Syntactic context matters for Llama** — echo-e activates memorized ANSI associations (0%→75%)
3. **Llama knows pairs, not principles** — Ligurian palette collapses echo advantage (75%→5%)
4. **Reasoning is architecture-dependent** — C3 transforms Gemma (40%→100%), overloads Llama, incompatible with Mistral (formatting issues)
5. **The "However" marker** — 12/12 correct in Gemma C3 reasonings; perfect predictor of successful interference resolution
6. **Prompt cannibalism** — at n=1, C3 prompt (~500 tokens) overwhelms a single item; model responds to the prompt itself
7. **Chartreuse: zero false positives** — rare color = maximum precision across all variants

---

## Compatibility Notes

**Mistral-7B-Instruct-v0.3** does not maintain valid JSON output under C3 conditions (verbose reasoning breaks JSON string parsing). A single-line format reminder was tested but did not resolve the issue. Mistral C3 results are reported as inconclusive. This is itself a finding: C3 exposes a format/reasoning tradeoff absent in Gemma and Llama.

---

## Citation

```bibtex
@inproceedings{bagnato2026stroop,
  title     = {ANSI Stroop: Probing Inhibitory Control in Small Local LLMs},
  author    = {Bagnato, Alessandra},
  booktitle = {Proceedings of the 38th International Conference on 
               Testing Software and Systems (ICTSS 2026)},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026},
  note      = {All experiments conducted CPU-only on a consumer laptop}
}
```

---

## License

 GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

---

*"All experiments reported here were conducted on a consumer laptop, CPU-only, no GPU req
