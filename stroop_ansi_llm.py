#!/usr/bin/env python3
"""
stroop_ansi_llm.py — Test de Stroop ANSI pour LLMs locaux via LM Studio
Inspiré de : Patel, Wang, Fan — "Deficient executive control in transformer attention"
             PNAS Nexus, juin 2026

Principe : injecter des codes ANSI dans le prompt textuel.
Le LLM voit les tokens d'échappement — pas une couleur rendue.
La question : nomme la couleur ANSI, pas le mot.

Modèles testés (LM Studio, CPU-only, temperature=0.0) :
  gemma   : google/gemma-3n-e4b
  llama   : llama-3.2-3b-instruct
  mistral : mistralai/mistral-7b-instruct-v0.3

Usage : python3 stroop_ansi_llm.py [--host localhost] [--port 1234] [--output file.csv]
        Relancer avec le même --output reprend là où on s'est arrêté.

        python stroop_ansi_llm-paperversion.py --variant c3 --output stroop_c3-papervesionC3.csv --timeout 600
"""

import json
import csv
import os
import time
import argparse
import itertools
from datetime import datetime
import urllib.request
import urllib.error

# ── Palette ANSI ─────────────────────────────────────────────────────────────
ANSI_COLORS = {
    "rouge":      "\033[31m",
    "vert":       "\033[32m",
    "bleu":       "\033[34m",
    "jaune":      "\033[33m",
    "magenta":    "\033[35m",
    "cyan":       "\033[36m",
    "chartreuse": "\033[38;5;76m",   # couleur 76 — préférée d'Alessandra
}
RESET = "\033[0m"
COLOR_NAMES = list(ANSI_COLORS.keys())

# ── Paires Stroop ─────────────────────────────────────────────────────────────

def make_congruent_pair(color_name):
    return {"word": color_name.upper(), "ink": color_name,
            "ansi_code": ANSI_COLORS[color_name],
            "condition": "congruent", "correct_answer": color_name}

def make_incongruent_pair(word_color, ink_color):
    return {"word": word_color.upper(), "ink": ink_color,
            "ansi_code": ANSI_COLORS[ink_color],
            "condition": "incongruent", "correct_answer": ink_color}

def build_stroop_pool():
    pool = [make_congruent_pair(c) for c in COLOR_NAMES]
    pool += [make_incongruent_pair(w, i)
             for w, i in itertools.permutations(COLOR_NAMES, 2)]
    return pool

def build_list(pool, n, condition):
    import random
    random.seed(42)
    congruents   = [p for p in pool if p["condition"] == "congruent"]
    incongruents = [p for p in pool if p["condition"] == "incongruent"]
    if condition == "congruent":
        return random.choices(congruents, k=n)
    elif condition == "incongruent":
        return random.choices(incongruents, k=n)
    else:  # mixed
        half = n // 2
        src = random.choices(congruents, k=half) + random.choices(incongruents, k=n-half)
        random.shuffle(src)
        return src[:n]

# ── Prompts ───────────────────────────────────────────────────────────────────

ANSI_ECHO = {
    "rouge":      r"\e[31m",
    "vert":       r"\e[32m",
    "bleu":       r"\e[34m",
    "jaune":      r"\e[33m",
    "magenta":    r"\e[35m",
    "cyan":       r"\e[36m",
    "chartreuse": r"\e[38;5;76m",
}

def build_prompt(items, variant="ansi", model_key=""):
    """
    variant='ansi' : codes ANSI bruts + table (variante A)
    variant='echo' : style bash echo -e (variante B)
    variant='c3'   : echo + avertissement interférence + reasoning (variante C3)
    """
    if variant == "echo":
        return _build_prompt_echo(items)
    if variant == "c3":
        return _build_prompt_c3(items, model_key)
    return _build_prompt_ansi(items)

def _build_prompt_ansi(items):
    lines = []
    for i, item in enumerate(items, 1):
        token = f"{item['ansi_code']}{item['word']}{RESET}"
        lines.append(f"{i}. {token}  [ANSI code: {repr(item['ansi_code'])[1:-1]}]")
    items_block = "\n".join(lines)

    return f"""You are performing a Stroop color-word task.

ANSI color codes used in this test:
- \\033[31m = rouge (red)
- \\033[32m = vert (green)
- \\033[33m = jaune (yellow)
- \\033[34m = bleu (blue)
- \\033[35m = magenta
- \\033[36m = cyan
- \\033[38;5;76m = chartreuse (bright yellow-green)

INSTRUCTION: For each item below, identify the INK COLOR (the ANSI color code applied), NOT the word written.

Items:
{items_block}

Respond ONLY with strict JSON — no text before or after:
{{
  "answers": [
    {{"item": 1, "word": "<the written word>", "ink_color": "<the ANSI ink color name>"}},
    {{"item": 2, "word": "<the written word>", "ink_color": "<the ANSI ink color name>"}},
    ...
  ]
}}

Use only these color names: rouge, vert, bleu, jaune, magenta, cyan, chartreuse"""

def _build_prompt_echo(items):
    """
    Variante B — chaque item est présenté comme une commande bash echo -e.
    Hypothèse : le contexte 'echo -e' active la sémantique terminal dans les poids,
    rendant le code couleur plus saillant que le mot écrit.
    """
    lines = []
    for i, item in enumerate(items, 1):
        echo_code = ANSI_ECHO[item["ink"]]
        lines.append(f'{i}. echo -e "{echo_code}{item["word"]}\\e[0m"')
    items_block = "\n".join(lines)

    return f"""You are performing a Stroop color-word task using bash terminal color codes.

Each item below is a bash echo -e command that prints a word in a specific ink color.
The \\e[Xm codes set the terminal color. Reference:
- \\e[31m = rouge (red)
- \\e[32m = vert (green)
- \\e[33m = jaune (yellow)
- \\e[34m = bleu (blue)
- \\e[35m = magenta
- \\e[36m = cyan
- \\e[38;5;76m = chartreuse (bright yellow-green)
- \\e[0m  = reset (end of color)

INSTRUCTION: For each echo command, identify the INK COLOR (the \\e[Xm code), NOT the word printed.

Items:
{items_block}

Respond ONLY with strict JSON — no text before or after:
{{
  "answers": [
    {{"item": 1, "word": "<the printed word>", "ink_color": "<the ink color name>"}},
    {{"item": 2, "word": "<the printed word>", "ink_color": "<the ink color name>"}},
    ...
  ]
}}

Use only these color names: rouge, vert, bleu, jaune, magenta, cyan, chartreuse"""

def _build_prompt_c3(items, model_key=""):
    """
    Variante C3 — echo -e + avertissement interférence Stroop explicite + reasoning.
    Hypothèse : nommer le mécanisme de conflit ET exiger un raisonnement avant
    la réponse compense partiellement l'absence de conflict-monitoring architectural.
    L'ordre dans le JSON est délibéré : reasoning AVANT ink_color.

    Compatibility note: Mistral-7B-Instruct-v0.3 produces multi-line reasoning
    that breaks JSON string parsing. For Mistral only, we append a single-line
    format reminder. The reasoning content and task conditions remain identical.
    """
    lines = []
    for i, item in enumerate(items, 1):
        echo_code = ANSI_ECHO[item["ink"]]
        lines.append(f'{i}. echo -e "{echo_code}{item["word"]}\\e[0m"')
    items_block = "\n".join(lines)

    # Format reminder pour Mistral uniquement
    mistral_note = ""
    if model_key == "mistral":
        mistral_note = "\nCOMPATIBILITY NOTE: Keep your reasoning on a single line — no line breaks inside JSON string values."

    return f"""You are performing a Stroop color-word task using bash terminal color codes.

ANSI color reference:
- \\e[31m = rouge (red)
- \\e[32m = vert (green)
- \\e[33m = jaune (yellow)
- \\e[34m = bleu (blue)
- \\e[35m = magenta
- \\e[36m = cyan
- \\e[38;5;76m = chartreuse (bright yellow-green)
- \\e[0m  = reset

WARNING: This task contains a deliberate cognitive interference effect known as the Stroop effect.
The written word will automatically pull your attention toward its own meaning.
This is a TRAP. The word is IRRELEVANT.
Your ONLY source of truth is the \\e[Xm color code BEFORE the word.
Resist the interference. Focus exclusively on the ANSI code.

INSTRUCTION: For each echo command, identify the INK COLOR (the \\e[Xm code), NOT the word.

For each item, first explain your reasoning (which code you see and what color it maps to),
Items:
{items_block}

then give your answer. Respond ONLY with strict JSON — no text before or after:{mistral_note}
{{
  "answers": [
    {{"item": 1, "word": "<the printed word>", "reasoning": "<which \\e[Xm code you see and why it maps to that color>", "ink_color": "<the ink color name>"}},
    {{"item": 2, "word": "<the printed word>", "reasoning": "<which \\e[Xm code you see and why it maps to that color>", "ink_color": "<the ink color name>"}},
    ...
  ]
}}

Use only these color names: rouge, vert, bleu, jaune, magenta, cyan, chartreuse"""

LM_STUDIO_MODELS = {
    "gemma":   "google/gemma-3n-e4b",
    "llama":   "llama-3.2-3b-instruct",
    "mistral": "mistralai/mistral-7b-instruct-v0.3",
}

def call_lmstudio(prompt, model_id, host, port, timeout=600):
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {"model": model_id,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.0, "max_tokens": 4096, "stream": False}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def extract_content(api_response):
    return api_response["choices"][0]["message"]["content"].strip()

def parse_model_response(raw):
    cleaned = raw.strip()
    # Nettoyer les blocs markdown ```json ... ```
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    # \e[ dans les champs reasoning casse le parser JSON (\e n'est pas valide en JSON)
    # On remplace \e[ par \\e[ pour que JSON l'accepte comme string littérale
    cleaned = cleaned.replace("\\e[", "\\\\e[")
    try:
        return json.loads(cleaned).get("answers", [])
    except json.JSONDecodeError:
        return []

# ── Évaluation ────────────────────────────────────────────────────────────────

VALID_COLORS = set(ANSI_COLORS.keys())

def normalize_color(name):
    if not name:
        return ""
    n = name.lower().strip().rstrip(".")
    aliases = {"red": "rouge", "green": "vert", "blue": "bleu",
               "yellow": "jaune", "yellow-green": "chartreuse",
               "bright yellow-green": "chartreuse",
               "vert chartreuse": "chartreuse"}
    return aliases.get(n, n)

def evaluate_answers(items, answers):
    n = len(items)
    if not answers:
        return {"total": n, "correct": 0, "accuracy": 0.0,
                "word_read_errors": 0, "invalid_errors": 0, "parse_failed": True}
    correct = word_read_errors = invalid_errors = 0
    for i, item in enumerate(items):
        if i >= len(answers):
            invalid_errors += 1
            continue
        given    = normalize_color(answers[i].get("ink_color", ""))
        expected = item["correct_answer"]
        word     = item["word"].lower()
        if given == expected:
            correct += 1
        elif given == word:
            word_read_errors += 1   # interférence Stroop
        elif given not in VALID_COLORS:
            invalid_errors += 1
    return {"total": n, "correct": correct,
            "accuracy": round(correct / n * 100, 1),
            "word_read_errors": word_read_errors,
            "invalid_errors": invalid_errors, "parse_failed": False}

# ── Reprise CSV ───────────────────────────────────────────────────────────────

def load_existing_results(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["length"] = int(r["length"])
        r["accuracy"] = float(r["accuracy"])
    print(f"  ↺ Reprise — {len(rows)} résultats déjà chargés depuis {csv_path}\n")
    return rows

def already_done(results, model_key, length, condition):
    return any(r["model_key"] == model_key
               and r["length"] == length
               and r["condition"] == condition
               for r in results)

def save_results(results, csv_path):
    if not results:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

# ── Expérience ────────────────────────────────────────────────────────────────

LENGTHS    = [1, 5, 10, 20]
CONDITIONS = ["congruent", "incongruent", "mixed"]

def save_reasonings(path, timestamp, model_key, length, condition, items, answers):
    """
    Sauvegarde les reasonings C3 dans un fichier JSONL — un objet JSON par ligne.
    Format : {timestamp, model, length, condition, item, word, correct_answer,
              reasoning, ink_color, correct}
    Append mode — accumule au fil des appels.
    """
    with open(path, "a", encoding="utf-8") as f:
        for i, item in enumerate(items):
            if i >= len(answers):
                break
            ans = answers[i]
            correct = normalize_color(ans.get("ink_color", "")) == item["correct_answer"]
            record = {
                "timestamp":      timestamp,
                "model":          model_key,
                "length":         length,
                "condition":      condition,
                "item":           i + 1,
                "word":           item["word"],
                "correct_answer": item["correct_answer"],
                "reasoning":      ans.get("reasoning", ""),
                "ink_color":      ans.get("ink_color", ""),
                "correct":        correct,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def run_experiment(host, port, output_csv, debug=False, variant="ansi", timeout=600):
    pool           = build_stroop_pool()
    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    results        = load_existing_results(output_csv)
    reasoning_path = output_csv.replace(".csv", "_reasonings.jsonl")

    variant_label = ("A — codes ANSI bruts" if variant == "ansi" else
                     "B — style bash echo -e" if variant == "echo" else
                     "C3 — echo + interférence nommée + reasoning")
    print(f"\n{'='*70}")
    print(f"  TEST DE STROOP ANSI — LLMs locaux via LM Studio")
    print(f"  Inspiré de Patel, Wang & Fan — PNAS Nexus 2026")
    print(f"  Variante {variant_label}")
    print(f"  {timestamp}")
    print(f"{'='*70}\n")

    for model_key, model_id in LM_STUDIO_MODELS.items():
        print(f"\n{'─'*70}")
        print(f"  MODÈLE : {model_key}  ({model_id})")
        print(f"{'─'*70}")

        for length in LENGTHS:
            for condition in CONDITIONS:

                if already_done(results, model_key, length, condition):
                    print(f"  {model_key} | n={length:2d} | {condition:11s} → ↺ déjà calculé")
                    continue

                items  = build_list(pool, length, condition)
                prompt = build_prompt(items, variant, model_key)
                label  = f"{model_key} | n={length:2d} | {condition:11s}"
                print(f"  {label} → ", end="", flush=True)

                if debug:
                    print(f"\n\n{'·'*70}")
                    print(f"  PROMPT ({label})")
                    print(f"{'·'*70}")
                    print(prompt)
                    print(f"{'·'*70}\n")

                t0 = time.time()
                try:
                    api_resp = call_lmstudio(prompt, model_id, host, port, timeout)
                    raw      = extract_content(api_resp)

                    if debug:
                        print(f"\n  RÉPONSE BRUTE :")
                        print(f"{'·'*70}")
                        print(raw)
                        print(f"{'·'*70}\n  → ", end="")

                    answers  = parse_model_response(raw)
                    metrics  = evaluate_answers(items, answers)
                    elapsed  = round(time.time() - t0, 1)

                    # Sauvegarder les reasonings si variant C3
                    if variant == "c3" and answers:
                        save_reasonings(
                            reasoning_path, timestamp, model_key,
                            length, condition, items, answers
                        )

                    acc = metrics["accuracy"]
                    col = ("\033[32m" if acc >= 80 else
                           "\033[33m" if acc >= 50 else "\033[31m")
                    print(f"{col}{acc:5.1f}%{RESET}  "
                          f"({metrics['correct']}/{metrics['total']})  "
                          f"interférences={metrics['word_read_errors']}  "
                          f"invalides={metrics['invalid_errors']}  [{elapsed}s]"
                          + ("  ⚠ parse_failed" if metrics["parse_failed"] else ""))

                except (urllib.error.URLError, TimeoutError) as e:
                    print(f"\033[31mTIMEOUT / ERREUR\033[0m ({e})")
                    metrics = {"total": length, "correct": 0, "accuracy": -1.0,
                               "word_read_errors": 0, "invalid_errors": 0,
                               "parse_failed": True}
                    elapsed = 0.0

                results.append({
                    "timestamp": timestamp,
                    "variant": variant,
                    "model_key": model_key, "model_id": model_id,
                    "length": length, "condition": condition,
                    "accuracy": metrics["accuracy"],
                    "correct": metrics["correct"], "total": metrics["total"],
                    "word_read_errors": metrics["word_read_errors"],
                    "invalid_errors": metrics["invalid_errors"],
                    "parse_failed": metrics["parse_failed"],
                    "elapsed_s": elapsed,
                })
                # Sauvegarde incrémentale après chaque appel
                save_results(results, output_csv)

    # ── Récapitulatif ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  RÉCAPITULATIF — Accuracy (%) par modèle × longueur × condition")
    print(f"{'='*70}")
    print(f"  {'Modèle':<10} {'Condition':<12} {'n=1':>6} {'n=5':>6} {'n=10':>6} {'n=20':>6}")
    print(f"  {'─'*54}")
    for model_key in LM_STUDIO_MODELS:
        for condition in CONDITIONS:
            vals = []
            for length in LENGTHS:
                m = next((r for r in results
                          if r["model_key"] == model_key
                          and r["length"] == length
                          and r["condition"] == condition), None)
                vals.append(f"{m['accuracy']:5.1f}" if m and m["accuracy"] >= 0 else "  N/A")
            print(f"  {model_key:<10} {condition:<12} {vals[0]:>6} {vals[1]:>6} {vals[2]:>6}")
        print()

    print(f"  Résultats → {output_csv}")
    print(f"\n  Note : 'interférences' = modèle a nommé le MOT plutôt que la couleur ANSI.\n")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test de Stroop ANSI — LLMs locaux")
    parser.add_argument("--host",    default="localhost")
    parser.add_argument("--port",    type=int, default=1234)
    parser.add_argument("--output",  default=f"stroop_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    parser.add_argument("--debug",   action="store_true",
                        help="Affiche le prompt envoyé et la réponse brute")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Timeout en secondes par appel (défaut: 600)")
    parser.add_argument("--variant", choices=["ansi", "echo", "c3"], default="ansi",
                        help="Variante A=ansi (défaut), B=echo, C3=echo+interference+reasoning")
    args = parser.parse_args()
    run_experiment(args.host, args.port, args.output,
                   debug=args.debug, variant=args.variant,
                   timeout=args.timeout)
