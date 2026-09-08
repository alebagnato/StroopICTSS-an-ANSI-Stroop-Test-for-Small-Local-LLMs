
"""
stroop_analyze_reasonings.py — Analyse qualitative des reasonings C3
Lit le fichier JSONL produit par stroop_ansi_llm.py --variant c3

Usage : python3 stroop_analyze_reasonings.py stroop_c3_reasonings.jsonl
"""

import json
import sys
from collections import defaultdict

# ── Lecture ───────────────────────────────────────────────────────────────────

def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

# ── Analyse faux positifs / faux négatifs ─────────────────────────────────────
#
# Définitions :
#   Faux positif  : le modèle donne ink_color=X alors que correct_answer ≠ X
#                   → la couleur X est *annoncée à tort*
#   Faux négatif  : correct_answer=X mais le modèle ne donne pas X
#                   → la couleur X est *manquée*
#
# Sur items incongruents uniquement — les seuls où l'interférence joue.

def analyze(records):
    # Filtrer incongruents uniquement
    inc = [r for r in records if r["condition"] == "incongruent"]

    fp_by_color   = defaultdict(int)
    fn_by_color   = defaultdict(int)
    tp_by_color   = defaultdict(int)   # correct_answer=X et donné=X
    tn_by_color   = defaultdict(int)   # correct_answer≠X et donné≠X
    total_as_ink  = defaultdict(int)
    total_given   = defaultdict(int)
    confusion     = defaultdict(int)

    all_colors = set()

    for r in inc:
        expected = r["correct_answer"]
        given    = r["ink_color"]
        all_colors.add(expected)
        all_colors.add(given)
        total_as_ink[expected] += 1
        total_given[given]     += 1

        if r["correct"]:
            tp_by_color[expected] += 1
        else:
            fn_by_color[expected] += 1
            fp_by_color[given]    += 1
            confusion[(expected, given)] += 1

    # TN : pour chaque couleur X, items où X n'était pas attendu ET pas donné
    n = len(inc)
    for color in all_colors:
        tp = tp_by_color[color]
        fp = fp_by_color[color]
        fn = fn_by_color[color]
        tn_by_color[color] = n - tp - fp - fn

    return {
        "total_incongruent": n,
        "tp_by_color":   dict(tp_by_color),
        "fp_by_color":   dict(fp_by_color),
        "fn_by_color":   dict(fn_by_color),
        "tn_by_color":   dict(tn_by_color),
        "total_as_ink":  dict(total_as_ink),
        "total_given":   dict(total_given),
        "confusion":     dict(confusion),
        "all_colors":    sorted(all_colors),
    }

def analyze_by_model(records):
    models = sorted(set(r["model"] for r in records))
    return {m: analyze([r for r in records if r["model"] == m]) for m in models}

# ── Affichage ─────────────────────────────────────────────────────────────────

RESET = "\033[0m"
RED   = "\033[31m"
GRN   = "\033[32m"
YLW   = "\033[33m"
BLU   = "\033[34m"
CYN   = "\033[36m"

def bar(n, total, width=20):
    if total == 0:
        return "─" * width
    filled = round(n / total * width)
    return "█" * filled + "░" * (width - filled)

def print_analysis(label, stats):
    n = stats["total_incongruent"]
    if n == 0:
        print(f"  {label} — aucun item incongruent")
        return

    total_errors = sum(stats["fn_by_color"].values())
    print(f"\n  {YLW}{label}{RESET} — {n} items incongruents, {total_errors} erreurs")
    print(f"  {'─'*70}")

    # Tableau TP / FP / FN / TN + précision + rappel + F1
    print(f"\n  {'Couleur':<14} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}  {'Précision':>10} {'Rappel':>8} {'F1':>6}")
    print(f"  {'─'*66}")

    colors = stats["all_colors"]
    for color in colors:
        tp = stats["tp_by_color"].get(color, 0)
        fp = stats["fp_by_color"].get(color, 0)
        fn = stats["fn_by_color"].get(color, 0)
        tn = stats["tn_by_color"].get(color, 0)

        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        col_p = GRN if precision >= 80 else (YLW if precision >= 50 else RED)
        col_r = GRN if recall    >= 80 else (YLW if recall    >= 50 else RED)
        col_f = GRN if f1        >= 80 else (YLW if f1        >= 50 else RED)

        print(f"  {color:<14} {tp:>4} {fp:>4} {fn:>4} {tn:>4}  "
              f"{col_p}{precision:>9.1f}%{RESET} "
              f"{col_r}{recall:>7.1f}%{RESET} "
              f"{col_f}{f1:>5.1f}%{RESET}")

    # Top confusions
    if stats["confusion"]:
        print(f"\n  {CYN}Top confusions{RESET} (attendu → donné) :")
        top = sorted(stats["confusion"].items(), key=lambda x: -x[1])[:5]
        for (expected, given), count in top:
            print(f"    {expected:<14} → {given:<14} × {count}")

def print_global(records):
    stats = analyze(records)
    print(f"\n{'='*64}")
    print(f"  ANALYSE GLOBALE — tous modèles confondus")
    print(f"{'='*64}")
    print_analysis("tous modèles", stats)

def print_by_model(records):
    by_model = analyze_by_model(records)
    print(f"\n{'='*64}")
    print(f"  ANALYSE PAR MODÈLE")
    print(f"{'='*64}")
    for model, stats in by_model.items():
        print_analysis(model, stats)

# ── Analyse reasoning ─────────────────────────────────────────────────────────

def analyze_reasoning_patterns(records):
    """
    Détecte les patterns dans les reasonings :
    - "However" : reconnaissance explicite du conflit
    - commence par le code vs commence par le mot
    - raisonnement post-hoc (mot d'abord, code ensuite)
    """
    inc = [r for r in records if r["condition"] == "incongruent"]

    however_correct = however_wrong = 0
    code_first_correct = code_first_wrong = 0
    word_first_correct = word_first_wrong = 0

    for r in inc:
        reasoning = r.get("reasoning", "").lower()
        correct   = r["correct"]
        word      = r["word"].lower()

        has_however  = "however" in reasoning
        starts_code  = reasoning.strip().startswith("\\e[") or reasoning.strip().startswith("the code")
        starts_word  = reasoning.strip().startswith(f"the word {word}") or \
                       reasoning.strip().startswith(f"{word}")

        if has_however:
            if correct: however_correct += 1
            else:       however_wrong   += 1
        if starts_code:
            if correct: code_first_correct += 1
            else:       code_first_wrong   += 1
        if starts_word:
            if correct: word_first_correct += 1
            else:       word_first_wrong   += 1

    print(f"\n{'='*64}")
    print(f"  PATTERNS DE REASONING (items incongruents)")
    print(f"{'='*64}")
    print(f"\n  {'Pattern':<25} {'Corrects':>9} {'Incorrects':>11} {'Taux':>8}")
    print(f"  {'─'*55}")

    for label, c, w in [
        ("contient 'However'",   however_correct,    however_wrong),
        ("commence par le code", code_first_correct, code_first_wrong),
        ("commence par le mot",  word_first_correct, word_first_wrong),
    ]:
        total = c + w
        rate  = c / total * 100 if total else 0
        col   = GRN if rate >= 70 else (YLW if rate >= 40 else RED)
        print(f"  {label:<25} {c:>9} {w:>11} {col}{rate:>7.1f}%{RESET}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "stroop_c3-R_reasonings.jsonl"

    records = load_jsonl(path)
    print(f"\n  {len(records)} reasonings chargés depuis {path}")
    print(f"  Modèles : {sorted(set(r['model'] for r in records))}")
    print(f"  Corrects globaux : {sum(1 for r in records if r['correct'])} / {len(records)}")

    print_global(records)
    print_by_model(records)
    analyze_reasoning_patterns(records)

    print()
