"""Nested Cross-Validation: unverzerrte Gueteschaetzung trotz Modellauswahl.

Aeussere Schleife (Leave-One-Tool-Out): Werkzeug T wird komplett zurueckgelegt.
Innere Schleife: Auf den verbleibenden Werkzeugen wird per LOTO-CV aus einer
kleinen Kandidatenliste die beste Konfiguration gewaehlt (Kriterium:
image_auroc). Diese wird dann einmalig auf Werkzeug T angewendet.

Die ueber alle aeusseren Folds gesammelten Scores ergeben die berichtbare
Gueteschaetzung der GESAMTPROZEDUR (inkl. Auswahl) - genau das beantwortet die
Frage "duerfen wir mit LOTO-CV selektieren, wenn kein Holdout uebrig bleibt".

Kandidaten: bewusst wenige, fachlich begruendete Konfigurationen (--fast fuer
einen CPU-tauglichen Schnelldurchlauf; Standardliste fuer GPU/Colab).

Beispiel:
  python nested_eval.py --fast
  python nested_eval.py --device cuda
"""

import argparse
import json
from pathlib import Path

from cv_core import run_loto_cv, load_manifest, compute_metrics, IMPL
from patchcore_model import PatchCoreConfig

CANDIDATES = [
    dict(backbone="wide_resnet50_2", input_size=384, aug_n=3, use_extras=True,
         score_topk=50),
    dict(backbone="wide_resnet50_2", input_size=384, aug_n=3, use_extras=False,
         score_topk=50),
    dict(backbone="wide_resnet50_2", input_size=384, aug_n=3, use_extras=True,
         grayscale=True, score_topk=50),
    dict(backbone="wide_resnet50_2", input_size=518, aug_n=3, use_extras=True,
         score_topk=50),
    dict(backbone="resnet18", input_size=384, aug_n=3, use_extras=True,
         score_topk=50),
]
CANDIDATES_FAST = [
    dict(backbone="resnet18", input_size=256, aug_n=0, use_extras=True),
    dict(backbone="resnet18", input_size=256, aug_n=0, use_extras=False),
    dict(backbone="resnet18", input_size=256, aug_n=0, use_extras=True,
         grayscale=True),
]


def optuna_top_candidates(n: int) -> list[dict]:
    """Die n besten (unterschiedlichen) Konfigurationen aus der Optuna-Studie."""
    import optuna
    storage = f"sqlite:///{(IMPL / 'results' / 'optuna.db').as_posix()}"
    study = optuna.load_study(study_name="patchcore_search", storage=storage)
    seen, out = set(), []
    for t in sorted(study.trials, key=lambda t: -(t.value or float("-inf"))):
        if t.value is None:
            continue
        key = tuple(sorted(t.params.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(t.params))
        if len(out) >= n:
            break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true", help="kleine CPU-Kandidatenliste")
    p.add_argument("--device", default="cpu")
    p.add_argument("--from-optuna", type=int, default=0, metavar="N",
                   help="zusaetzlich die N besten Optuna-Konfigurationen aufnehmen")
    a = p.parse_args()
    candidates = list(CANDIDATES_FAST if a.fast else CANDIDATES)
    if a.from_optuna > 0:
        extra = optuna_top_candidates(a.from_optuna)
        print(f"Aus Optuna uebernommen: {extra}")
        candidates = extra + [c for c in candidates if c not in extra]

    tools = sorted({r["tool"] for r in load_manifest()
                    if r["source"] == "official_crop"})
    outer_oof, selection_log = [], {}

    for outer in tools:
        inner_tools = [t for t in tools if t != outer]
        print(f"\n=== Aeusserer Fold: {outer} (innere Auswahl auf {inner_tools}) ===")
        best, best_auc = None, -1.0
        for cand in candidates:
            cfg = PatchCoreConfig(**cand, device=a.device)
            m, _ = run_loto_cv(cfg, tools=inner_tools, verbose=False)
            auc = m.get("image_auroc", float("nan"))
            print(f"  Kandidat {cand} -> inner image_auroc={auc:.4f}")
            if auc == auc and auc > best_auc:
                best, best_auc = cand, auc
        selection_log[outer] = {"selected": best, "inner_auroc": best_auc}
        print(f"  -> gewaehlt: {best}")

        cfg = PatchCoreConfig(**best, device=a.device)
        _, oof = run_loto_cv(cfg, tools=[outer], verbose=False)
        outer_oof.extend(oof)

    metrics = compute_metrics(outer_oof)
    out = IMPL / "results" / "nested_cv"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "nested_results.json", "w", encoding="utf-8") as f:
        json.dump({"selection_per_fold": selection_log,
                   "final_metrics": metrics}, f, indent=2)

    print("\n================ FINALE (unverzerrte) SCHAETZUNG ================")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Details: {out / 'nested_results.json'}")


if __name__ == "__main__":
    main()
