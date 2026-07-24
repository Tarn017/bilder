"""Ein-Kommando-Einstieg: kompletter Trainings-/Evaluationsprozess.

Fuehrt nacheinander aus:
  1. Sanity-Lauf   (schnelle Konfiguration; sollte image_auroc ~ 0.61 ergeben)
  2. Optuna-Suche  (Exploration von Preprocessing/Augmentierung, nur Selektion)
  3. Nested CV     (finale, berichtbare Gueteschaetzung; nimmt die 2 besten
                    Optuna-Konfigurationen mit in die Kandidatenliste)
  4. Finaler Lauf  (beste Optuna-Konfiguration mit Heatmap-Export fuer Folien)

Hinweis: Bei PatchCore gibt es kein Epochen-Training - das "Training" ist der
Aufbau der Memory Bank und steckt in jedem dieser Laeufe automatisch drin.

Aufruf (Device wird automatisch erkannt):
  python run_all.py                 # Standard: 25 Optuna-Trials
  python run_all.py --trials 40
  python run_all.py --skip-sanity
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent


def run(args: list[str]) -> None:
    print(f"\n>>> {' '.join(args)}\n{'=' * 60}")
    r = subprocess.run([sys.executable] + args, cwd=SRC)
    if r.returncode != 0:
        sys.exit(f"Abbruch: '{' '.join(args)}' schlug fehl (Code {r.returncode})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=25, help="Optuna-Trials")
    p.add_argument("--device", default=None, help="cpu/cuda (Standard: Auto)")
    p.add_argument("--skip-sanity", action="store_true")
    a = p.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNUNG: keine GPU gefunden - das wird mehrere Stunden dauern. "
              "Fuer den GPU-PC gedacht.")

    if not a.skip_sanity:
        run(["run_cv.py", "--name", "sanity_check", "--backbone", "resnet18",
             "--input-size", "256", "--aug-n", "0", "--device", device])

    run(["optuna_search.py", "--trials", str(a.trials), "--device", device])
    run(["nested_eval.py", "--device", device, "--from-optuna", "2"])

    # Finaler Lauf mit der besten Optuna-Konfiguration inkl. Heatmaps
    from nested_eval import optuna_top_candidates
    best = optuna_top_candidates(1)[0]
    final = ["run_cv.py", "--name", "final", "--device", device, "--heatmaps",
             "--backbone", str(best.get("backbone", "wide_resnet50_2")),
             "--input-size", str(best.get("input_size", 384)),
             "--aug-n", str(best.get("aug_n", 3)),
             "--coreset-ratio", str(best.get("coreset_ratio", 0.10)),
             "--score-topk", str(best.get("score_topk", 50))]
    if best.get("grayscale"):
        final.append("--grayscale")
    if best.get("clahe"):
        final.append("--clahe")
    if not best.get("use_extras", True):
        final.append("--no-extras")
    run(final)

    print("\nFERTIG.")
    print("  Berichtbare Zahlen : results/nested_cv/nested_results.json")
    print("  Heatmaps fuer Folien: results/final/heatmaps/")
    print("  Optuna-Verlauf      : results/optuna.db")


if __name__ == "__main__":
    main()
