"""Holdout-Protokoll: ausfuehrliche Suche auf VAL-Werkzeugen, EINE finale
Auswertung auf TEST-Werkzeugen.

Split (Werkzeuge strikt getrennt, Defekt- und Gutbilder in beiden Gruppen):
  VAL  = tool02, tool08, tool97   -> 6 Defekt- / 6 Gut-Basisbilder
  TEST = tool03, tool09, tool10, tool99 -> 5 Defekt- / 2 Gut-Basisbilder

Trainiert wird pro Phase auf den Gut-Bildern aller NICHT evaluierten
Werkzeuge (+ Extras). Bewusst NICHT "auf allen Gutbildern": laegen die
Val-/Test-Gutbilder mit in der Memory Bank, bekaemen sie automatisch
Score ~ 0 und die False-Positive-Rate waere geschoent.

Ablauf:
  python holdout_eval.py --search --trials 200 --device cuda   # Suche auf VAL
  python holdout_eval.py --test --device cuda                  # finale TEST-Auswertung
                                                               # (nimmt beste Suche-Konfig)
  python holdout_eval.py --test --trial-id 42                  # bestimmten Trial testen

WICHTIG: --test genau EINMAL am Ende ausfuehren. Wer mehrfach auf TEST schaut
und danach weitersucht, macht TEST zu einem zweiten VAL.
"""

import argparse
import json

import optuna

from cv_core import run_split, IMPL
from patchcore_model import PatchCoreConfig

VAL_TOOLS = ["tool02", "tool08", "tool97"]
TEST_TOOLS = ["tool03", "tool09", "tool10", "tool99"]
STUDY = "holdout_search"


def storage() -> str:
    (IMPL / "results").mkdir(exist_ok=True)
    return f"sqlite:///{(IMPL / 'results' / 'optuna.db').as_posix()}"


def suggest_cfg(trial: optuna.Trial, device: str) -> PatchCoreConfig:
    return PatchCoreConfig(
        backbone=trial.suggest_categorical("backbone",
                                           ["resnet18", "wide_resnet50_2"]),
        input_size=trial.suggest_categorical("input_size", [256, 384, 518]),
        grayscale=trial.suggest_categorical("grayscale", [False, True]),
        clahe=trial.suggest_categorical("clahe", [False, True]),
        aug_n=trial.suggest_categorical("aug_n", [0, 3, 7]),
        aug_rot_n=trial.suggest_categorical("aug_rot_n", [0, 2, 4]),
        aug_photo_n=trial.suggest_categorical("aug_photo_n", [0, 2, 4]),
        coreset_ratio=trial.suggest_categorical("coreset_ratio",
                                                [0.05, 0.10, 0.25]),
        use_extras=trial.suggest_categorical("use_extras", [False, True]),
        score_topk=trial.suggest_categorical("score_topk", [1, 10, 50, 200]),
        device=device,
    )


def cfg_from_params(params: dict, device: str) -> PatchCoreConfig:
    return PatchCoreConfig(**params, device=device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--search", action="store_true", help="Suche auf VAL")
    p.add_argument("--test", action="store_true", help="finale TEST-Auswertung")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--trial-id", type=int, default=None,
                   help="--test: diesen Trial statt des besten verwenden")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    if a.search:
        study = optuna.create_study(direction="maximize", study_name=STUDY,
                                    storage=storage(), load_if_exists=True)

        def objective(trial):
            cfg = suggest_cfg(trial, a.device)
            m, _ = run_split(cfg, VAL_TOOLS, verbose=False)
            trial.set_user_attr("pixel_auroc", m.get("pixel_auroc"))
            trial.set_user_attr("best_f1", m.get("image_best_f1"))
            return m["image_auroc"]

        study.optimize(objective, n_trials=a.trials)
        print(f"\nBeste VAL-Konfiguration: auroc={study.best_value:.4f}")
        print(f"  {study.best_params}")
        print("Finale Bewertung: python holdout_eval.py --test")

    if a.test:
        study = optuna.load_study(study_name=STUDY, storage=storage())
        if a.trial_id is not None:
            trial = study.trials[a.trial_id]
        else:
            trial = study.best_trial
        cfg = cfg_from_params(trial.params, a.device)
        print(f"Konfiguration (VAL-auroc={trial.value:.4f}): {trial.params}")

        out = IMPL / "results" / "holdout_test"
        val_m, _ = run_split(cfg, VAL_TOOLS, verbose=False)
        test_m, _ = run_split(cfg, TEST_TOOLS, out_dir=out, save_heatmaps=True)
        result = {"config": cfg.to_dict(),
                  "val_metrics": val_m, "test_metrics": test_m}
        with open(out / "holdout_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print("\n===== VAL (Selektionswert, optimistisch) =====")
        for k, v in val_m.items():
            print(f"  {k}: {v}")
        print("===== TEST (finale Zahl, nur einmal ansehen!) =====")
        for k, v in test_m.items():
            print(f"  {k}: {v}")
        print(f"\nErgebnisse: {out}")

    if not (a.search or a.test):
        print("--search und/oder --test angeben (siehe Docstring).")


if __name__ == "__main__":
    main()
