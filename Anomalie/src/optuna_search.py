"""Optuna-Suche ueber Preprocessing/Augmentierung/Modell-Hyperparameter.

WICHTIG (Methodik): Die hier erzielten CV-Werte sind SELEKTIONS-Werte und
duerfen nicht als finale Guete berichtet werden (Selection Bias). Die
unverzerrte Schaetzung liefert nested_eval.py.

Der Suchraum ist bewusst klein gehalten: Bei 19 Basisbildern kann eine grosse
Suche das Validierungssignal auswendig lernen. Jede Dimension ist fachlich
begruendet (Graustufen: Farbe ist evtl. uninformativ; CLAHE: Kontrast der
Verschleissspuren; Augmentierung: Rotationssymmetrie des Zahnrads; extras:
Nutzen schwach verifizierter Gut-Bilder).

Beispiel:
  python optuna_search.py --trials 25 --device cuda
"""

import argparse

import optuna

from cv_core import run_loto_cv, IMPL
from patchcore_model import PatchCoreConfig


def suggest_cfg(trial: optuna.Trial, device: str) -> PatchCoreConfig:
    return PatchCoreConfig(
        backbone=trial.suggest_categorical("backbone",
                                           ["resnet18", "wide_resnet50_2"]),
        input_size=trial.suggest_categorical("input_size", [256, 384, 518]),
        grayscale=trial.suggest_categorical("grayscale", [False, True]),
        clahe=trial.suggest_categorical("clahe", [False, True]),
        aug_n=trial.suggest_categorical("aug_n", [0, 3, 7]),
        coreset_ratio=trial.suggest_categorical("coreset_ratio", [0.05, 0.10, 0.25]),
        use_extras=trial.suggest_categorical("use_extras", [False, True]),
        score_topk=trial.suggest_categorical("score_topk", [1, 10, 50, 200]),
        device=device,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--device", default="cpu")
    p.add_argument("--study-name", default="patchcore_search")
    a = p.parse_args()

    storage = f"sqlite:///{(IMPL / 'results' / 'optuna.db').as_posix()}"
    (IMPL / "results").mkdir(exist_ok=True)
    study = optuna.create_study(direction="maximize", study_name=a.study_name,
                                storage=storage, load_if_exists=True)

    def objective(trial):
        cfg = suggest_cfg(trial, a.device)
        metrics, _ = run_loto_cv(cfg, verbose=False)
        trial.set_user_attr("pixel_auroc", metrics.get("pixel_auroc"))
        return metrics["image_auroc"]

    study.optimize(objective, n_trials=a.trials)
    print("\nBeste Konfiguration (Selektionswert, nicht finale Guete!):")
    print(f"  image_auroc={study.best_value:.4f}")
    print(f"  params={study.best_params}")
    print(f"Study-DB: {storage}")


if __name__ == "__main__":
    main()
