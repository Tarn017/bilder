"""CLI: eine Konfiguration per Leave-One-Tool-Out-CV evaluieren.

Beispiele:
  python run_cv.py --name baseline
  python run_cv.py --name schnell --backbone resnet18 --input-size 256 --aug-n 0
  python run_cv.py --name gray --grayscale --heatmaps

Ergebnisse landen unter Implementation/results/<name>/
(metrics.json, oof_scores.csv, optional heatmaps/).
"""

import argparse
import time
from pathlib import Path

from cv_core import run_loto_cv, IMPL
from patchcore_model import PatchCoreConfig


def build_cfg(a) -> PatchCoreConfig:
    return PatchCoreConfig(
        backbone=a.backbone, input_size=a.input_size, grayscale=a.grayscale,
        clahe=a.clahe, aug_n=a.aug_n, coreset_ratio=a.coreset_ratio,
        use_extras=not a.no_extras, score_topk=a.score_topk,
        seed=a.seed, device=a.device,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="Run-Name (Ergebnisordner)")
    p.add_argument("--backbone", default="wide_resnet50_2",
                   choices=["resnet18", "wide_resnet50_2"])
    p.add_argument("--input-size", type=int, default=384)
    p.add_argument("--grayscale", action="store_true")
    p.add_argument("--clahe", action="store_true")
    p.add_argument("--aug-n", type=int, default=3, choices=range(0, 8))
    p.add_argument("--coreset-ratio", type=float, default=0.10)
    p.add_argument("--score-topk", type=int, default=1,
                   help="Bild-Score = Mittel der k hoechsten Map-Werte")
    p.add_argument("--no-extras", action="store_true",
                   help="schwache Extra-Gutbilder NICHT verwenden")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", help="cpu oder cuda")
    p.add_argument("--heatmaps", action="store_true")
    a = p.parse_args()

    cfg = build_cfg(a)
    out = IMPL / "results" / a.name
    print(f"Konfiguration: {cfg.to_dict()}")
    t0 = time.time()
    metrics, _ = run_loto_cv(cfg, out_dir=out, save_heatmaps=a.heatmaps)
    print(f"\nLaufzeit: {time.time() - t0:.0f}s")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nErgebnisse: {out}")


if __name__ == "__main__":
    main()
