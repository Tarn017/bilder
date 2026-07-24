"""Lauf A: Leave-One-Tool-Out-CV OHNE synthetische Daten (Referenz).

Bank pro Fold = reale Gut-Crops der uebrigen Werkzeuge. Einfach starten:
    python run_loto_ohne_synth.py
Device (CPU/GPU) wird automatisch erkannt. Ergebnisse: results/loto_A_ohne_synth/
"""

import time

import torch

from cv_core import run_loto_cv, IMPL
from patchcore_model import PatchCoreConfig

CFG = PatchCoreConfig(
    backbone="resnet18",
    input_size=256,
    clahe=True,
    aug_n=7,                # volle D4-Augmentierung (Rotationen/Spiegelungen)
    coreset_ratio=0.05,
    use_extras=False,
    use_synth=False,        # <-- Referenz ohne synthetische Bilder
    score_topk=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

if __name__ == "__main__":
    print(f"Lauf A (ohne Synth) | Device: {CFG.device}")
    t0 = time.time()
    out = IMPL / "results" / "loto_A_ohne_synth"
    metrics, _ = run_loto_cv(CFG, out_dir=out, save_heatmaps=True)
    print(f"\nLaufzeit: {time.time() - t0:.0f}s")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Ergebnisse: {out}")
