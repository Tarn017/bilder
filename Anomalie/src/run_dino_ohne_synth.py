"""Lauf C: LOTO-CV mit DINOv2-Backbone (AnomalyDINO-Ansatz), OHNE Synth.

Gleiche Folds und gleiches Schema wie die Laeufe A/B, nur das Backbone ist
DINOv2 ViT-S/14 bei nativer 518px-Eingabe. Beim ersten Start werden einmalig
~90 MB Gewichte von torch.hub geladen (Internet noetig).

    python run_dino_ohne_synth.py

Hinweis: auf CPU mehrere Stunden -> fuer den GPU-PC gedacht.
Ergebnisse: results/loto_C_dino_ohne_synth/
"""

import time

import torch

from cv_core import run_loto_cv, IMPL
from patchcore_model import PatchCoreConfig

CFG = PatchCoreConfig(
    backbone="dinov2_vits14",
    input_size=518,          # nativ fuer DINOv2 (14er-Patchraster, 37x37)
    clahe=True,
    aug_n=7,
    coreset_ratio=0.05,
    use_extras=False,
    use_synth=False,         # <-- ohne synthetische Bilder
    score_topk=50,           # robuste Bild-Score-Aggregation bei hoher Aufloesung
    device="cuda" if torch.cuda.is_available() else "cpu",
)

if __name__ == "__main__":
    print(f"Lauf C (DINOv2, ohne Synth) | Device: {CFG.device}")
    t0 = time.time()
    out = IMPL / "results" / "loto_C_dino_ohne_synth"
    metrics, _ = run_loto_cv(CFG, out_dir=out, save_heatmaps=True)
    print(f"\nLaufzeit: {time.time() - t0:.0f}s")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Ergebnisse: {out}")
