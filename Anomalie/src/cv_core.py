"""Leave-One-Tool-Out-Cross-Validation fuer die Anomalieerkennung.

Split-Prinzip (methodische Begruendung):
  Dasselbe physische Werkzeug ist mehrfach fotografiert und jedes Basisbild in
  ~5 ueberlappende Crops zerlegt. Zufaellige Splits waeren daher Data Leakage.
  Gruppierungseinheit ist die Werkzeug-ID: In Fold "toolXX" bilden ALLE
  offiziellen Crops dieses Werkzeugs das Testset; die Memory Bank wird
  ausschliesslich aus Gut-Bildern der UEBRIGEN Werkzeuge gebaut (offizielle
  Gut-Crops + optional schwache Extra-Gutbilder).

  So erhaelt jedes Basisbild einen Out-of-Fold-Score von einem Modell, das
  sein Werkzeug nie gesehen hat -> unverzerrte Score-Sammlung ueber alle Bilder.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve

from patchcore_model import PatchCore, PatchCoreConfig

IMPL = Path(__file__).resolve().parents[1]
DATA = IMPL / "data"
PIXEL_EVAL_SIZE = 256    # Aufloesung fuer Pixel-Metriken (Speicher/Laufzeit)


def load_manifest():
    with open(DATA / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["has_defect"] = int(r["has_defect"])
        r["weak_label"] = int(r["weak_label"])
    return rows


def _eval_fold(cfg, train, test, fold_name, oof, pixel_scores, pixel_labels,
               out_dir=None, save_heatmaps=False):
    """Ein Fit auf `train`-Normalen, Scoring aller `test`-Crops (in-place)."""
    model = PatchCore(cfg)
    model.fit([DATA / r["path"] for r in train])
    for r in test:
        score, amap = model.score(DATA / r["path"])
        oof.append({**r, "score": score, "fold": fold_name})

        mask = cv2.imread(str(DATA / r["mask_path"]), cv2.IMREAD_GRAYSCALE)
        m = (cv2.resize(mask, (PIXEL_EVAL_SIZE, PIXEL_EVAL_SIZE),
                        interpolation=cv2.INTER_NEAREST) > 127)
        a = cv2.resize(amap, (PIXEL_EVAL_SIZE, PIXEL_EVAL_SIZE),
                       interpolation=cv2.INTER_LINEAR)
        pixel_scores.append(a.ravel())
        pixel_labels.append(m.ravel())

        if save_heatmaps and out_dir is not None and r["has_defect"]:
            _save_heatmap(DATA / r["path"], amap, mask,
                          out_dir / "heatmaps" / (Path(r["path"]).stem + ".png"),
                          cfg.input_size)


def _split_rows(cfg, eval_tools):
    """(train, test) fuer eine Werkzeug-Gruppe: Training = Gut-Bilder aller
    NICHT evaluierten Werkzeuge (verhindert, dass Test-Gutbilder bereits in
    der Memory Bank liegen und die FPR geschoent wird)."""
    rows = load_manifest()
    official = [r for r in rows if r["source"] == "official_crop"]
    extras = [r for r in rows if r["source"] == "extra_roboflow"]
    test = [r for r in official if r["tool"] in eval_tools]
    train = [r for r in official
             if r["tool"] not in eval_tools and not r["has_defect"]]
    if cfg.use_extras:
        train += [r for r in extras if r["tool"] not in eval_tools]
    if getattr(cfg, "use_synth", False):
        # synthetische Normale: reine Trainingsdaten, in jedem Fold dabei
        train += [r for r in rows if r["source"] == "synth_normal"]
    return train, test


def run_split(cfg: PatchCoreConfig, eval_tools: list[str],
              out_dir: Path | None = None, save_heatmaps: bool = False,
              verbose: bool = True):
    """Ein einzelner Holdout-Split: eine Werkzeug-Gruppe wird evaluiert."""
    train, test = _split_rows(cfg, eval_tools)
    if verbose:
        print(f"  Split {sorted(eval_tools)}: {len(train)} Normal-Bilder "
              f"-> {len(test)} Test-Crops")
    oof, pixel_scores, pixel_labels = [], [], []
    _eval_fold(cfg, train, test, "+".join(sorted(eval_tools)),
               oof, pixel_scores, pixel_labels, out_dir, save_heatmaps)
    metrics = compute_metrics(oof, pixel_scores, pixel_labels)
    if out_dir is not None:
        _write_results(out_dir, cfg, metrics, oof)
    return metrics, oof


def run_loto_cv(cfg: PatchCoreConfig, tools: list[str] | None = None,
                out_dir: Path | None = None, save_heatmaps: bool = False,
                verbose: bool = True):
    """Fuehrt die LOTO-CV aus und liefert (metrics, oof_rows)."""
    rows = load_manifest()
    official = [r for r in rows if r["source"] == "official_crop"]
    all_tools = sorted({r["tool"] for r in official})
    tools = tools or all_tools

    oof = []
    pixel_scores, pixel_labels = [], []
    for tool in tools:
        train, test = _split_rows(cfg, [tool])
        if verbose:
            print(f"  Fold {tool}: {len(train)} Normal-Bilder -> {len(test)} Test-Crops")
        _eval_fold(cfg, train, test, tool, oof, pixel_scores, pixel_labels,
                   out_dir, save_heatmaps)

    metrics = compute_metrics(oof, pixel_scores, pixel_labels)
    if out_dir is not None:
        _write_results(out_dir, cfg, metrics, oof)
    return metrics, oof


def _write_results(out_dir: Path, cfg, metrics, oof):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "oof_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(oof[0].keys()))
        w.writeheader()
        w.writerows(oof)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg.to_dict(), "metrics": metrics}, f, indent=2)


def compute_metrics(oof, pixel_scores=None, pixel_labels=None):
    """Bild-Level (Basisbild = max ueber Crops) + Pixel-Level Metriken."""
    by_base = defaultdict(lambda: {"score": -np.inf, "label": 0})
    for r in oof:
        b = by_base[r["base_image"]]
        b["score"] = max(b["score"], r["score"])
        b["label"] = max(b["label"], r["has_defect"])
    y = np.array([v["label"] for v in by_base.values()])
    s = np.array([v["score"] for v in by_base.values()])

    metrics = {"n_images": int(len(y)), "n_defect_images": int(y.sum())}
    if 0 < y.sum() < len(y):
        metrics["image_auroc"] = float(roc_auc_score(y, s))
        prec, rec, thr = precision_recall_curve(y, s)
        f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
        i = int(np.nanargmax(f1[:-1]))
        t = float(thr[i])
        tp = int(((s >= t) & (y == 1)).sum()); fp = int(((s >= t) & (y == 0)).sum())
        fn = int(((s < t) & (y == 1)).sum()); tn = int(((s < t) & (y == 0)).sum())
        metrics.update({
            "image_best_f1": float(f1[i]),
            "best_f1_threshold_note": "optimistisch (auf OOF-Scores gewaehlt)",
            "confusion_at_best_f1": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        })
    if pixel_scores:
        ps = np.concatenate(pixel_scores)
        pl = np.concatenate(pixel_labels)
        if 0 < pl.sum() < len(pl):
            # Subsampling der Negativ-Pixel haelt die AUROC-Berechnung schlank
            neg = np.flatnonzero(~pl)
            rng = np.random.default_rng(0)
            keep = rng.choice(neg, size=min(len(neg), 500_000), replace=False)
            idx = np.concatenate([np.flatnonzero(pl), keep])
            metrics["pixel_auroc"] = float(roc_auc_score(pl[idx], ps[idx]))
    return metrics


def _save_heatmap(img_path, amap, mask, out_path, input_size):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(img_path))
    a = (amap - amap.min()) / max(amap.max() - amap.min(), 1e-9)
    a = cv2.resize(a, (img.shape[1], img.shape[0]))
    heat = cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
    contours, _ = cv2.findContours((mask > 127).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)  # GT weiss
    cv2.imwrite(str(out_path), overlay)
