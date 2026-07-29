"""Gemeinsame Grundlagen fuer die YOLO-Segmentierungs-Baseline.

Datenquellen
------------
  defekte_daten/all_19/base_images   19 echte Aufnahmen, 5472x3648
  defekte_daten/all_19/masks         13 Defektmasken zu 11 dieser Bilder
  data/augments/{images,masks}       2 Offline-Augmentierungen je echtem Bild
  defekt_synth/cropped               150 synthetische Defektbilder (freigestellt)
  defekt_synth/masks_png             deren Defektmasken
  defekt_synth/object_masks_png      deren Bauteilsilhouetten
  intakt_synth/cropped               42 synthetische Gut-Bilder

Aufteilung
----------
Leave-One-Tool-Out ueber die sieben physischen Werkzeuge. Bewertet wird immer
ausschliesslich auf den ECHTEN Bildern des ausgehaltenen Werkzeugs.
Augmentierungen und synthetische Bilder stehen ausnahmslos im Training - eine
Augmentierung des Testwerkzeugs waere ein direktes Leck, ein synthetisches
Bild gehoert zu keinem physischen Werkzeug.

Labels
------
Eine einzige Klasse (0 = defect); die Defektart wird nicht unterschieden.
Gut-Bilder erhalten eine leere Label-Datei und wirken damit als
Negativbeispiele. Bei den synthetischen Defektbildern wird die Defektmaske
mit der Bauteilmaske verschnitten: Was beim Freistellen weiss geworden ist,
ist im Bild nicht mehr sichtbar und darf nicht als Defekt gelabelt werden.

Bildablage
----------
Alle Bilder liegen einmalig in data/pool/{images,labels}, auf POOL_SIZE als
laengste Seite verkleinert. Je Fold und Variante entsteht daraus nur eine
Textdatei mit Bildpfaden - so wird kein Bild mehrfach kopiert. Die
Polygonkoordinaten sind normiert und daher von der Ablagegroesse unabhaengig.
"""

import pathlib

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parents[1]
DATA = HERE / "data"
RESULTS = HERE / "results"

REAL_IMG = ROOT / "defekte_daten" / "all_19" / "base_images"
REAL_MASK = ROOT / "defekte_daten" / "all_19" / "masks"
AUG_IMG = DATA / "augments" / "images"
AUG_MASK = DATA / "augments" / "masks"
SYN_DEF = ROOT / "defekt_synth" / "cropped"
SYN_DEF_MASK = ROOT / "defekt_synth" / "masks_png"
SYN_DEF_OBJ = ROOT / "defekt_synth" / "object_masks_png"
SYN_OK = ROOT / "intakt_synth" / "cropped"
POOL = DATA / "pool"

POOL_SIZE = 1024          # laengste Seite in der Ablage; deckt beide imgsz ab
RESOLUTIONS = [640, 1024]
VARIANTS = ["basis", "aug", "synth", "aug_synth"]
CONF = 0.5                # ab dieser Konfidenz gilt eine Instanz als Fund
EPOCHS = 60
# Polygone werden erst NACH dem Verkleinern auf POOL_SIZE bestimmt. Nur so
# bedeutet eine Mindestflaeche bei allen Quellen dasselbe: die echten Bilder
# sind 5472 px breit, die synthetischen 1088 px - eine absolute Schwelle auf
# der Quellaufloesung wuerde bei den synthetischen Bildern feine Schadstellen
# (Pitting) verwerfen und bei den echten nichts.
MIN_AREA = 9              # kleinste Konturflaeche in Pixeln der Poolgroesse
APPROX_EPS = 0.0015       # Konturvereinfachung, Anteil der Bilddiagonale


# ------------------------------------------------------------------- Daten
def collect() -> dict:
    """Die 19 echten Bilder mit Werkzeug, Label und zugehoerigen Masken."""
    masks = sorted(REAL_MASK.glob("*.png"))
    out = {}
    for p in sorted(REAL_IMG.glob("*.jpg")):
        mine = [m for m in masks if m.stem.startswith(p.stem + "_")]
        out[p.stem] = {"path": p, "tool": p.stem.split("_")[0],
                       "label": int(bool(mine)), "masks": mine}
    if not out:
        raise SystemExit(f"Keine Bilder in {REAL_IMG}")
    return out


def augment_files() -> dict:
    """Augmentierte Fassungen, nach Ursprungsbild gruppiert."""
    out = {}
    for p in sorted(AUG_IMG.glob("*.jpg")):
        stem = p.stem.rsplit("_aug", 1)[0]
        mine = [m for m in sorted(AUG_MASK.glob("*.png"))
                if m.stem.startswith(p.stem + "_")]
        out.setdefault(stem, []).append({"path": p, "masks": mine})
    return out


def tools(images: dict) -> list:
    return sorted({v["tool"] for v in images.values()})


# ---------------------------------------------------------------- Polygone
def mask_to_polygons(mask: np.ndarray) -> list:
    """Binaermaske -> Liste normierter Polygone im YOLO-Segmentierungsformat.

    Es werden nur aeussere Konturen verwendet; Loecher innerhalb einer
    Schadstelle spielen fuer die Ja/Nein-Entscheidung keine Rolle.
    """
    h, w = mask.shape[:2]
    eps = APPROX_EPS * float(np.hypot(h, w))
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float64)
        if len(c) < 3:
            continue
        c[:, 0] = np.clip(c[:, 0] / w, 0.0, 1.0)
        c[:, 1] = np.clip(c[:, 1] / h, 0.0, 1.0)
        polys.append(c.reshape(-1))
    return polys


def label_text(polys: list) -> str:
    return "\n".join("0 " + " ".join(f"{v:.6f}" for v in p) for p in polys)


def merged_mask(paths: list, shape: tuple = None) -> np.ndarray:
    """Mehrere Maskendateien eines Bildes zu einer Binaermaske vereinen."""
    out = None
    for m in paths:
        a = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
        if a is None:
            raise SystemExit(f"Maske unlesbar: {m}")
        a = a > 127
        out = a if out is None else (out | a)
    if out is None:
        out = np.zeros(shape, dtype=bool)
    return out


# ------------------------------------------------------------------ Metrik
def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson-Konfidenzintervall fuer einen Anteil."""
    if n == 0:
        return ("", "")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - r), 3), round(min(1.0, c + r), 3))


def counts(rows: list) -> dict:
    tp = sum(1 for r in rows if r["label"] and r["pred"])
    fp = sum(1 for r in rows if not r["label"] and r["pred"])
    fn = sum(1 for r in rows if r["label"] and not r["pred"])
    tn = sum(1 for r in rows if not r["label"] and not r["pred"])
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics(rows: list) -> dict:
    """Kennzahlen einer Bildmenge. Nicht definierte Werte bleiben leer."""
    c = counts(rows)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    rec = tp / (tp + fn) if tp + fn else None
    spez = tn / (tn + fp) if tn + fp else None
    prec = tp / (tp + fp) if tp + fp else None
    f1 = (2 * prec * rec / (prec + rec)
          if prec is not None and rec is not None and prec + rec else None)
    out = dict(c, n=len(rows),
               accuracy=round((tp + tn) / len(rows), 4) if rows else None)
    for k, v in (("recall", rec), ("spezifitaet", spez),
                 ("precision", prec), ("f1", f1)):
        out[k] = round(v, 4) if v is not None else ""
    return out


def auroc(rows: list) -> float:
    """Rangbasierte AUROC ueber den Bildscore; leer, wenn eine Klasse fehlt."""
    pos = [r["score"] for r in rows if r["label"]]
    neg = [r["score"] for r in rows if not r["label"]]
    if not pos or not neg:
        return ""
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)
