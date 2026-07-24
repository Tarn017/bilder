"""Schneidet aus jedem freigestellten Synth-Bild (defec_synth/cropped/) einen
518x518-Ausschnitt und speichert ihn in defec_synth/518x518/.

Auswahlkriterium fuer die Ausschnittposition (ueber die Alpha-Maske):
  - Bauteilanteil >= 50 % der Ausschnittsflaeche
  - Hintergrund sichtbar: Bauteilanteil <= 90 %
Unter allen gueltigen Positionen (Rasterschritt 8 px) wird die gewaehlt, deren
Bauteilanteil am naechsten an 70 % liegt (deterministisch). Ist keine Position
gueltig, wird die naechstliegende genommen und gemeldet.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "defec_synth"
OUT = SRC / "518x518"
SIZE = 518          # Fenstergroesse im Quellbild (via --window aenderbar:
                    # kleineres Fenster = staerkerer Zoom, Ausgabe bleibt 518)
OUT_SIZE = 518
FG_MIN, FG_MAX, FG_TARGET = 0.50, 0.90, 0.70
STRIDE = 8


def best_window(fg: np.ndarray) -> tuple[int, int, float]:
    """Beste (x, y)-Position + Bauteilanteil via Integralbild."""
    h, w = fg.shape
    ii = cv2.integral(fg.astype(np.uint8))          # (h+1, w+1)
    area = SIZE * SIZE
    best, best_valid = None, None
    for y in range(0, h - SIZE + 1, STRIDE):
        for x in range(0, w - SIZE + 1, STRIDE):
            s = ii[y + SIZE, x + SIZE] - ii[y, x + SIZE] - ii[y + SIZE, x] + ii[y, x]
            frac = s / area
            cand = (abs(frac - FG_TARGET), x, y, frac)
            if best is None or cand < best:
                best = cand
            if FG_MIN <= frac <= FG_MAX and (best_valid is None or cand < best_valid):
                best_valid = cand
    chosen = best_valid or best
    return chosen[1], chosen[2], chosen[3]


def main() -> None:
    global SIZE, OUT
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=518,
                   help="Fenstergroesse im Quellbild; < 518 zoomt staerker ran")
    p.add_argument("--out", default="518x518", help="Zielordner in defec_synth")
    a = p.parse_args()
    SIZE = a.window
    OUT = SRC / a.out

    OUT.mkdir(exist_ok=True)
    fallback = []
    for img_p in sorted((SRC / "cropped").glob("*.png")):
        img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        alpha = cv2.imread(str(SRC / "vision_masks" / img_p.name),
                           cv2.IMREAD_GRAYSCALE)
        if alpha.shape != img.shape[:2]:
            alpha = cv2.resize(alpha, (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        fg = alpha > 127
        x, y, frac = best_window(fg)
        if not (FG_MIN <= frac <= FG_MAX):
            fallback.append((img_p.name, round(frac, 2)))
        crop = img[y:y + SIZE, x:x + SIZE]
        if SIZE != OUT_SIZE:
            crop = cv2.resize(crop, (OUT_SIZE, OUT_SIZE),
                              interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(OUT / img_p.name), crop)
        print(f"{img_p.name}: x={x} y={y} bauteilanteil={frac:.0%}")

    print(f"\ngespeichert: {len(list(OUT.glob('*.png')))} -> {OUT} "
          f"(Fenster {SIZE}px -> Ausgabe {OUT_SIZE}px)")
    if fallback:
        print(f"ausserhalb 50-90% (naechstliegende Position genommen): {fallback}")


if __name__ == "__main__":
    main()
