"""Stellt die synthetischen Defektbilder frei (weisser Hintergrund) und
skaliert sie auf 1088 px lange Seite -> defec_synth/cropped/.

Besonderheiten der Quelldaten:
  - raw/ ist ~1536x1024, vision_masks/ ist bereits 1088x725
    -> das Raw-Bild wird auf die Maskengroesse skaliert
  - die Masken sind weiche Alpha-Masken (Werte 0..255)
    -> Alpha-Compositing auf Weiss statt hartem Schwellwert (saubere Kanten,
       konsistent mit den *_isolated-Bildern der Realdaten)
  - Raw-Bilder ohne Maske (0046-0050) werden uebersprungen und gemeldet
"""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "defec_synth"
OUT = SRC / "cropped"
TARGET_LONG = 1088


def main() -> None:
    OUT.mkdir(exist_ok=True)
    done, skipped = 0, []
    for raw_p in sorted((SRC / "raw").glob("*.png")):
        mask_p = SRC / "vision_masks" / raw_p.name
        if not mask_p.exists():
            skipped.append(raw_p.name)
            continue
        raw = cv2.imread(str(raw_p), cv2.IMREAD_COLOR)
        alpha = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        h, w = alpha.shape

        img = cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        comp = img * alpha[..., None] + 255.0 * (1.0 - alpha[..., None])

        if max(h, w) != TARGET_LONG:                 # Sicherheitsnetz
            s = TARGET_LONG / max(h, w)
            comp = cv2.resize(comp, (round(w * s), round(h * s)),
                              interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / raw_p.name), comp.astype(np.uint8))
        done += 1

    print(f"freigestellt und skaliert: {done} -> {OUT}")
    if skipped:
        print(f"uebersprungen (keine Maske): {skipped}")


if __name__ == "__main__":
    main()
