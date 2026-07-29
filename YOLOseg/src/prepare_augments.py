"""Offline-Augmentierung der 19 echten Aufnahmen (Variante "aug").

Je Bild entstehen zwei Fassungen; zusammen mit dem Original verdreifacht das
die Menge der echten Trainingsbilder. Anders als bei der Anomalieerkennung
werden hier ALLE Bilder augmentiert, nicht nur die defektfreien: Ein
Segmentierungsnetz lernt aus Defektbildern, und mit elf davon ist jede
zusaetzliche Fassung relevant.

Die Defektmasken machen dieselbe Geometrie mit und werden mitgeschrieben -
sonst zeigte das Label nach der Rotation an die falsche Stelle. Helligkeit
und Kontrast wirken naturgemaess nur auf das Bild.

Rezepte - jede Variante kombiniert Geometrie und Photometrie:
  aug1  Horizontalspiegelung + Rotation +12 Grad + Zoom, Helligkeit/Kontrast +5 %
  aug2  Vertikalspiegelung  + Rotation -12 Grad + Zoom, Helligkeit/Kontrast -5 %

Begruendung der Transformationen:
  * Rotation ist physikalisch motiviert - das Bauteil liegt auf einem
    Drehteller, die Winkellage ist beliebig.
  * Spiegelungen sind verlustfrei und kosten keinen Zoom. Sie bilden kein
    real existierendes Bauteil ab (ein gespiegeltes Spiralkegelrad waere
    linksdrehend), veraendern die Oberflaechenmerkmale eines Schadens aber
    nicht.
  * Der Zoom ist kein freier Parameter, sondern folgt aus dem Rotations-
    winkel (siehe min_scale), damit keine Fuellflaeche sichtbar bleibt.
  * Helligkeit und Kontrast bleiben bewusst schwach (+-5 %): zwei der fuenf
    Defektklassen (polishing_wear, fretting_corrosion) sind ueber Glanz bzw.
    Helligkeit definiert.

Hinweis: Ultralytics augmentiert mit Standardparametern zusaetzlich online
(Mosaic, Flip, HSV, Scale, Translate). Diese Offline-Augmentierung kommt
also OBENDRAUF; die Variante heisst deshalb korrekt "zusaetzliche
Offline-Augmentierung" und nicht "mit Augmentierung".

Aufruf:
  python prepare_augments.py
  python prepare_augments.py --force
"""

import argparse
import math

import albumentations as A
import cv2
import numpy as np

from loto import AUG_IMG, AUG_MASK, collect

JPEG_QUALITY = 95
RECIPES = {
    "aug1": dict(flip="h", rotate=12.0, zoom=1.00,
                 brightness=0.05, contrast=0.05),
    "aug2": dict(flip="v", rotate=-12.0, zoom=1.05,
                 brightness=-0.05, contrast=-0.05),
}


def min_scale(h: int, w: int, angle_deg: float) -> float:
    """Kleinster Zoom, bei dem nach der Rotation keine Fuellflaeche sichtbar ist."""
    a = abs(math.radians(angle_deg))
    c, s = math.cos(a), math.sin(a)
    return max((w * c + h * s) / w, (h * c + w * s) / h)


def background_color(img: np.ndarray, border: int = 80) -> tuple:
    """Medianfarbe der vier Bildecken - Fuellwert fuer die Rotation."""
    corners = np.concatenate([
        img[:border, :border].reshape(-1, 3), img[:border, -border:].reshape(-1, 3),
        img[-border:, :border].reshape(-1, 3), img[-border:, -border:].reshape(-1, 3)])
    return tuple(int(v) for v in np.median(corners, axis=0))


def build(recipe: dict, fill: tuple, shape: tuple) -> A.Compose:
    scale = min_scale(shape[0], shape[1], recipe["rotate"]) * recipe["zoom"]
    steps = []
    if recipe["flip"] == "h":
        steps.append(A.HorizontalFlip(p=1.0))
    elif recipe["flip"] == "v":
        steps.append(A.VerticalFlip(p=1.0))
    steps += [
        A.Affine(rotate=(recipe["rotate"], recipe["rotate"]),
                 scale=(scale, scale), fit_output=False,
                 border_mode=cv2.BORDER_CONSTANT, fill=fill, fill_mask=0, p=1.0),
        A.RandomBrightnessContrast(
            brightness_limit=(recipe["brightness"], recipe["brightness"]),
            contrast_limit=(recipe["contrast"], recipe["contrast"]), p=1.0),
    ]
    return A.Compose(steps)


def ensure_augments(force: bool = False, verbose: bool = True) -> int:
    AUG_IMG.mkdir(parents=True, exist_ok=True)
    AUG_MASK.mkdir(parents=True, exist_ok=True)
    images = collect()
    neu = vorhanden = 0

    for stem, entry in images.items():
        img = masks = None
        for tag, recipe in RECIPES.items():
            dst = AUG_IMG / f"{stem}_{tag}.jpg"
            if dst.exists() and not force:
                vorhanden += 1
                continue
            if img is None:
                img = cv2.imread(str(entry["path"]), cv2.IMREAD_COLOR)
                masks = [cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
                         for m in entry["masks"]]
            tf = build(recipe, background_color(img), img.shape)
            res = tf(image=img, masks=masks) if masks else tf(image=img)
            cv2.imwrite(str(dst), res["image"],
                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            for src_mask, out_mask in zip(entry["masks"], res.get("masks", [])):
                art = src_mask.stem[len(stem) + 1:]
                cv2.imwrite(str(AUG_MASK / f"{stem}_{tag}_{art}.png"),
                            (out_mask > 127).astype(np.uint8) * 255)
            neu += 1
            if verbose:
                rest = (sum((m > 127).sum() for m in res.get("masks", []))
                        / max(sum((m > 127).sum() for m in masks), 1)
                        if masks else 1.0)
                print(f"  {dst.name:46s} {len(entry['masks'])} Maske(n), "
                      f"{rest:.0%} der Defektflaeche erhalten")

    if verbose:
        defekt = sum(1 for v in images.values() if v["label"])
        print(f"\nEchte Bilder: {len(images)} ({defekt} defekt / "
              f"{len(images) - defekt} gut)")
        print(f"Augmentierungen: neu {neu}, vorhanden {vorhanden} "
              f"-> Trainingsmenge {len(images)} auf "
              f"{len(images) * (1 + len(RECIPES))} (Faktor {1 + len(RECIPES)})")
    return neu


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    print(f"Augmentierung -> {AUG_IMG}\n")
    ensure_augments(force=a.force)
