"""Bildpool und YOLO-Labels erzeugen.

Alle vier Datenquellen landen einmalig in data/pool/{images,labels}, auf
POOL_SIZE als laengste Seite verkleinert. Die Foldaufteilung erfolgt spaeter
ausschliesslich ueber Textdateien mit Bildpfaden - dadurch existiert jedes
Bild genau einmal auf der Platte statt 56-mal.

Namensraeume im Pool (die Quellen haben teils gleiche Dateinamen):
  <stem>              19 echte Aufnahmen
  <stem>_aug1/_aug2   38 Offline-Augmentierungen
  sydef_<stem>        150 synthetische Defektbilder
  syok_<name>         42 synthetische Gut-Bilder

Labels: eine Klasse (0 = defect). Gut-Bilder bekommen eine leere Datei und
wirken damit als Negativbeispiel. Bei den synthetischen Defektbildern wird
die Defektmaske mit der Bauteilmaske verschnitten - was beim Freistellen
weiss geworden ist, ist nicht mehr sichtbar und darf nicht gelabelt werden.

Aufruf:
  python prepare_pool.py
  python prepare_pool.py --force
"""

import argparse

import cv2
import numpy as np

from loto import (POOL, POOL_SIZE, SYN_DEF, SYN_DEF_MASK, SYN_DEF_OBJ, SYN_OK,
                  augment_files, collect, label_text, mask_to_polygons,
                  merged_mask)

JPEG_QUALITY = 95


def _write(name: str, img: np.ndarray, mask: np.ndarray, force: bool) -> list:
    """Bild verkleinert ablegen, Polygone auf Poolgroesse bestimmen, Label schreiben.

    Die Polygone entstehen bewusst erst nach dem Verkleinern, damit die
    Mindestflaeche MIN_AREA fuer echte und synthetische Bilder dasselbe
    bedeutet (siehe Kommentar in loto.py).
    """
    dst = POOL / "images" / f"{name}.jpg"
    lbl = POOL / "labels" / f"{name}.txt"
    s = POOL_SIZE / max(img.shape[:2])
    if s < 1.0:
        size = (round(img.shape[1] * s), round(img.shape[0] * s))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), size,
                          interpolation=cv2.INTER_NEAREST) > 0
    polys = mask_to_polygons(mask)
    if not (dst.exists() and lbl.exists() and not force):
        cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        lbl.write_text(label_text(polys), encoding="utf-8")
    return polys


def build(force: bool = False, verbose: bool = True) -> dict:
    (POOL / "images").mkdir(parents=True, exist_ok=True)
    (POOL / "labels").mkdir(parents=True, exist_ok=True)
    stat = {"echt": 0, "augment": 0, "synth_defekt": 0, "synth_gut": 0}
    leere_labels = []

    def ablegen(name: str, img: np.ndarray, mask: np.ndarray,
                ist_defekt: bool, art: str) -> None:
        if img is None:
            raise SystemExit(f"Bild unlesbar: {name}")
        if not _write(name, img, mask, force) and ist_defekt:
            leere_labels.append(name)
        stat[art] += 1

    # --- echte Aufnahmen ------------------------------------------------
    images = collect()
    for stem, e in images.items():
        img = cv2.imread(str(e["path"]), cv2.IMREAD_COLOR)
        ablegen(stem, img, merged_mask(e["masks"], img.shape[:2]),
                bool(e["label"]), "echt")

    # --- Augmentierungen -------------------------------------------------
    for stem, varianten in augment_files().items():
        for v in varianten:
            img = cv2.imread(str(v["path"]), cv2.IMREAD_COLOR)
            ablegen(v["path"].stem, img, merged_mask(v["masks"], img.shape[:2]),
                    bool(images[stem]["label"]), "augment")

    # --- synthetische Defektbilder ---------------------------------------
    for p in sorted(SYN_DEF.glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        dm = cv2.imread(str(SYN_DEF_MASK / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
        om = cv2.imread(str(SYN_DEF_OBJ / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if dm is None or om is None:
            raise SystemExit(f"Maske fehlt zu {p.name}")
        ablegen(f"sydef_{p.stem}", img, (dm > 127) & (om > 127), True,
                "synth_defekt")

    # --- synthetische Gut-Bilder -----------------------------------------
    for p in sorted(SYN_OK.glob("*")):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        ablegen(f"syok_{p.stem}", img,
                np.zeros(img.shape[:2], dtype=bool), False, "synth_gut")

    if verbose:
        print(f"Pool: {POOL}")
        for k, v in stat.items():
            print(f"  {k:14s} {v:4d}")
        print(f"  {'gesamt':14s} {sum(stat.values()):4d}")
        if leere_labels:
            print(f"\nWARNUNG: {len(leere_labels)} Defektbild(er) ohne "
                  f"verwertbares Polygon - sie wirken im Training wie Gut-Bilder:")
            for s in leere_labels[:10]:
                print(f"    {s}")
    return stat


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    build(force=p.parse_args().force)
