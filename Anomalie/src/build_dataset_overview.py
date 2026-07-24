"""Baut den Ueberblicks-Ordner `dataset/` aus den freigestellten Closeups.

Bildquelle ist ausschliesslich `all closeups/` (71 freigestellte Closeups,
1088x725). BGAD_CNN_Dataset dient nur als Defekt-LABELQUELLE:

  defect/images/  Closeups, bei denen mindestens ein BGAD-Crop derselben
                  Closeup-Nummer Defektpixel traegt
  defect/labels/  zugehoerige Klassen-Masken aus defekte_daten (5472x3648),
                  auf die Closeup-Groesse (1088x725) skaliert
  no_defect/      Closeups, deren BGAD-Crops alle defektfrei sind, plus
                  nie offiziell bearbeitete Closeups OHNE eigene
                  Roboflow-Defektpolygone
  unknown/        nie offiziell bearbeitete Closeups MIT eigenen
                  Roboflow-Defektpolygonen (vermutlich defekt, keine Masken)

Vorher wird der alte Inhalt der vier Zielordner geleert (Neuaufbau).
"""

import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CLOSEUPS = ROOT / "all closeups"
BGAD = ROOT / "BGAD_CNN_Dataset"
DEFEKTE = ROOT / "defekte_daten"
ROBOFLOW = ROOT / "KI_Prod.v3i.yolov11" / "train"
DST = ROOT / "dataset"


def canonical(name: str) -> str:
    name = re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", name)
    name = name.replace("_isolated", "")
    return re.sub(r"_v\d+$", "", name)


def main() -> None:
    # --- Zielordner leeren und anlegen --------------------------------------
    dirs = {
        "def_img": DST / "defect" / "images",
        "def_lbl": DST / "defect" / "labels",
        "ok": DST / "no_defect",
        "unk": DST / "unknown",
    }
    for d in [DST / "defect", dirs["ok"], dirs["unk"]]:
        if d.exists():
            shutil.rmtree(d)
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # --- Defektstatus pro Basisbild aus den BGAD-Crop-Masken ----------------
    base_defect: dict[str, bool] = {}
    for mask_p in (BGAD / "masks").glob("*.png"):
        base = canonical(re.sub(r"_c\d{3}_direct_b01_c00$", "", mask_p.stem))
        has = bool((np.array(Image.open(mask_p).convert("L")) > 127).any())
        base_defect[base] = base_defect.get(base, False) or has

    # --- Roboflow-Defektstatus (fuer die nie bearbeiteten Closeups) ---------
    rf_defect = {canonical(l.stem) for l in (ROBOFLOW / "labels").glob("*.txt")
                 if l.stat().st_size > 0}

    # --- Closeups einsortieren ----------------------------------------------
    counts = {"defect": 0, "no_defect": 0, "unknown": 0}
    for img in sorted(CLOSEUPS.glob("*.jpg")):
        cbase = canonical(img.stem)
        if cbase in base_defect:
            target = dirs["def_img"] if base_defect[cbase] else dirs["ok"]
            counts["defect" if base_defect[cbase] else "no_defect"] += 1
        elif cbase in rf_defect:
            target = dirs["unk"]
            counts["unknown"] += 1
        else:
            target = dirs["ok"]
            counts["no_defect"] += 1
        shutil.copy2(img, target / img.name)

    # --- Klassen-Masken aus defekte_daten skaliert uebernehmen --------------
    seen, n_lbl = set(), 0
    ref = cv2.imread(str(next(CLOSEUPS.glob("*.jpg"))))
    h, w = ref.shape[:2]
    for mask_p in sorted(DEFEKTE.rglob("masks/*.png")):
        if mask_p.name in seen:          # train/ und val/ sind identische Kopien
            continue
        seen.add(mask_p.name)
        m = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(dirs["def_lbl"] / mask_p.name), small)
        n_lbl += 1

    print(f"defect/images: {counts['defect']}")
    print(f"defect/labels: {n_lbl} Masken (auf {w}x{h} skaliert)")
    print(f"no_defect:     {counts['no_defect']}")
    print(f"unknown:       {counts['unknown']}")
    print(f"Summe Bilder:  {sum(counts.values())} (erwartet: 71)")

    # Konsistenz: jedes defect-Bild braucht mindestens eine Maske
    def_bases = {canonical(p.stem) for p in dirs["def_img"].glob("*.jpg")}
    lbl_bases = {b for m in seen for b in def_bases if canonical(m[:-4]).startswith(b)}
    missing = def_bases - lbl_bases
    if missing:
        print(f"WARNUNG: defect-Bilder ohne Maske: {sorted(missing)}")


if __name__ == "__main__":
    main()
