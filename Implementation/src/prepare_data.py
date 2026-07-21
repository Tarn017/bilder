"""Datenaufbereitung fuer die Anomalieerkennung (AP3).

Baut aus den Quelldaten einen konsistenten, versionierten Datenbestand unter
Implementation/data auf:

  - crops/          die 91 offiziellen BGAD-Crops (518x518) + Binaermasken
  - extra_normals/  zusaetzliche (schwach verifizierte) Gut-Closeups aus dem
                    eigenen Roboflow-Datensatz (leere Label-Datei = kein Defekt
                    annotiert). Bilder, die bereits zu den 19 offiziellen
                    Basisbildern gehoeren, werden ausgeschlossen (Dedup ueber
                    kanonischen Namen, damit z.B. "..._0008_isolated_v2" und
                    "..._0008_isolated" als dasselbe Bild erkannt werden).
  - manifest.csv    ein Eintrag pro Bild mit Werkzeug-ID (Split-Gruppe!),
                    Basisbild, Defektstatus und Herkunft.

Konsistenzpruefungen: Roboflow-"gut" vs. offizielle Defektbilder werden als
Konflikt gemeldet (waere ein Hinweis auf uebersehene Defekte).
"""

import csv
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]          # "KI Prod CC"
IMPL = Path(__file__).resolve().parents[1]           # Implementation/
BGAD = ROOT / "BGAD_CNN_Dataset"
ROBOFLOW = ROOT / "KI_Prod.v3i.yolov11" / "train"
DATA = IMPL / "data"

CROP_RE = re.compile(r"^(?P<base>.*)_c(?P<view>\d{3})_direct_b01_c00$")


def canonical(name: str) -> str:
    """Kanonischer Bildname: ohne _isolated/_v2-Suffixe und Roboflow-Hash."""
    name = re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", name)
    name = name.replace("_isolated", "")
    name = re.sub(r"_v\d+$", "", name)
    return name


def tool_of(name: str) -> str:
    return name.split("_")[0]


def main() -> None:
    rows = []
    (DATA / "crops" / "images").mkdir(parents=True, exist_ok=True)
    (DATA / "crops" / "masks").mkdir(parents=True, exist_ok=True)
    (DATA / "extra_normals").mkdir(parents=True, exist_ok=True)

    # --- 1) Offizielle BGAD-Crops -------------------------------------------
    official_bases = {}   # canonical base -> has_defect (auf Basisbild-Ebene)
    crop_rows = []
    for img in sorted((BGAD / "images").glob("*.jpg")):
        m = CROP_RE.match(img.stem)
        if not m:
            print(f"WARNUNG: unerwarteter Crop-Name {img.name}", file=sys.stderr)
            continue
        base = m.group("base")
        mask_src = BGAD / "masks" / (img.stem + ".png")
        mask = np.array(Image.open(mask_src).convert("L"))
        has_defect = bool((mask > 127).any())

        shutil.copy2(img, DATA / "crops" / "images" / img.name)
        shutil.copy2(mask_src, DATA / "crops" / "masks" / mask_src.name)

        cbase = canonical(base)
        official_bases[cbase] = official_bases.get(cbase, False) or has_defect
        crop_rows.append({
            "path": f"crops/images/{img.name}",
            "mask_path": f"crops/masks/{mask_src.name}",
            "base_image": cbase,
            "tool": tool_of(base),
            "view": m.group("view"),
            "source": "official_crop",
            "weak_label": 0,
            "has_defect": int(has_defect),
        })
    rows.extend(crop_rows)

    # --- 2) Zusaetzliche schwache Gut-Bilder aus dem Roboflow-Datensatz -----
    n_extra, n_dedup, conflicts = 0, 0, []
    for lbl in sorted((ROBOFLOW / "labels").glob("*.txt")):
        if lbl.stat().st_size > 0:
            continue                      # Defekt annotiert -> kein Gut-Bild
        img = ROBOFLOW / "images" / (lbl.stem + ".jpg")
        if not img.exists():
            continue
        cbase = canonical(lbl.stem)
        if cbase in official_bases:
            if official_bases[cbase]:     # offiziell defekt, von uns "gut"?!
                conflicts.append(cbase)
            n_dedup += 1
            continue
        dst_name = cbase + ".jpg"
        shutil.copy2(img, DATA / "extra_normals" / dst_name)
        rows.append({
            "path": f"extra_normals/{dst_name}",
            "mask_path": "",
            "base_image": cbase,
            "tool": tool_of(cbase),
            "view": "full",
            "source": "extra_roboflow",
            "weak_label": 1,              # nur schwach verifiziert defektfrei
            "has_defect": 0,
        })
        n_extra += 1

    # --- 3) Manifest schreiben ----------------------------------------------
    with open(DATA / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- 4) Zusammenfassung --------------------------------------------------
    n_def_crops = sum(r["has_defect"] for r in crop_rows)
    def_bases = sorted(b for b, d in official_bases.items() if d)
    good_bases = sorted(b for b, d in official_bases.items() if not d)
    print(f"Offizielle Crops:    {len(crop_rows)} "
          f"({n_def_crops} mit Defekt, {len(crop_rows) - n_def_crops} ohne)")
    print(f"  Basisbilder:       {len(official_bases)} "
          f"({len(def_bases)} defekt / {len(good_bases)} gut)")
    print(f"Extra-Gutbilder:     {n_extra} uebernommen, "
          f"{n_dedup} als Duplikat der offiziellen Bilder verworfen")
    tools = sorted({r['tool'] for r in rows})
    for t in tools:
        sub = [r for r in rows if r["tool"] == t]
        print(f"  {t}: {len(sub)} Bilder "
              f"(defekt-Crops: {sum(r['has_defect'] for r in sub)}, "
              f"extras: {sum(r['weak_label'] for r in sub)})")
    if conflicts:
        print("\nKONFLIKTE (offiziell defekt, aber im eigenen Roboflow-Set als "
              "defektfrei gelabelt -> pruefen!):")
        for c in conflicts:
            print(f"  {c}")
    print(f"\nManifest: {DATA / 'manifest.csv'}")


if __name__ == "__main__":
    main()
