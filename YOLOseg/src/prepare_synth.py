"""Synthetische Defektbilder freistellen.

Die 150 Bilder in defekt_synth/images haben noch den Aufnahmehintergrund
(Werkbank, Maschinenbett). Die echten Aufnahmen in defekte_daten und die
synthetischen Gut-Bilder in intakt_synth/cropped sind dagegen freigestellt.
Dieses Skript entfernt den Hintergrund anhand von object_masks_png und legt
das Ergebnis in defekt_synth/cropped ab - gleiches Format wie
intakt_synth/cropped (1088x725, PNG, reinweisser Hintergrund).

Zusaetzlich entsteht results/synth_qualitaet.csv mit zwei Kennzahlen je Bild,
die fuer die spaetere Auswahl gebraucht werden:

  abdeckung        Flaechenanteil des Bauteils im Bild. Niedrige Werte
                   bedeuten meist eine Gesamtansicht des Rads statt einer
                   Zahn-Nahaufnahme - also einen anderen Bildausschnitt als
                   die Auswertungsdaten.
  defekt_ausserhalb Anteil der Defektmaske, der ausserhalb der Bauteilmaske
                   liegt. Hohe Werte weisen auf Maskenartefakte hin (Striche
                   am Bildrand, Flecken neben dem Bauteil).

Wichtig: Was beim Freistellen weissgetont wird, ist im Bild nicht mehr
sichtbar und darf spaeter auch nicht als Defekt gelabelt werden. Die fuer
YOLO nutzbare Defektmaske ist daher stets defektmaske UND bauteilmaske;
die Spalte defekt_rest_px weist aus, was davon uebrig bleibt.

Aufruf:
  python prepare_synth.py
  python prepare_synth.py --force     # vorhandene Dateien ueberschreiben
"""

import argparse
import csv
import pathlib

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "defekt_synth"
DST = SRC / "cropped"
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
BG = 255                      # reinweiss, wie in intakt_synth/cropped


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    imgs = sorted(SRC.glob("images/*.jpg")) + sorted(SRC.glob("images/*.png"))
    if not imgs:
        raise SystemExit(f"Keine Bilder in {SRC / 'images'}")
    DST.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    rows, uebersprungen = [], 0
    for src in imgs:
        ziel = DST / (src.stem + ".png")
        if ziel.exists() and not a.force:
            uebersprungen += 1
            continue

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        om = cv2.imread(str(SRC / "object_masks_png" / f"{src.stem}.png"),
                        cv2.IMREAD_GRAYSCALE)
        dm = cv2.imread(str(SRC / "masks_png" / f"{src.stem}.png"),
                        cv2.IMREAD_GRAYSCALE)
        if img is None or om is None or dm is None:
            raise SystemExit(f"Datei fehlt oder unlesbar zu {src.stem}")
        if om.shape != img.shape[:2] or dm.shape != img.shape[:2]:
            raise SystemExit(f"Maskengroesse passt nicht zu {src.stem}")

        obj, def_ = om > 127, dm > 127
        frei = img.copy()
        frei[~obj] = BG
        cv2.imwrite(str(ziel), frei)

        d_ges = int(def_.sum())
        d_rest = int((def_ & obj).sum())
        rows.append({
            "bild": src.stem,
            "defektart": src.stem.rsplit("_", 1)[0].replace(
                "synth_bevel_gear_spindle_closeup_", ""),
            "abdeckung": round(float(obj.mean()), 4),
            "defekt_px": d_ges,
            "defekt_anteil": round(d_ges / def_.size, 5),
            "defekt_rest_px": d_rest,
            "defekt_ausserhalb": round(1 - d_rest / d_ges, 4) if d_ges else "",
        })

    if rows:
        with open(RESULTS / "synth_qualitaet.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"Freigestellt: {len(rows)}   uebersprungen (vorhanden): {uebersprungen}")
    print(f"Ziel        : {DST}")
    if not rows:
        return

    cov = np.array([r["abdeckung"] for r in rows])
    out = np.array([r["defekt_ausserhalb"] for r in rows if r["defekt_ausserhalb"] != ""])
    print(f"\nAbdeckung        Median {np.median(cov):.1%}  "
          f"unter 60 %: {(cov < 0.60).sum()} von {len(cov)}")
    print(f"Defekt ausserhalb Median {np.median(out):.2%}  "
          f"ueber 5 %: {(out > 0.05).sum()} von {len(out)}")
    print(f"Bilder ohne verbleibende Defektflaeche: "
          f"{sum(1 for r in rows if r['defekt_rest_px'] == 0)}")
    print(f"\nKennzahlen  : {RESULTS / 'synth_qualitaet.csv'}")


if __name__ == "__main__":
    main()
