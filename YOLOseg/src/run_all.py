"""YOLOv11n-seg als Baseline: Leave-One-Tool-Out ueber alle Varianten.

Versuchsraster
--------------
  2 Aufloesungen (640, 1024) x 4 Datenvarianten x 7 Folds = 56 Trainingslaeufe
  je 60 Epochen mit den Ultralytics-Standardparametern.

  basis      nur die 19 echten Aufnahmen
  aug        + 2 Offline-Augmentierungen je echtem Bild
  synth      + 150 synthetische Defekt- und 42 synthetische Gut-Bilder
  aug_synth  + beides

Augmentierungen und synthetische Bilder stehen ausnahmslos im Training.
Bewertet wird immer nur auf den ECHTEN Bildern des ausgehaltenen Werkzeugs.

Kein Modell wird ausgewaehlt
---------------------------
Ausgewertet wird immer der Stand nach genau 60 Epochen (last.pt). Wuerde
stattdessen best.pt nach der val-mAP genommen und diente das ausgehaltene
Werkzeug als val, waere das Modellauswahl auf der Testmenge und die
Kennzahlen waeren optimistisch verzerrt.

Zwei Vorkehrungen sichern das ab: val=False schaltet die Validierung je
Epoche ab, und val zeigt in der data.yaml auf die Trainingsbilder. Letzteres
ist noetig, weil Ultralytics in der Schlussepoche auch bei val=False
validiert - das ausgehaltene Werkzeug kommt so waehrend des Trainings
nirgends vor.

Auswertung
----------
Jedes der 19 echten Bilder wird von genau dem Fold vorhergesagt, in dem sein
Werkzeug ausgehalten war. Anschliessend werden alle 19 Vorhersagen gepoolt
und EINE Konfusionsmatrix gebildet. Ein Mittelwert ueber die Folds waere hier
irrefuehrend: tool97 und tool99 enthalten keinen einzigen Defekt (Recall dort
undefiniert), tool03/08/09/10 kein einziges Gut-Bild (Spezifitaet undefiniert).
Die gepoolte Auswertung ist zugleich direkt mit der Anomalieerkennung und dem
VLM vergleichbar, die ebenfalls auf denselben 19 Bildern berichtet werden.

  Bildscore     hoechste Instanzkonfidenz im Bild (0.0, wenn keine Instanz)
  Entscheidung  defekt, sobald der Bildscore >= CONF (0.5) ist
  zusaetzlich   schwellwertfreie Bild-AUROC ueber denselben Score

Hinweis: Ultralytics augmentiert mit Standardparametern bereits online
(Mosaic, Flip, HSV, Scale, Translate). Die Variante "aug" fuegt Offline-
Augmentierung ZUSAETZLICH hinzu; der Vergleich basis/aug misst daher nicht
"ohne gegen mit Augmentierung".

Aufruf
------
  python run_all.py                       # alles
  python run_all.py --resolutions 640     # nur eine Aufloesung
  python run_all.py --variants basis aug
  python run_all.py --epochs 2 --smoke    # schneller Funktionstest
"""

import argparse
import csv
import time

import torch
from ultralytics import YOLO, settings

from loto import (CONF, DATA, EPOCHS, POOL, RESOLUTIONS, RESULTS, VARIANTS,
                  auroc, collect, metrics, wilson)
from prepare_augments import ensure_augments
from prepare_pool import build as build_pool

MODEL = "yolo11n-seg.pt"
RUNS = DATA.parent / "runs"
DETAIL_FIELDS = ["bauteil", "tool", "label", "score", "vorhersage",
                 "instanzen", "trainingsbilder"]
SUMMARY_FIELDS = ["aufloesung", "variante", "n", "tp", "fp", "fn", "tn",
                  "recall", "recall_ki_low", "recall_ki_high",
                  "spezifitaet", "spezifitaet_ki_low", "spezifitaet_ki_high",
                  "precision", "f1", "accuracy", "auroc", "f1_trivial",
                  "train_min", "train_max", "laufzeit_s"]


def train_files(variant: str, exclude_tool: str, images: dict) -> list:
    """Bildpfade im Pool fuer eine Variante ohne das ausgehaltene Werkzeug."""
    echte = [s for s, v in images.items() if v["tool"] != exclude_tool]
    files = [POOL / "images" / f"{s}.jpg" for s in echte]
    if variant in ("aug", "aug_synth"):
        files += [p for s in echte
                  for p in sorted((POOL / "images").glob(f"{s}_aug*.jpg"))]
    if variant in ("synth", "aug_synth"):
        files += sorted((POOL / "images").glob("sydef_*.jpg"))
        files += sorted((POOL / "images").glob("syok_*.jpg"))
    return files


def fold_dataset(variant: str, tool: str, images: dict) -> tuple:
    """train.txt / val.txt / data.yaml fuer einen Fold schreiben.

    val zeigt bewusst auf die TRAININGSBILDER, nicht auf das ausgehaltene
    Werkzeug: Ultralytics validiert in der letzten Epoche auch dann, wenn
    val=False gesetzt ist, und waehlt best.pt anhand dieser Kennzahl. Zeigte
    val auf das Testwerkzeug, taeuchten die Testdaten waehrend des Trainings
    auf. So kommen sie in der data.yaml ueberhaupt nicht vor; die Auswertung
    laeuft ausschliesslich ueber die eigene Vorhersage mit last.pt.
    """
    d = DATA / "folds" / f"{variant}_{tool}"
    d.mkdir(parents=True, exist_ok=True)
    tr = train_files(variant, tool, images)
    liste = "\n".join(p.as_posix() for p in tr)
    (d / "train.txt").write_text(liste, encoding="utf-8")
    (d / "val.txt").write_text(liste, encoding="utf-8")
    (d / "data.yaml").write_text(
        f"path: {d.as_posix()}\ntrain: train.txt\nval: val.txt\n"
        f"nc: 1\nnames:\n  0: defect\n", encoding="utf-8")
    test = [POOL / "images" / f"{s}.jpg"
            for s, v in images.items() if v["tool"] == tool]
    return d, len(tr), test


def run_fold(variant: str, tool: str, imgsz: int, images: dict, epochs: int,
             device: str, seed: int) -> tuple:
    d, n_train, test_paths = fold_dataset(variant, tool, images)
    name = f"{imgsz}_{variant}_{tool}"
    weights = RUNS / name / "weights" / "last.pt"

    if not weights.exists():
        YOLO(MODEL).train(
            data=str(d / "data.yaml"), epochs=epochs, imgsz=imgsz,
            device=device, seed=seed, deterministic=True,
            val=False,               # kein val-Ranking: last.pt zaehlt
            project=str(RUNS), name=name, exist_ok=True,
            verbose=False, plots=False)

    model = YOLO(str(weights))
    rows = []
    for p in test_paths:
        res = model.predict(source=str(p), imgsz=imgsz, conf=0.001,
                            device=device, verbose=False)[0]
        confs = res.boxes.conf.tolist() if res.boxes is not None else []
        score = max(confs) if confs else 0.0
        stem = p.stem
        rows.append({"bauteil": stem, "tool": images[stem]["tool"],
                     "label": images[stem]["label"], "score": round(score, 4),
                     "vorhersage": int(score >= CONF),
                     "instanzen": sum(1 for c in confs if c >= CONF),
                     "trainingsbilder": n_train})
    return rows, n_train


def summarize(rows: list, imgsz: int, variant: str, dauer: float) -> dict:
    for r in rows:
        r["pred"] = r["vorhersage"]
    m = metrics(rows)
    p = sum(r["label"] for r in rows) / len(rows)
    n_tr = [r["trainingsbilder"] for r in rows]
    return {"aufloesung": imgsz, "variante": variant, **m,
            "recall_ki_low": wilson(m["tp"], m["tp"] + m["fn"])[0],
            "recall_ki_high": wilson(m["tp"], m["tp"] + m["fn"])[1],
            "spezifitaet_ki_low": wilson(m["tn"], m["tn"] + m["fp"])[0],
            "spezifitaet_ki_high": wilson(m["tn"], m["tn"] + m["fp"])[1],
            "auroc": auroc(rows),
            "f1_trivial": round(2 * p / (p + 1), 4),
            "train_min": min(n_tr), "train_max": max(n_tr),
            "laufzeit_s": round(dauer)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolutions", nargs="*", type=int, default=RESOLUTIONS)
    ap.add_argument("--variants", nargs="*", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="nur der erste Fold je Kombination")
    a = ap.parse_args()
    device = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    settings.update({"mlflow": False})

    print("Daten bereitstellen ...")
    ensure_augments(verbose=False)
    build_pool(verbose=False)
    RESULTS.mkdir(parents=True, exist_ok=True)

    images = collect()
    tool_list = sorted({v["tool"] for v in images.values()})
    if a.smoke:
        tool_list = tool_list[:1]
    n_runs = len(a.resolutions) * len(a.variants) * len(tool_list)
    print(f"{len(images)} echte Bilder, {len(tool_list)} Folds | "
          f"{n_runs} Trainingslaeufe a {a.epochs} Epochen | Device {device}\n")

    summaries, t_start = [], time.time()
    for imgsz in a.resolutions:
        for variant in a.variants:
            t0 = time.time()
            rows = []
            for tool in tool_list:
                print(f"[{imgsz} px | {variant} | Fold {tool}] ", end="", flush=True)
                r, n_tr = run_fold(variant, tool, imgsz, images, a.epochs,
                                   device, a.seed)
                rows += r
                treffer = sum(1 for x in r if x["vorhersage"] == x["label"])
                print(f"{n_tr:3d} Trainingsbilder -> {treffer}/{len(r)} richtig")

            with open(RESULTS / f"detail_{imgsz}_{variant}.csv", "w",
                      newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=DETAIL_FIELDS,
                                   extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)

            s = summarize(rows, imgsz, variant, time.time() - t0)
            summaries.append(s)
            print(f"  => Recall {s['recall']}  Spezifitaet {s['spezifitaet']}  "
                  f"F1 {s['f1']}  AUROC {s['auroc']}  "
                  f"[{s['tp']}/{s['fp']}/{s['fn']}/{s['tn']}]  "
                  f"({s['laufzeit_s']}s)\n")

    with open(RESULTS / "uebersicht.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(summaries)
    print(f"Gesamtlaufzeit: {time.time() - t_start:.0f}s")
    print(f"Uebersicht : {RESULTS / 'uebersicht.csv'}")
    print(f"Detaildaten: {RESULTS}/detail_<aufloesung>_<variante>.csv")


if __name__ == "__main__":
    main()
