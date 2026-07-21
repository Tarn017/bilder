# AP3 – Defekterkennung als Anomalieerkennung (Kegelrad-Closeups)

Implementierung der Defekterkennung für die Nahaufnahmen (Datensatz C) als
**Anomalieerkennung**: Das Modell lernt ausschließlich aus defektfreien
Bildern, wie „normal" aussieht; die wenigen Defektbilder samt Pixelmasken
werden **nur zur Evaluation** verwendet.

## Datengrundlage

| Quelle | Inhalt | Verwendung |
|---|---|---|
| `BGAD_CNN_Dataset` | 91 Crops (518×518) der 19 offiziellen Basisbilder + Binärmasken; 55 Crops mit Defekt (aus 11 Basisbildern), 36 ohne (aus 8 Basisbildern) | Trainings-Normale (Gut-Crops) + kompletter Testbestand |
| `KI_Prod.v3i.yolov11` (eigene Roboflow-Annotation) | 25 zusätzliche Closeups ohne annotierten Defekt (nach Dedup gegen die 19 offiziellen Bilder) | optionale, **schwach verifizierte** Zusatz-Normale (`use_extras`) |
| `defekte_daten` | Vollauflösung + 5 Defektklassen | Referenz / spätere Defekttyp-Klassifikation (Stufe 2) |

`src/prepare_data.py` baut daraus `data/` (Crops, Extra-Normale,
`manifest.csv` mit Werkzeug-ID als Split-Gruppe) und meldet
Label-Konflikte. Gefundener Konflikt: `tool10_..._closeup_0004` ist offiziell
defekt (polishing_wear), wurde im eigenen Roboflow-Set aber als defektfrei
gelabelt → in der Doku als Datenqualitätsbefund erwähnen.

## Modell

`src/patchcore_model.py`: kompakte PatchCore-Implementierung (Roth et al.,
CVPR 2022). Begründung der Wahl:

- **kein Training nötig** (ImageNet-Features + Memory Bank) → geeignet für
  sehr wenige Normal-Bilder, läuft auf CPU;
- **positionsunabhängig** → robust gegenüber den unterschiedlichen
  Ansichten/Zoomstufen der Crops (im Gegensatz zu PaDiM, das ausgerichtete
  Bilder voraussetzt);
- liefert **Pixel-Anomaliekarte + Bild-Score aus einem Modell** („defekt
  ja/nein" und „wo" gleichzeitig).

Konfigurierbar (CLI/Optuna): Backbone (resnet18 / wide_resnet50_2),
Eingabegröße, Graustufen, CLAHE, geometrische Augmentierung der Normal-Bilder
(D4-Rotationen/Spiegelungen – begründbar über die Rotationssymmetrie des
Zahnrads), Coreset-Rate, Nutzung der Extra-Normale.

## Evaluationsprinzip

- **Split-Gruppe = Werkzeug-ID** (Leave-One-Tool-Out): dasselbe physische
  Zahnrad ist mehrfach fotografiert und jedes Bild in ~5 überlappende Crops
  zerlegt – jeder zufällige Split wäre Data Leakage.
- **Bewertungseinheit = Basisbild**: Bild-Score = Maximum über dessen Crops
  (Crops überlappen; Crop-Metriken würden Defekte mehrfach zählen).
- Metriken: Bild-AUROC (primär), Pixel-AUROC gegen die Masken, F1/
  Konfusionsmatrix (Schwellwert auf OOF-Scores, als optimistisch markiert).

## Skripte

| Skript | Zweck |
|---|---|
| `src/prepare_data.py` | Datenbestand + Manifest bauen (einmalig) |
| `src/run_cv.py --name X [Optionen]` | eine Konfiguration per LOTO-CV evaluieren, optional `--heatmaps` |
| `src/optuna_search.py --trials N` | Hyperparameter-/Preprocessing-Suche (**nur Selektion!**) |
| `src/nested_eval.py [--fast]` | **Nested CV** → unverzerrte finale Gütezahl trotz Modellauswahl |

Ergebnisse: `results/<name>/metrics.json`, `oof_scores.csv`, `heatmaps/`
(Anomalie-Heatmap mit Ground-Truth-Kontur in weiß).

## Methodik: Modellauswahl ohne Holdout?

Direkte Antwort auf die Projektfrage: Mit LOTO-CV selektieren **und** dieselben
Zahlen berichten wäre optimistisch verzerrt (Selection Bias). Ein fester
Holdout ist bei 19 Basisbildern aber statistisch wertlos. Lösung = **Nested
CV** (`nested_eval.py`): außen wird je ein Werkzeug komplett zurückgelegt,
innen wird auf den übrigen Werkzeugen selektiert, das gewählte Modell wird
einmalig auf das äußere Werkzeug angewendet. Die gesammelten äußeren Scores
schätzen die Güte der *Gesamtprozedur inklusive Auswahl* unverzerrt.
Die Optuna-Suche dient nur der Exploration/Kandidatenfindung; die finale
Zahl kommt aus `nested_eval.py`.

## Ausführung auf dem GPU-PC

Der Ordner ist self-contained: `data/` (Crops, Extra-Normale, `manifest.csv`)
ist bereits gebaut — einfach den kompletten `Implementation`-Ordner kopieren,
die Quelldatensätze werden auf dem Zielrechner **nicht** benötigt
(`prepare_data.py` muss dort nicht laufen).

```bash
# 1) Umgebung (einmalig) — torch mit CUDA passend zur Karte:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 2) GPU-Check + Sanity-Lauf (~1 Minute, sollte image_auroc ≈ 0.61 ergeben):
python -c "import torch; print(torch.cuda.is_available())"
cd src
python run_cv.py --name gpu_check --backbone resnet18 --input-size 256 --aug-n 0 --device cuda

# 3) Alles in einem: Sanity -> Optuna -> Nested CV -> finaler Heatmap-Lauf
python run_all.py                # Device wird automatisch erkannt
```

Alternativ die Schritte einzeln (`optuna_search.py`, `nested_eval.py
--from-optuna 2`, `run_cv.py --heatmaps ...`), siehe unten.

## Empfohlener Ablauf

```bash
cd Implementation/src
python prepare_data.py                        # einmalig
python run_cv.py --name baseline --heatmaps   # Referenzlauf ansehen
python optuna_search.py --trials 25           # Exploration (GPU/Colab: --device cuda)
# ggf. CANDIDATES in nested_eval.py um Optuna-Favoriten ergänzen
python nested_eval.py                         # finale, berichtbare Zahlen
```

CPU-Hinweis: `--backbone resnet18 --input-size 256 --aug-n 0` läuft lokal in
wenigen Minuten; die Vollkonfiguration (wide_resnet50_2, 384–518 px,
Augmentierung) ist für GPU/Colab gedacht.

## Limitationen (für die Doku)

- n = 19 Basisbilder → alle Metriken haben breite Unsicherheitsbänder;
  deshalb CV statt Einzelsplit und Angabe der Streuung.
- Extra-Normale sind nur schwach verifiziert (eigene Annotation); ihr Nutzen
  ist bewusst ein Ablationsparameter (`use_extras`).
- Die 5 Defektklassen werden hier binär zusammengefasst; Defekttyp-
  Klassifikation (Few-Shot auf den `defekte_daten`-Klassenmasken) ist als
  zweite Stufe vorgesehen.
