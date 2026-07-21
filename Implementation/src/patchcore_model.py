"""PatchCore-Anomalieerkennung (Roth et al., CVPR 2022), kompakte Eigenimplementierung.

Warum PatchCore:
  - benoetigt kein Training, nur ImageNet-Features + Memory Bank -> geeignet
    fuer sehr wenige Normal-Bilder und CPU-Betrieb
  - positionsUNabhaengig (globale Patch-Bank) -> robust gegenueber den
    unterschiedlichen Ansichten/Zooms der BGAD-Crops (PaDiM o.ae. setzen
    ausgerichtete Bilder voraus und sind hier methodisch ungeeignet)

Konfigurierbare Bausteine (fuer Optuna / Ablationen):
  backbone, Eingabegroesse, Graustufen, CLAHE, geometrische Augmentierung der
  Normal-Bilder (Rotationen/Spiegelungen -> begruendbar ueber die
  Rotationssymmetrie des Zahnrads), Coreset-Subsampling-Rate.
"""

from dataclasses import dataclass, field, asdict

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet18, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PAD_VALUE = (200, 200, 200)   # heller Hintergrund der freigestellten Bilder


@dataclass
class PatchCoreConfig:
    backbone: str = "wide_resnet50_2"   # oder "resnet18"
    input_size: int = 384               # durch 8 teilbar
    grayscale: bool = False
    clahe: bool = False
    aug_n: int = 3                      # 0..7 geometrische Zusatzvarianten pro Normal-Bild
    coreset_ratio: float = 0.10         # Anteil der Patches in der Memory Bank
    max_bank: int = 30000               # harte Obergrenze (Speicher/Laufzeit)
    use_extras: bool = True             # schwache Gut-Bilder mitverwenden
    score_topk: int = 1                 # Bild-Score = Mittel der k hoechsten
                                        # Map-Werte (1 = Maximum; groessere k
                                        # robuster gegen Einzelpixel-Ausreisser)
    seed: int = 42
    device: str = "cpu"

    def to_dict(self):
        return asdict(self)


# Die 7 nicht-trivialen Elemente der Dieder-Gruppe D4 (Rotationen/Spiegelungen).
_D4 = [
    lambda a: np.rot90(a, 1),
    lambda a: np.rot90(a, 2),
    lambda a: np.rot90(a, 3),
    lambda a: a[:, ::-1],
    lambda a: np.rot90(a[:, ::-1], 1),
    lambda a: np.rot90(a[:, ::-1], 2),
    lambda a: np.rot90(a[:, ::-1], 3),
]


class PatchCore:
    def __init__(self, cfg: PatchCoreConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)
        if cfg.backbone == "resnet18":
            net = resnet18(weights="IMAGENET1K_V1")
        elif cfg.backbone == "wide_resnet50_2":
            net = wide_resnet50_2(weights="IMAGENET1K_V1")
        else:
            raise ValueError(f"unbekanntes Backbone: {cfg.backbone}")
        net.eval()
        self.extractor = create_feature_extractor(
            net, return_nodes={"layer2": "l2", "layer3": "l3"}
        ).to(self.device)
        for p in self.extractor.parameters():
            p.requires_grad_(False)

        pre = [A.LongestMaxSize(cfg.input_size),
               A.PadIfNeeded(cfg.input_size, cfg.input_size,
                             border_mode=cv2.BORDER_CONSTANT, fill=PAD_VALUE)]
        if cfg.clahe:
            pre.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0))
        if cfg.grayscale:
            pre.append(A.ToGray(num_output_channels=3, p=1.0))
        self.preprocess = A.Compose(pre)

        self.bank: torch.Tensor | None = None
        self.grid: tuple[int, int] | None = None

    # ------------------------------------------------------------------ utils
    def _load(self, path) -> np.ndarray:
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        return self.preprocess(image=img)["image"]

    def _embed(self, imgs: list[np.ndarray]) -> torch.Tensor:
        """Bilder -> Patch-Features (B, H2*W2, C)."""
        x = np.stack([(i.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
                      for i in imgs])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            out = self.extractor(x)
        l2, l3 = out["l2"], out["l3"]
        # lokale Mittelung (3x3) glaettet die Patch-Deskriptoren (wie im Paper)
        l2 = F.avg_pool2d(l2, 3, 1, 1)
        l3 = F.avg_pool2d(l3, 3, 1, 1)
        l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear",
                           align_corners=False)
        feat = torch.cat([l2, l3], dim=1)                  # (B, C, H2, W2)
        b, c, h, w = feat.shape
        self.grid = (h, w)
        return feat.permute(0, 2, 3, 1).reshape(b, h * w, c)

    @staticmethod
    def _greedy_coreset(feats: torch.Tensor, n_select: int,
                        proj_dim: int = 128, seed: int = 42) -> torch.Tensor:
        """Greedy-k-Center-Auswahl (auf zufaellig projizierten Features)."""
        g = torch.Generator().manual_seed(seed)
        pmat = torch.randn(feats.shape[1], proj_dim, generator=g).to(feats.device)
        proj = feats @ pmat
        n = proj.shape[0]
        # quadrierte Distanzen per Matvec (torch.cdist gegen einen Einzelpunkt
        # ist auf CPU um Groessenordnungen langsamer)
        sq = (proj * proj).sum(dim=1)
        idx = torch.zeros(n_select, dtype=torch.long, device=feats.device)
        idx[0] = torch.randint(n, (1,), generator=g).to(feats.device)
        dmin = sq + sq[idx[0]] - 2.0 * (proj @ proj[idx[0]])
        for i in range(1, n_select):
            idx[i] = torch.argmax(dmin)
            d = sq + sq[idx[i]] - 2.0 * (proj @ proj[idx[i]])
            dmin = torch.minimum(dmin, d)
        return feats[idx]

    # ------------------------------------------------------------------- API
    def fit(self, image_paths: list) -> None:
        """Memory Bank aus defektfreien Bildern aufbauen."""
        cfg = self.cfg
        variants = []
        for p in image_paths:
            img = self._load(p)
            variants.append(img)
            variants += [np.ascontiguousarray(f(img)) for f in _D4[:cfg.aug_n]]
        # Subsampling bereits pro Batch: begrenzt den Spitzen-Speicherbedarf
        # (bei 518px fallen sonst >1 GB Roh-Features an, bevor das Coreset greift)
        n_batches = max(1, -(-len(variants) // 8))
        quota = max(256, (cfg.max_bank * 4) // n_batches)
        g = torch.Generator().manual_seed(cfg.seed)
        feats = []
        for i in range(0, len(variants), 8):
            e = self._embed(variants[i:i + 8])
            e = e.reshape(-1, e.shape[-1])
            if e.shape[0] > quota:
                e = e[torch.randperm(e.shape[0], generator=g)[:quota]]
            feats.append(e)
        feats = torch.cat(feats)
        n_select = min(int(feats.shape[0] * cfg.coreset_ratio), cfg.max_bank)
        n_select = max(n_select, min(1000, feats.shape[0]))
        self.bank = self._greedy_coreset(feats, n_select, seed=cfg.seed)

    def score(self, image_path) -> tuple[float, np.ndarray]:
        """Liefert (Bild-Score, Anomaliekarte in Eingabegroesse)."""
        assert self.bank is not None, "fit() zuerst aufrufen"
        emb = self._embed([self._load(image_path)])[0]          # (P, C)
        dmin = torch.full((emb.shape[0],), float("inf"), device=emb.device)
        for i in range(0, self.bank.shape[0], 8192):            # chunked kNN
            d = torch.cdist(emb, self.bank[i:i + 8192])
            dmin = torch.minimum(dmin, d.min(dim=1).values)
        h, w = self.grid
        amap = dmin.reshape(h, w).cpu().numpy()
        amap = cv2.resize(amap, (self.cfg.input_size, self.cfg.input_size),
                          interpolation=cv2.INTER_LINEAR)
        amap = cv2.GaussianBlur(amap, (0, 0), sigmaX=4.0)
        k = max(1, int(self.cfg.score_topk))
        flat = np.sort(amap.ravel())
        return float(flat[-k:].mean()), amap
