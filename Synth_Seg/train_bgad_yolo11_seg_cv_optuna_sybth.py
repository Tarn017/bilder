from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import itertools
import json
import math
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PARTITIONS = ("train", "val")
BASE_SEED = 42

DEFAULT_DATASET_DIR = Path("BGAD_CNN_Dataset")
DEFAULT_SYNTHETIC_DIR = Path("Synthetic defects")
DEFAULT_OUTPUT_DIR = Path("runs_bgad_yolo11_seg_cv_synthetic")
DEFAULT_MODEL = "yolo11n-seg.pt"
DEFAULT_N_TRIALS = 40
DEFAULT_N_FOLDS = 3
DEFAULT_IMGSZ = 640
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 20
DEFAULT_BATCH = 8
DEFAULT_AUG_COPIES = 1
DEFAULT_SYNTHETIC_AUG_COPIES = 0
DEFAULT_MIN_COMPONENT_AREA = 10
DEFAULT_SYNTHETIC_MIN_COMPONENT_AREA = 25
DEFAULT_SYNTHETIC_MAX_COMPONENTS = 15
DEFAULT_CONTOUR_EPSILON = 0.001
DEFAULT_OBJECTIVE_MAP50_WEIGHT = 0.5
DEFAULT_OBJECTIVE_STD_PENALTY = 0.25

CAPTURE_SUFFIX = re.compile(r"_c\d{3}_direct_b\d+_c\d+$")


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path
    tool_id: str
    capture_id: str
    has_defect: bool
    label_rows: tuple[str, ...]
    component_count: int
    dropped_components: int
    source: str = "real"
    defect_type: str = ""


@dataclass(frozen=True)
class GroupSummary:
    group_id: str
    samples: tuple[Sample, ...]
    images: int
    positives: int
    negatives: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize YOLO11n segmentation on BGAD binary masks with capture- and "
            "tool-grouped cross-validation, then train one final model per CV mode."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        default=DEFAULT_SYNTHETIC_DIR,
        help="Directory containing synthetic images/ and masks_png/.",
    )
    parser.add_argument(
        "--synthetic-policy",
        choices=["partitioned", "all", "none"],
        default="partitioned",
        help=(
            "partitioned assigns one balanced synthetic subset to each fold; "
            "all adds every synthetic image to every training fold; none disables it."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help="Target number of completed Optuna trials per CV mode.",
    )
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument(
        "--objective-map50-weight",
        type=float,
        default=DEFAULT_OBJECTIVE_MAP50_WEIGHT,
        help=(
            "Weight of mask mAP50 in the fold score; the remaining weight is "
            "assigned to mask mAP50-95."
        ),
    )
    parser.add_argument(
        "--objective-std-penalty",
        type=float,
        default=DEFAULT_OBJECTIVE_STD_PENALTY,
        help="Penalty applied to cross-fold score standard deviation.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--cv-modes",
        nargs="+",
        default=["capture_grouped", "tool_grouped"],
        choices=["capture_grouped", "tool_grouped"],
    )
    parser.add_argument("--aug-copies", type=int, default=DEFAULT_AUG_COPIES)
    parser.add_argument(
        "--synthetic-aug-copies",
        type=int,
        default=DEFAULT_SYNTHETIC_AUG_COPIES,
        help="Additional Albumentations copies per synthetic training image.",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=DEFAULT_MIN_COMPONENT_AREA,
        help="Ignore isolated mask components smaller than this many pixels.",
    )
    parser.add_argument(
        "--synthetic-min-component-area",
        type=int,
        default=DEFAULT_SYNTHETIC_MIN_COMPONENT_AREA,
        help=(
            "Ignore synthetic mask components smaller than this many source pixels. "
            "The higher default prevents tiny synthetic islands from dominating "
            "instance-segmentation training."
        ),
    )
    parser.add_argument(
        "--synthetic-max-components",
        type=int,
        default=DEFAULT_SYNTHETIC_MAX_COMPONENTS,
        help=(
            "Maximum retained defect components per synthetic image. The largest "
            "components are kept so dense pitting masks cannot dominate training."
        ),
    )
    parser.add_argument(
        "--contour-epsilon",
        type=float,
        default=DEFAULT_CONTOUR_EPSILON,
        help="Polygon simplification as a fraction of contour perimeter.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=3000,
        help="Search width for balancing more than 12 capture groups.",
    )
    parser.add_argument("--keep-trial-data", action="store_true")
    parser.add_argument("--skip-final-training", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if not args.inspect_only and args.n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    if args.epochs < 1:
        raise ValueError("epochs must be positive.")
    if args.batch == 0:
        raise ValueError("batch must be non-zero; use -1 for YOLO auto-batch.")
    if not 0.0 <= args.objective_map50_weight <= 1.0:
        raise ValueError("objective_map50_weight must be between 0 and 1.")
    if args.objective_std_penalty < 0.0:
        raise ValueError("objective_std_penalty must be non-negative.")
    if args.aug_copies < 0:
        raise ValueError("aug_copies must be non-negative.")
    if args.synthetic_aug_copies < 0:
        raise ValueError("synthetic_aug_copies must be non-negative.")
    if args.min_component_area < 1:
        raise ValueError("min_component_area must be positive.")
    if args.synthetic_min_component_area < 1:
        raise ValueError("synthetic_min_component_area must be positive.")
    if args.synthetic_max_components < 1:
        raise ValueError("synthetic_max_components must be positive.")
    if not 0.0 <= args.contour_epsilon <= 0.05:
        raise ValueError("contour_epsilon must be between 0 and 0.05.")
    if args.beam_width < 10:
        raise ValueError("beam_width must be at least 10.")


def check_training_dependencies() -> None:
    required = {
        "optuna": "optuna",
        "ultralytics": "ultralytics",
        "albumentations": "albumentations",
        "cv2": "opencv-python",
    }
    missing = [
        package
        for module, package in required.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise RuntimeError(
            "Missing required training packages: "
            + ", ".join(missing)
            + ". Install them with: python -m pip install -r "
            + "requirements_yolo_optuna.txt"
        )


def tool_id_from_name(name: str) -> str:
    match = re.match(r"^(tool\d+)", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not extract tool ID from filename: {name}")
    return match.group(1).lower()


def capture_id_from_stem(stem: str) -> str:
    capture_id = CAPTURE_SUFFIX.sub("", stem)
    if capture_id == stem:
        raise ValueError(
            f"Could not extract capture ID from {stem}; expected suffix such as "
            "_c000_direct_b01_c00."
        )
    return capture_id


def mask_to_yolo_rows(
    mask: np.ndarray,
    min_component_area: int,
    contour_epsilon: float,
    max_components: int | None = None,
) -> tuple[tuple[str, ...], int, int]:
    import cv2

    binary = (mask > 0).astype(np.uint8)
    height, width = binary.shape
    component_total, component_map, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    polygons: list[tuple[int, int, str]] = []
    dropped = 0

    component_indices = [
        component_idx
        for component_idx in range(1, component_total)
        if int(stats[component_idx, cv2.CC_STAT_AREA]) >= min_component_area
    ]
    dropped += (component_total - 1) - len(component_indices)
    if max_components is not None and len(component_indices) > max_components:
        component_indices.sort(
            key=lambda component_idx: (
                -int(stats[component_idx, cv2.CC_STAT_AREA]),
                component_idx,
            )
        )
        dropped += len(component_indices) - max_components
        component_indices = component_indices[:max_components]

    for component_idx in component_indices:
        component = (component_map == component_idx).astype(np.uint8)
        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            dropped += 1
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, closed=True)
        approximation = cv2.approxPolyDP(
            contour,
            epsilon=contour_epsilon * perimeter,
            closed=True,
        )
        points = approximation.reshape(-1, 2)
        if len(points) < 3:
            points = contour.reshape(-1, 2)
        if len(points) < 3:
            dropped += 1
            continue

        coordinates: list[str] = []
        for x, y in points:
            coordinates.extend(
                [
                    f"{min(max(float(x) / width, 0.0), 1.0):.8f}",
                    f"{min(max(float(y) / height, 0.0), 1.0):.8f}",
                ]
            )
        x, y, _, _ = cv2.boundingRect(contour)
        polygons.append((y, x, "0 " + " ".join(coordinates)))

    polygons.sort(key=lambda item: (item[0], item[1], item[2]))
    rows = tuple(item[2] for item in polygons)
    return rows, component_total - 1, dropped


def discover_samples(
    dataset_dir: Path,
    min_component_area: int,
    contour_epsilon: float,
) -> list[Sample]:
    import cv2

    image_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {image_dir} and {mask_dir} containing paired images and masks."
        )

    images = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    masks = {
        path.stem: path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    missing_masks = sorted(set(images) - set(masks))
    orphan_masks = sorted(set(masks) - set(images))
    if missing_masks or orphan_masks:
        raise ValueError(
            f"Image/mask pairing failed: {len(missing_masks)} missing masks and "
            f"{len(orphan_masks)} orphan masks."
        )
    if not images:
        raise ValueError(f"No images found in {image_dir}.")

    samples: list[Sample] = []
    for stem in sorted(images):
        image = cv2.imread(str(images[stem]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(masks[stem]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Could not read image/mask pair for {stem}.")
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Shape mismatch for {stem}: image={image.shape[:2]}, mask={mask.shape}."
            )
        values = set(map(int, np.unique(mask)))
        if not values.issubset({0, 255}):
            raise ValueError(f"Mask {masks[stem]} is not binary 0/255: {sorted(values)}")

        label_rows, component_count, dropped = mask_to_yolo_rows(
            mask,
            min_component_area=min_component_area,
            contour_epsilon=contour_epsilon,
        )
        has_defect = bool(np.any(mask > 0))
        if has_defect and not label_rows:
            raise ValueError(
                f"Positive mask {masks[stem]} produced no YOLO polygons. "
                "Lower --min-component-area or --contour-epsilon."
            )
        samples.append(
            Sample(
                image=images[stem],
                mask=masks[stem],
                tool_id=tool_id_from_name(stem),
                capture_id=capture_id_from_stem(stem),
                has_defect=has_defect,
                label_rows=label_rows,
                component_count=component_count,
                dropped_components=dropped,
            )
        )
    return samples


def synthetic_defect_type_from_stem(stem: str) -> str:
    match = re.match(
        r"^synth_bevel_gear_spindle_closeup_(.+)_\d+$",
        stem,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Could not extract synthetic defect type from: {stem}")
    return match.group(1).lower()


def discover_synthetic_samples(
    dataset_dir: Path,
    min_component_area: int,
    contour_epsilon: float,
    max_components: int = DEFAULT_SYNTHETIC_MAX_COMPONENTS,
) -> list[Sample]:
    import cv2

    image_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks_png"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {image_dir} and {mask_dir} containing paired images and masks."
        )

    images = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    masks = {
        path.stem: path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    missing_masks = sorted(set(images) - set(masks))
    orphan_masks = sorted(set(masks) - set(images))
    if missing_masks or orphan_masks:
        raise ValueError(
            f"Synthetic image/mask pairing failed: {len(missing_masks)} missing "
            f"masks and {len(orphan_masks)} orphan masks."
        )
    if not images:
        raise ValueError(f"No synthetic images found in {image_dir}.")

    samples: list[Sample] = []
    for stem in sorted(images):
        image = cv2.imread(str(images[stem]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(masks[stem]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Could not read synthetic image/mask pair for {stem}.")
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Synthetic shape mismatch for {stem}: "
                f"image={image.shape[:2]}, mask={mask.shape}."
            )
        values = set(map(int, np.unique(mask)))
        if not values.issubset({0, 255}):
            raise ValueError(
                f"Synthetic mask {masks[stem]} is not binary 0/255: {sorted(values)}"
            )

        label_rows, component_count, dropped = mask_to_yolo_rows(
            mask,
            min_component_area=min_component_area,
            contour_epsilon=contour_epsilon,
            max_components=max_components,
        )
        if not np.any(mask > 0) or not label_rows:
            raise ValueError(
                f"Synthetic defect mask {masks[stem]} produced no YOLO polygons."
            )
        samples.append(
            Sample(
                image=images[stem],
                mask=masks[stem],
                tool_id="synthetic",
                capture_id=stem,
                has_defect=True,
                label_rows=label_rows,
                component_count=component_count,
                dropped_components=dropped,
                source="synthetic",
                defect_type=synthetic_defect_type_from_stem(stem),
            )
        )
    return samples


def group_samples(
    samples: list[Sample],
    group_fn: Callable[[Sample], str],
) -> list[GroupSummary]:
    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        grouped.setdefault(group_fn(sample), []).append(sample)
    return [
        GroupSummary(
            group_id=group_id,
            samples=tuple(sorted(group, key=lambda sample: sample.image.name)),
            images=len(group),
            positives=sum(sample.has_defect for sample in group),
            negatives=sum(not sample.has_defect for sample in group),
        )
        for group_id, group in sorted(grouped.items())
    ]


def fold_balance_score(
    counts: list[list[int]],
    totals: tuple[int, int, int],
) -> float:
    score = 0.0
    weights = (1.0, 1.5, 1.0)
    n_folds = len(counts)
    for metric_idx, total in enumerate(totals):
        target = total / n_folds
        if not total:
            continue
        score += weights[metric_idx] * sum(
            ((fold[metric_idx] - target) / total) ** 2 for fold in counts
        )
    return score


def assignment_counts(
    assignment: tuple[int, ...],
    groups: list[GroupSummary],
    n_folds: int,
) -> list[list[int]]:
    counts = [[0, 0, 0] for _ in range(n_folds)]
    for group, fold_idx in zip(groups, assignment):
        counts[fold_idx][0] += group.images
        counts[fold_idx][1] += group.positives
        counts[fold_idx][2] += group.negatives
    return counts


def assignment_is_valid(
    assignment: tuple[int, ...],
    groups: list[GroupSummary],
    n_folds: int,
) -> bool:
    if set(assignment) != set(range(n_folds)):
        return False
    counts = assignment_counts(assignment, groups, n_folds)
    return all(images > 0 and positives > 0 and negatives > 0 for images, positives, negatives in counts)


def exact_group_assignment(
    groups: list[GroupSummary],
    n_folds: int,
) -> tuple[int, ...]:
    totals = (
        sum(group.images for group in groups),
        sum(group.positives for group in groups),
        sum(group.negatives for group in groups),
    )
    best_assignment: tuple[int, ...] | None = None
    best_score = math.inf
    for assignment in itertools.product(range(n_folds), repeat=len(groups)):
        if not assignment_is_valid(assignment, groups, n_folds):
            continue
        score = fold_balance_score(
            assignment_counts(assignment, groups, n_folds),
            totals,
        )
        if score < best_score or (
            math.isclose(score, best_score) and assignment < (best_assignment or assignment)
        ):
            best_score = score
            best_assignment = tuple(assignment)
    if best_assignment is None:
        raise ValueError(
            f"Could not assign {len(groups)} groups to {n_folds} folds with both classes."
        )
    return best_assignment


def beam_group_assignment(
    groups: list[GroupSummary],
    n_folds: int,
    beam_width: int,
) -> tuple[int, ...]:
    ordered = sorted(
        groups,
        key=lambda group: (
            -max(group.positives, group.negatives),
            -group.images,
            group.group_id,
        ),
    )
    totals = (
        sum(group.images for group in ordered),
        sum(group.positives for group in ordered),
        sum(group.negatives for group in ordered),
    )
    states: list[tuple[tuple[int, ...], list[list[int]]]] = [
        (tuple(), [[0, 0, 0] for _ in range(n_folds)])
    ]

    for group in ordered:
        expanded: list[tuple[float, tuple[int, ...], list[list[int]]]] = []
        for assignment, counts in states:
            max_used = max(assignment, default=-1)
            allowed_folds = range(min(n_folds, max_used + 2))
            for fold_idx in allowed_folds:
                next_counts = [fold.copy() for fold in counts]
                next_counts[fold_idx][0] += group.images
                next_counts[fold_idx][1] += group.positives
                next_counts[fold_idx][2] += group.negatives
                next_assignment = assignment + (fold_idx,)
                processed_totals = tuple(sum(fold[idx] for fold in next_counts) for idx in range(3))
                score = fold_balance_score(next_counts, processed_totals)
                expanded.append((score, next_assignment, next_counts))
        expanded.sort(key=lambda item: (item[0], item[1]))
        states = [
            (assignment, counts)
            for _, assignment, counts in expanded[:beam_width]
        ]

    valid = [
        (
            fold_balance_score(counts, totals),
            assignment,
        )
        for assignment, counts in states
        if assignment_is_valid(assignment, ordered, n_folds)
    ]
    if not valid:
        raise ValueError(
            "Beam search could not form balanced folds. Increase --beam-width or "
            "reduce --n-folds."
        )
    valid.sort(key=lambda item: (item[0], item[1]))
    ordered_assignment = valid[0][1]
    fold_by_group = {
        group.group_id: fold_idx
        for group, fold_idx in zip(ordered, ordered_assignment)
    }
    return tuple(fold_by_group[group.group_id] for group in groups)


def make_cv_folds(
    samples: list[Sample],
    mode: str,
    n_folds: int,
    beam_width: int,
) -> list[dict[str, list[Sample]]]:
    group_fn = (
        (lambda sample: sample.capture_id)
        if mode == "capture_grouped"
        else (lambda sample: sample.tool_id)
    )
    groups = group_samples(samples, group_fn)
    if len(groups) < n_folds:
        raise ValueError(
            f"{mode} has only {len(groups)} groups for {n_folds} folds."
        )
    assignment = (
        exact_group_assignment(groups, n_folds)
        if len(groups) <= 12
        else beam_group_assignment(groups, n_folds, beam_width)
    )

    validation_groups = [
        {group.group_id for group, fold_idx in zip(groups, assignment) if fold_idx == idx}
        for idx in range(n_folds)
    ]
    folds: list[dict[str, list[Sample]]] = []
    for fold_idx, held_out in enumerate(validation_groups):
        train = [sample for sample in samples if group_fn(sample) not in held_out]
        val = [sample for sample in samples if group_fn(sample) in held_out]
        rng = random.Random(BASE_SEED + fold_idx)
        rng.shuffle(train)
        rng.shuffle(val)
        folds.append({"train": train, "val": val})
    validate_cv_folds(samples, folds, group_fn, mode)
    return folds


def stratified_synthetic_partitions(
    samples: list[Sample],
    n_folds: int,
    seed: int = BASE_SEED,
) -> list[list[Sample]]:
    partitions: list[list[Sample]] = [[] for _ in range(n_folds)]
    by_defect: dict[str, list[Sample]] = {}
    for sample in samples:
        if sample.source != "synthetic":
            raise ValueError("Synthetic partitioning received a non-synthetic sample.")
        by_defect.setdefault(sample.defect_type, []).append(sample)

    for defect_type, defect_samples in sorted(by_defect.items()):
        defect_seed = int.from_bytes(
            hashlib.sha256(defect_type.encode("utf-8")).digest()[:4],
            byteorder="big",
        )
        shuffled = sorted(defect_samples, key=lambda sample: sample.image.name)
        random.Random(seed + defect_seed).shuffle(shuffled)
        for sample_idx, sample in enumerate(shuffled):
            partitions[sample_idx % n_folds].append(sample)

    for fold_idx, partition in enumerate(partitions):
        random.Random(seed + fold_idx).shuffle(partition)
    return partitions


def add_synthetic_training_data(
    folds: list[dict[str, list[Sample]]],
    synthetic_samples: list[Sample],
    policy: str,
) -> list[dict[str, list[Sample]]]:
    if policy not in {"partitioned", "all", "none"}:
        raise ValueError(f"Unknown synthetic policy: {policy}")
    if policy == "none":
        additions = [[] for _ in folds]
    elif policy == "all":
        additions = [list(synthetic_samples) for _ in folds]
    else:
        additions = stratified_synthetic_partitions(
            synthetic_samples,
            len(folds),
        )

    enriched: list[dict[str, list[Sample]]] = []
    for fold_idx, (fold, synthetic_train) in enumerate(zip(folds, additions)):
        train = list(fold["train"]) + list(synthetic_train)
        random.Random(BASE_SEED + 50_000 + fold_idx).shuffle(train)
        enriched.append({"train": train, "val": list(fold["val"])})
    validate_synthetic_training_only(enriched, synthetic_samples, policy)
    return enriched


def validate_synthetic_training_only(
    folds: list[dict[str, list[Sample]]],
    synthetic_samples: list[Sample],
    policy: str,
) -> None:
    expected = {sample.image for sample in synthetic_samples}
    occurrences: list[Path] = []
    for fold_idx, fold in enumerate(folds):
        synthetic_val = [sample for sample in fold["val"] if sample.source == "synthetic"]
        if synthetic_val:
            raise ValueError(
                f"Fold {fold_idx} contains synthetic validation images."
            )
        train_synthetic = [
            sample for sample in fold["train"] if sample.source == "synthetic"
        ]
        occurrences.extend(sample.image for sample in train_synthetic)
        if policy == "none" and train_synthetic:
            raise ValueError(f"Fold {fold_idx} contains disabled synthetic data.")
        if policy == "all" and {sample.image for sample in train_synthetic} != expected:
            raise ValueError(f"Fold {fold_idx} does not contain all synthetic data.")

    if policy == "partitioned":
        if len(occurrences) != len(set(occurrences)) or set(occurrences) != expected:
            raise ValueError(
                "Partitioned synthetic samples must occur in exactly one training fold."
            )


def validate_cv_folds(
    samples: list[Sample],
    folds: list[dict[str, list[Sample]]],
    group_fn: Callable[[Sample], str],
    mode: str,
) -> None:
    all_images = {sample.image for sample in samples}
    validation_images: list[Path] = []
    for fold_idx, fold in enumerate(folds):
        train_images = {sample.image for sample in fold["train"]}
        val_images = {sample.image for sample in fold["val"]}
        if train_images & val_images:
            raise ValueError(f"{mode} fold {fold_idx} leaks images across train and val.")
        if train_images | val_images != all_images:
            raise ValueError(f"{mode} fold {fold_idx} does not cover all images.")
        if not any(sample.has_defect for sample in fold["val"]):
            raise ValueError(f"{mode} fold {fold_idx} has no positive validation image.")
        if not any(not sample.has_defect for sample in fold["val"]):
            raise ValueError(f"{mode} fold {fold_idx} has no negative validation image.")
        train_groups = {group_fn(sample) for sample in fold["train"]}
        val_groups = {group_fn(sample) for sample in fold["val"]}
        if train_groups & val_groups:
            raise ValueError(f"{mode} fold {fold_idx} leaks groups across train and val.")
        validation_images.extend(val_images)
    if len(validation_images) != len(set(validation_images)):
        raise ValueError(f"{mode} validation folds overlap.")
    if set(validation_images) != all_images:
        raise ValueError(f"{mode} does not validate every image exactly once.")


def write_fold_manifest(
    folds: list[dict[str, list[Sample]]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "partition",
                "source",
                "defect_type",
                "tool",
                "capture",
                "has_defect",
                "components",
                "dropped_components",
                "image",
                "mask",
            ],
        )
        writer.writeheader()
        for fold_idx, fold in enumerate(folds):
            for partition in PARTITIONS:
                for sample in fold[partition]:
                    writer.writerow(
                        {
                            "fold": fold_idx,
                            "partition": partition,
                            "source": sample.source,
                            "defect_type": sample.defect_type,
                            "tool": sample.tool_id,
                            "capture": sample.capture_id,
                            "has_defect": int(sample.has_defect),
                            "components": sample.component_count,
                            "dropped_components": sample.dropped_components,
                            "image": str(sample.image),
                            "mask": str(sample.mask),
                        }
                    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_cv_assignment_manifest(
    folds_by_mode: dict[str, list[dict[str, list[Sample]]]],
    dataset_dir: Path,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hash_cache: dict[Path, str] = {}
    rows: list[dict[str, Any]] = []
    for mode, folds in folds_by_mode.items():
        for fold_idx, fold in enumerate(folds):
            for sample in fold["val"]:
                if sample.image not in hash_cache:
                    hash_cache[sample.image] = sha256_file(sample.image)
                if sample.mask not in hash_cache:
                    hash_cache[sample.mask] = sha256_file(sample.mask)
                image_hash = hash_cache[sample.image]
                mask_hash = hash_cache[sample.mask]
                rows.append(
                    {
                        "cv_mode": mode,
                        "validation_fold": fold_idx,
                        "tool": sample.tool_id,
                        "capture": sample.capture_id,
                        "has_defect": int(sample.has_defect),
                        "components": sample.component_count,
                        "dropped_components": sample.dropped_components,
                        "image_relpath": sample.image.relative_to(dataset_dir).as_posix(),
                        "mask_relpath": sample.mask.relative_to(dataset_dir).as_posix(),
                        "image_sha256": image_hash,
                        "mask_sha256": mask_hash,
                    }
                )
    rows.sort(
        key=lambda row: (
            row["cv_mode"],
            int(row["validation_fold"]),
            row["image_relpath"],
        )
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cv_mode",
                "validation_fold",
                "tool",
                "capture",
                "has_defect",
                "components",
                "dropped_components",
                "image_relpath",
                "mask_relpath",
                "image_sha256",
                "mask_sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_synthetic_assignment_manifest(
    folds_by_mode: dict[str, list[dict[str, list[Sample]]]],
    synthetic_dir: Path,
    policy: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hash_cache: dict[Path, str] = {}
    rows: list[dict[str, Any]] = []
    for mode, folds in folds_by_mode.items():
        for fold_idx, fold in enumerate(folds):
            for sample in fold["train"]:
                if sample.source != "synthetic":
                    continue
                if sample.image not in hash_cache:
                    hash_cache[sample.image] = sha256_file(sample.image)
                if sample.mask not in hash_cache:
                    hash_cache[sample.mask] = sha256_file(sample.mask)
                rows.append(
                    {
                        "cv_mode": mode,
                        "training_fold": fold_idx,
                        "policy": policy,
                        "defect_type": sample.defect_type,
                        "components": sample.component_count,
                        "dropped_components": sample.dropped_components,
                        "image_relpath": sample.image.relative_to(
                            synthetic_dir
                        ).as_posix(),
                        "mask_relpath": sample.mask.relative_to(
                            synthetic_dir
                        ).as_posix(),
                        "image_sha256": hash_cache[sample.image],
                        "mask_sha256": hash_cache[sample.mask],
                    }
                )
    rows.sort(
        key=lambda row: (
            row["cv_mode"],
            int(row["training_fold"]),
            row["defect_type"],
            row["image_relpath"],
        )
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "cv_mode",
            "training_fold",
            "policy",
            "defect_type",
            "components",
            "dropped_components",
            "image_relpath",
            "mask_relpath",
            "image_sha256",
            "mask_sha256",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_inspection(
    samples: list[Sample],
    synthetic_samples: list[Sample],
    synthetic_policy: str,
    folds_by_mode: dict[str, Any],
) -> None:
    print("BGAD dataset inspection")
    print(f"images/masks: {len(samples)}")
    print(f"defective: {sum(sample.has_defect for sample in samples)}")
    print(f"non-defective: {sum(not sample.has_defect for sample in samples)}")
    print(f"tools: {len({sample.tool_id for sample in samples})}")
    print(f"captures: {len({sample.capture_id for sample in samples})}")
    print(f"YOLO polygons: {sum(len(sample.label_rows) for sample in samples)}")
    print(f"dropped tiny components: {sum(sample.dropped_components for sample in samples)}")
    print(f"synthetic policy: {synthetic_policy}")
    print(f"synthetic images/masks: {len(synthetic_samples)}")
    print(
        "synthetic YOLO polygons: "
        f"{sum(len(sample.label_rows) for sample in synthetic_samples)}"
    )
    synthetic_types = group_samples(
        synthetic_samples,
        lambda sample: sample.defect_type,
    )
    if synthetic_types:
        print(
            "synthetic defect types: "
            + ", ".join(
                f"{group.group_id}={group.images}" for group in synthetic_types
            )
        )

    for mode, folds in folds_by_mode.items():
        print(f"\n{mode}")
        for fold_idx, fold in enumerate(folds):
            val = fold["val"]
            synthetic_train = [
                sample for sample in fold["train"] if sample.source == "synthetic"
            ]
            tools = ",".join(sorted({sample.tool_id for sample in val}))
            captures = len({sample.capture_id for sample in val})
            print(
                f"fold {fold_idx}: val_images={len(val)}, "
                f"positive={sum(sample.has_defect for sample in val)}, "
                f"negative={sum(not sample.has_defect for sample in val)}, "
                f"synthetic_train={len(synthetic_train)}, "
                f"captures={captures}, tools={tools}"
            )


def sample_config(trial: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        # Small, pretrained-data-friendly model search. Most YOLO defaults stay fixed.
        "lr0": trial.suggest_float("lr0", 1e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "mosaic": trial.suggest_categorical("mosaic", [0.0, 0.5, 1.0]),
        "copy_paste": trial.suggest_categorical("copy_paste", [0.0, 0.1, 0.3]),
        "scale": trial.suggest_float("scale", 0.2, 0.7),
        "preprocess_mode": trial.suggest_categorical(
            "preprocess_mode",
            ["none", "clahe", "gamma", "strong_contrast"],
        ),
        "aug_brightness_contrast_p": trial.suggest_float(
            "aug_brightness_contrast_p",
            0.0,
            0.6,
        ),
        "aug_contrast_limit": trial.suggest_float(
            "aug_contrast_limit",
            0.05,
            0.4,
        ),
        "aug_gamma_p": trial.suggest_float("aug_gamma_p", 0.0, 0.4),
        "aug_gamma_delta": trial.suggest_int("aug_gamma_delta", 10, 40),
        "aug_clahe_p": trial.suggest_float("aug_clahe_p", 0.0, 0.4),
        "aug_sharpen_p": trial.suggest_float("aug_sharpen_p", 0.0, 0.3),
        "aug_blur_noise_p": trial.suggest_float("aug_blur_noise_p", 0.0, 0.3),
    }
    if config["preprocess_mode"] == "clahe":
        config["pre_clahe_clip_limit"] = trial.suggest_float(
            "pre_clahe_clip_limit",
            1.0,
            4.0,
        )
    elif config["preprocess_mode"] == "gamma":
        config["pre_gamma"] = trial.suggest_float("pre_gamma", 0.7, 1.5)
    elif config["preprocess_mode"] == "strong_contrast":
        config["pre_contrast_factor"] = trial.suggest_float(
            "pre_contrast_factor",
            1.1,
            2.0,
        )
    return config


def apply_preprocessing(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    import cv2

    mode = config["preprocess_mode"]
    if mode == "none":
        return image
    if mode == "clahe":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=float(config.get("pre_clahe_clip_limit", 2.0)),
            tileGridSize=(8, 8),
        )
        enhanced_l = clahe.apply(l_channel)
        return cv2.cvtColor(
            cv2.merge((enhanced_l, a_channel, b_channel)),
            cv2.COLOR_LAB2BGR,
        )
    if mode == "gamma":
        gamma = float(config.get("pre_gamma", 1.0))
        inverse_gamma = 1.0 / max(gamma, 1e-6)
        table = np.array(
            [((value / 255.0) ** inverse_gamma) * 255 for value in range(256)]
        ).astype(np.uint8)
        return cv2.LUT(image, table)
    if mode == "strong_contrast":
        factor = float(config.get("pre_contrast_factor", 1.4))
        adjusted = (image.astype(np.float32) - 127.5) * factor + 127.5
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    raise ValueError(f"Unknown preprocess_mode: {mode}")


def make_train_augmentation(config: dict[str, Any], seed: int) -> Any:
    import albumentations as A

    gamma_delta = int(config["aug_gamma_delta"])
    if "var_limit" in inspect.signature(A.GaussNoise).parameters:
        noise_kwargs: dict[str, Any] = {"var_limit": (5.0, 30.0)}
    else:
        noise_kwargs = {"std_range": (0.02, 0.08)}

    compose = A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=float(config["aug_contrast_limit"]),
                p=float(config["aug_brightness_contrast_p"]),
            ),
            A.RandomGamma(
                gamma_limit=(max(1, 100 - gamma_delta), 100 + gamma_delta),
                p=float(config["aug_gamma_p"]),
            ),
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=float(config["aug_clahe_p"]),
            ),
            A.Sharpen(
                alpha=(0.1, 0.35),
                lightness=(0.8, 1.2),
                p=float(config["aug_sharpen_p"]),
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                    A.GaussNoise(**noise_kwargs, p=0.5),
                ],
                p=float(config["aug_blur_noise_p"]),
            ),
        ],
        **({"seed": seed} if "seed" in inspect.signature(A.Compose).parameters else {}),
    )
    if hasattr(compose, "set_random_seed"):
        compose.set_random_seed(seed)
    return compose


def seed_augmentation(augmentation: Any, seed: int) -> None:
    if hasattr(augmentation, "set_random_seed"):
        augmentation.set_random_seed(seed)
    else:
        random.seed(seed)
        np.random.seed(seed)


def write_label(rows: tuple[str, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def write_data_yaml(dataset_root: Path) -> Path:
    data = {
        "path": str(dataset_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["defect"],
    }
    path = dataset_root / "data.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


def materialize_dataset(
    split_data: dict[str, list[Sample]],
    dataset_root: Path,
    config: dict[str, Any],
    aug_copies: int,
    seed: int,
    synthetic_aug_copies: int = DEFAULT_SYNTHETIC_AUG_COPIES,
) -> Path:
    import cv2

    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    for partition in PARTITIONS:
        (dataset_root / partition / "images").mkdir(parents=True, exist_ok=True)
        (dataset_root / partition / "labels").mkdir(parents=True, exist_ok=True)

    augmentation = (
        make_train_augmentation(config, seed)
        if max(aug_copies, synthetic_aug_copies) > 0
        else None
    )
    manifest: list[dict[str, Any]] = []
    for partition in PARTITIONS:
        records = list(split_data[partition])
        random.Random(seed + (0 if partition == "train" else 10_000)).shuffle(records)
        for image_idx, sample in enumerate(records):
            image = cv2.imread(str(sample.image), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image: {sample.image}")
            image = apply_preprocessing(image, config)
            output_suffix = (
                sample.image.suffix.lower()
                if config["preprocess_mode"] == "none"
                else ".png"
            )
            output_stem = sample.image.stem
            output_image = (
                dataset_root / partition / "images" / f"{output_stem}{output_suffix}"
            )
            output_label = dataset_root / partition / "labels" / f"{output_stem}.txt"
            if config["preprocess_mode"] == "none":
                shutil.copy2(sample.image, output_image)
            else:
                if not cv2.imwrite(str(output_image), image):
                    raise IOError(f"Could not write image: {output_image}")
            write_label(sample.label_rows, output_label)
            manifest.append(
                {
                    "partition": partition,
                    "kind": "original",
                    "source": sample.source,
                    "defect_type": sample.defect_type,
                    "tool": sample.tool_id,
                    "capture": sample.capture_id,
                    "has_defect": int(sample.has_defect),
                    "source_image": str(sample.image),
                    "written_image": str(output_image),
                }
            )

            if partition != "train":
                continue
            sample_aug_copies = (
                synthetic_aug_copies
                if sample.source == "synthetic"
                else aug_copies
            )
            for aug_idx in range(sample_aug_copies):
                aug_seed = seed + image_idx * 1000 + aug_idx
                if augmentation is None:
                    raise RuntimeError("Augmentation pipeline was not initialized.")
                seed_augmentation(augmentation, aug_seed)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                augmented_rgb = augmentation(image=rgb_image)["image"]
                augmented = cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2BGR)
                aug_stem = f"{output_stem}_alb{aug_idx:02d}"
                aug_image = dataset_root / partition / "images" / f"{aug_stem}.png"
                aug_label = dataset_root / partition / "labels" / f"{aug_stem}.txt"
                if not cv2.imwrite(str(aug_image), augmented):
                    raise IOError(f"Could not write augmented image: {aug_image}")
                write_label(sample.label_rows, aug_label)
                manifest.append(
                    {
                        "partition": partition,
                        "kind": "albumentations",
                        "source": sample.source,
                        "defect_type": sample.defect_type,
                        "tool": sample.tool_id,
                        "capture": sample.capture_id,
                        "has_defect": int(sample.has_defect),
                        "source_image": str(sample.image),
                        "written_image": str(aug_image),
                    }
                )

    with (dataset_root / "manifest.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "partition",
                "kind",
                "source",
                "defect_type",
                "tool",
                "capture",
                "has_defect",
                "source_image",
                "written_image",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)
    return write_data_yaml(dataset_root)


def safe_f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    return 2.0 * precision * recall / denominator if denominator else 0.0


def segmentation_fold_score(
    seg_map50: float,
    seg_map50_95: float,
    map50_weight: float,
) -> float:
    return map50_weight * seg_map50 + (1.0 - map50_weight) * seg_map50_95


def robust_cv_score(
    fold_scores: list[float],
    std_penalty: float,
) -> tuple[float, float, float]:
    if not fold_scores:
        raise ValueError("At least one fold score is required.")
    mean_score = float(np.mean(fold_scores))
    std_score = (
        float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
    )
    return mean_score, std_score, mean_score - std_penalty * std_score


def extract_seg_metrics(metrics: Any) -> dict[str, float]:
    seg = getattr(metrics, "seg", None)
    if seg is not None and getattr(seg, "map", None) is not None:
        precision = float(seg.mp)
        recall = float(seg.mr)
        f1_values = np.asarray(getattr(seg, "f1", []), dtype=float)
        return {
            "seg_precision": precision,
            "seg_recall": recall,
            "seg_f1": (
                float(np.mean(f1_values))
                if f1_values.size
                else safe_f1(precision, recall)
            ),
            "seg_map50": float(seg.map50),
            "seg_map50_95": float(seg.map),
        }
    results = getattr(metrics, "results_dict", None)
    if isinstance(results, dict):
        precision = float(results.get("metrics/precision(M)", 0.0))
        recall = float(results.get("metrics/recall(M)", 0.0))
        return {
            "seg_precision": precision,
            "seg_recall": recall,
            "seg_f1": safe_f1(precision, recall),
            "seg_map50": float(results.get("metrics/mAP50(M)", 0.0)),
            "seg_map50_95": float(results.get("metrics/mAP50-95(M)", 0.0)),
        }
    raise AttributeError("Could not extract Ultralytics segmentation metrics.")


def collect_image_defect_scores(
    model: Any,
    dataset_root: Path,
    partition: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    image_dir = dataset_root / partition / "images"
    label_dir = dataset_root / partition / "labels"
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    predict_kwargs: dict[str, Any] = {
        "source": [str(path) for path in image_paths],
        "stream": True,
        "conf": 0.001,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "save": False,
        "verbose": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    records: list[dict[str, Any]] = []
    for image_idx, result in enumerate(model.predict(**predict_kwargs)):
        if image_idx >= len(image_paths):
            raise RuntimeError(
                f"Prediction returned more than {len(image_paths)} results for {partition}."
            )
        # Ultralytics may report aliases such as image0.jpg for a list source.
        image_path = image_paths[image_idx]
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing evaluation label: {label_path}")
        confidence = 0.0
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is not None and masks is not None and len(boxes) > 0:
            confidence = float(boxes.conf.max().detach().cpu().item())
        records.append(
            {
                "image": image_path.name,
                "ground_truth_defect": int(bool(label_path.read_text(encoding="utf-8").strip())),
                "max_defect_confidence": confidence,
            }
        )
    if len(records) != len(image_paths):
        raise RuntimeError(
            f"Prediction returned {len(records)} results for "
            f"{len(image_paths)} {partition} images."
        )
    return records


def image_defect_metrics(
    records: list[dict[str, Any]],
    threshold: float,
) -> dict[str, float | int]:
    tp = tn = fp = fn = 0
    for record in records:
        truth = bool(int(record["ground_truth_defect"]))
        prediction = float(record["max_defect_confidence"]) >= threshold
        if truth and prediction:
            tp += 1
        elif truth:
            fn += 1
        elif prediction:
            fp += 1
        else:
            tn += 1
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "image_accuracy": (tp + tn) / total if total else 0.0,
        "image_precision": precision,
        "image_recall": recall,
        "image_specificity": specificity,
        "image_f1": safe_f1(precision, recall),
        "image_tp": tp,
        "image_tn": tn,
        "image_fp": fp,
        "image_fn": fn,
        "image_threshold": threshold,
    }


def best_image_f1_threshold(records: list[dict[str, Any]]) -> float:
    scores = {
        float(record["max_defect_confidence"])
        for record in records
        if float(record["max_defect_confidence"]) >= 0.001
    }
    candidates = sorted({0.001, 1.0, *scores})
    scored: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        metrics = image_defect_metrics(records, threshold)
        scored.append(
            (
                float(metrics["image_f1"]),
                float(metrics["image_accuracy"]),
                float(metrics["image_precision"]),
                threshold,
            )
        )
    return max(scored)[-1]


def write_image_decisions(
    path: Path,
    records: list[dict[str, Any]],
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "ground_truth_defect",
                "max_defect_confidence",
                "predicted_defect",
                "correct",
            ],
        )
        writer.writeheader()
        for record in records:
            prediction = int(float(record["max_defect_confidence"]) >= threshold)
            truth = int(record["ground_truth_defect"])
            writer.writerow(
                {
                    **record,
                    "predicted_defect": prediction,
                    "correct": int(prediction == truth),
                }
            )


def base_train_kwargs(
    data_yaml: Path,
    run_root: Path,
    run_name: str,
    config: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "task": "segment",
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "project": str(run_root),
        "name": run_name,
        "exist_ok": True,
        "seed": seed,
        "deterministic": True,
        "workers": args.workers,
        "plots": False,
        "verbose": args.verbose,
        "optimizer": "AdamW",
        "lr0": float(config["lr0"]),
        "weight_decay": float(config["weight_decay"]),
        "mosaic": float(config["mosaic"]),
        "copy_paste": float(config["copy_paste"]),
        "scale": float(config["scale"]),
    }
    if args.device is not None:
        kwargs["device"] = args.device
    return kwargs


def append_metric_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_and_evaluate_fold(
    data_yaml: Path,
    run_root: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        **base_train_kwargs(
            data_yaml,
            run_root,
            "train",
            config,
            args,
            seed,
        )
    )
    run_dir = run_root / "train"
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Training produced no best weights: {best_weights}")

    stopper = getattr(getattr(model, "trainer", None), "stopper", None)
    # Ultralytics passes epoch + 1 into EarlyStopping, so best_epoch is already 1-based.
    best_epoch = min(
        max(int(getattr(stopper, "best_epoch", args.epochs)), 1),
        args.epochs,
    )

    eval_model = YOLO(str(best_weights))
    metrics = eval_model.val(
        data=str(data_yaml),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(run_root),
        name="train_val",
        exist_ok=True,
        plots=False,
        verbose=args.verbose,
        workers=args.workers,
        **({"device": args.device} if args.device is not None else {}),
    )
    evaluation: dict[str, Any] = extract_seg_metrics(metrics)
    validation_records = collect_image_defect_scores(
        eval_model,
        data_yaml.parent,
        "val",
        args,
    )
    threshold = best_image_f1_threshold(validation_records)
    evaluation.update(image_defect_metrics(validation_records, threshold))
    evaluation["image_threshold_source"] = "fold_val"
    evaluation["training_best_epoch"] = best_epoch

    write_image_decisions(
        run_root / "train_val_image_decisions.csv",
        validation_records,
        threshold,
    )
    with (run_root / "train_val_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, sort_keys=True)
    return evaluation, best_weights, validation_records


def summarize_best_trial(
    mode_root: Path,
    best_trial: Any,
    n_folds: int,
) -> dict[str, Any]:
    trial_root = mode_root / f"trial_{best_trial.number:04d}"
    records: list[dict[str, Any]] = []
    for fold_idx in range(n_folds):
        path = (
            trial_root
            / f"fold_{fold_idx:02d}"
            / "runs"
            / "train_val_image_decisions.csv"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing best-trial OOF decisions: {path}")
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                records.append(
                    {
                        "image": row["image"],
                        "ground_truth_defect": int(row["ground_truth_defect"]),
                        "max_defect_confidence": float(row["max_defect_confidence"]),
                    }
                )

    names = [record["image"] for record in records]
    if len(names) != len(set(names)):
        raise ValueError("Best-trial OOF predictions contain duplicate images.")
    threshold = best_image_f1_threshold(records)
    pooled_image_metrics = image_defect_metrics(records, threshold)
    write_image_decisions(
        mode_root / "best_oof_image_decisions.csv",
        records,
        threshold,
    )

    fold_evaluations = list(best_trial.user_attrs["fold_evaluations"])
    scalar_metrics = [
        "seg_precision",
        "seg_recall",
        "seg_f1",
        "seg_map50",
        "seg_map50_95",
        "image_accuracy",
        "image_precision",
        "image_recall",
        "image_specificity",
        "image_f1",
    ]
    summary: dict[str, Any] = {
        "best_trial": best_trial.number,
        "n_folds": n_folds,
        "fold_evaluations": fold_evaluations,
        "pooled_oof_image_metrics": pooled_image_metrics,
        "note": (
            "These are cross-validation model-selection metrics, not an independent "
            "external test."
        ),
    }
    for metric_name in scalar_metrics:
        values = [float(evaluation[metric_name]) for evaluation in fold_evaluations]
        summary[f"mean_{metric_name}"] = float(np.mean(values))
        summary[f"std_{metric_name}"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        )
    with (mode_root / "best_oof_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def train_final_model(
    mode: str,
    samples: list[Sample],
    best_config: dict[str, Any],
    best_trial: Any,
    oof_summary: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    from ultralytics import YOLO

    mode_root = args.output_dir / mode
    final_root = mode_root / "final_all_data"
    dataset_root = final_root / "dataset"
    all_samples = list(samples)
    data_yaml = materialize_dataset(
        {"train": all_samples, "val": all_samples},
        dataset_root,
        best_config,
        args.aug_copies,
        BASE_SEED + 100_000,
        synthetic_aug_copies=args.synthetic_aug_copies,
    )

    best_epochs = [
        int(evaluation["training_best_epoch"])
        for evaluation in best_trial.user_attrs["fold_evaluations"]
    ]
    final_epochs = min(
        max(int(round(float(np.median(best_epochs)))), 1),
        args.epochs,
    )
    model = YOLO(args.model)
    kwargs = base_train_kwargs(
        data_yaml,
        final_root / "runs",
        "train_all",
        best_config,
        args,
        BASE_SEED + 100_000,
    )
    kwargs.update(
        {
            "epochs": final_epochs,
            "patience": 0,
            "val": False,
        }
    )
    model.train(**kwargs)
    last_weights = final_root / "runs" / "train_all" / "weights" / "last.pt"
    if not last_weights.exists():
        raise FileNotFoundError(f"Final training produced no last weights: {last_weights}")
    final_model = mode_root / "final_model.pt"
    shutil.copy2(last_weights, final_model)

    metadata = {
        "cv_mode": mode,
        "model": args.model,
        "final_model": str(final_model),
        "final_training_images": len(samples),
        "final_real_training_images": sum(
            sample.source == "real" for sample in samples
        ),
        "final_synthetic_training_images": sum(
            sample.source == "synthetic" for sample in samples
        ),
        "synthetic_policy": args.synthetic_policy,
        "synthetic_aug_copies": args.synthetic_aug_copies,
        "synthetic_min_component_area": args.synthetic_min_component_area,
        "synthetic_max_components": args.synthetic_max_components,
        "objective_map50_weight": args.objective_map50_weight,
        "objective_std_penalty": args.objective_std_penalty,
        "final_epochs": final_epochs,
        "cv_best_epochs": best_epochs,
        "imgsz": args.imgsz,
        "best_trial": best_trial.number,
        "best_config": best_config,
        "inference_preprocess_mode": best_config["preprocess_mode"],
        "image_defect_threshold": oof_summary["pooled_oof_image_metrics"][
            "image_threshold"
        ],
        "metric_note": (
            "Performance is estimated by CV. The final all-data model has no "
            "independent internal test set."
        ),
    }
    with (mode_root / "final_model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    if not args.keep_trial_data:
        shutil.rmtree(dataset_root, ignore_errors=True)


def run_study(
    mode: str,
    folds: list[dict[str, list[Sample]]],
    samples: list[Sample],
    args: argparse.Namespace,
) -> None:
    import optuna

    mode_root = args.output_dir / mode
    mode_root.mkdir(parents=True, exist_ok=True)
    write_fold_manifest(folds, mode_root / "fold_manifest.csv")
    metrics_csv = mode_root / "fold_metrics.csv"
    storage = f"sqlite:///{(mode_root / 'study.db').resolve().as_posix()}"
    study = optuna.create_study(
        study_name=f"bgad_yolo11_seg_{mode}",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=BASE_SEED),
        pruner=optuna.pruners.NopPruner(),
    )
    training_context = {
        "real_validation_images": len(
            {
                sample.image
                for fold in folds
                for sample in fold["val"]
                if sample.source == "real"
            }
        ),
        "synthetic_policy": args.synthetic_policy,
        "synthetic_aug_copies": args.synthetic_aug_copies,
        "synthetic_min_component_area": args.synthetic_min_component_area,
        "synthetic_max_components": args.synthetic_max_components,
        "objective_map50_weight": args.objective_map50_weight,
        "objective_std_penalty": args.objective_std_penalty,
        "synthetic_training_images_per_fold": [
            sum(sample.source == "synthetic" for sample in fold["train"])
            for fold in folds
        ],
        "n_folds": len(folds),
    }
    stored_context = study.user_attrs.get("training_context")
    if study.trials and stored_context != training_context:
        raise RuntimeError(
            f"Existing study {study.study_name} was created with different training "
            "data or synthetic settings. Use a new --output-dir."
        )
    study.set_user_attr("training_context", training_context)

    def objective(trial: Any) -> float:
        config = sample_config(trial)
        trial_root = mode_root / f"trial_{trial.number:04d}"
        trial_root.mkdir(parents=True, exist_ok=True)
        with (trial_root / "params.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)

        fold_evaluations: list[dict[str, Any]] = []
        fold_scores: list[float] = []
        fold_map50_scores: list[float] = []
        fold_map50_95_scores: list[float] = []
        for fold_idx, fold in enumerate(folds):
            fold_root = trial_root / f"fold_{fold_idx:02d}"
            dataset_root = fold_root / "dataset"
            data_yaml = materialize_dataset(
                fold,
                dataset_root,
                config,
                args.aug_copies,
                BASE_SEED + fold_idx,
                synthetic_aug_copies=args.synthetic_aug_copies,
            )
            evaluation, weights, _ = train_and_evaluate_fold(
                data_yaml,
                fold_root / "runs",
                config,
                args,
                BASE_SEED + fold_idx,
            )
            fold_evaluations.append(evaluation)
            map50 = float(evaluation["seg_map50"])
            map50_95 = float(evaluation["seg_map50_95"])
            fold_map50_scores.append(map50)
            fold_map50_95_scores.append(map50_95)
            fold_scores.append(
                segmentation_fold_score(
                    map50,
                    map50_95,
                    args.objective_map50_weight,
                )
            )
            append_metric_row(
                metrics_csv,
                {
                    "trial": trial.number,
                    "cv_mode": mode,
                    "fold": fold_idx,
                    **evaluation,
                    "weights": str(weights),
                    "params_json": str(trial_root / "params.json"),
                },
            )
            if not args.keep_trial_data:
                shutil.rmtree(dataset_root, ignore_errors=True)

        mean_score, std_score, robust_score = robust_cv_score(
            fold_scores,
            args.objective_std_penalty,
        )
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("fold_seg_map50", fold_map50_scores)
        trial.set_user_attr("fold_seg_map50_95", fold_map50_95_scores)
        trial.set_user_attr("fold_evaluations", fold_evaluations)
        trial.set_user_attr("mean_objective_fold_score", mean_score)
        trial.set_user_attr("std_objective_fold_score", std_score)
        trial.set_user_attr("robust_objective_score", robust_score)
        return robust_score

    completed = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    remaining = max(0, args.n_trials - completed)
    if remaining:
        study.optimize(objective, n_trials=remaining)
    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete_trials:
        raise RuntimeError(f"Study {study.study_name} has no completed trials.")

    best_trial = study.best_trial
    with (mode_root / "best_params.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "cv_mode": mode,
                "objective": (
                    f"mean({args.objective_map50_weight:.3f} * validation mask "
                    f"mAP50 + {1.0 - args.objective_map50_weight:.3f} * validation "
                    f"mask mAP50-95) - {args.objective_std_penalty:.3f} * "
                    "cross-fold standard deviation"
                ),
                "best_trial": best_trial.number,
                "best_value_robust_objective": study.best_value,
                "best_params": best_trial.params,
                "training_context": training_context,
                "user_attrs": best_trial.user_attrs,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    study.trials_dataframe().to_csv(mode_root / "trials.csv", index=False)

    oof_summary = summarize_best_trial(mode_root, best_trial, len(folds))
    if not args.skip_final_training:
        train_final_model(
            mode,
            samples,
            best_trial.params,
            best_trial,
            oof_summary,
            args,
        )


def write_dataset_audit(
    samples: list[Sample],
    synthetic_samples: list[Sample],
    synthetic_policy: str,
    min_component_area: int,
    synthetic_min_component_area: int,
    synthetic_max_components: int,
    output_dir: Path,
) -> None:
    by_tool = group_samples(samples, lambda sample: sample.tool_id)
    by_capture = group_samples(samples, lambda sample: sample.capture_id)
    audit = {
        "images": len(samples),
        "positive_images": sum(sample.has_defect for sample in samples),
        "negative_images": sum(not sample.has_defect for sample in samples),
        "tools": len(by_tool),
        "captures": len(by_capture),
        "yolo_polygons": sum(len(sample.label_rows) for sample in samples),
        "source_components": sum(sample.component_count for sample in samples),
        "dropped_components": sum(sample.dropped_components for sample in samples),
        "min_component_area": min_component_area,
        "by_tool": [
            {
                "tool": group.group_id,
                "images": group.images,
                "positives": group.positives,
                "negatives": group.negatives,
            }
            for group in by_tool
        ],
        "by_capture": [
            {
                "capture": group.group_id,
                "images": group.images,
                "positives": group.positives,
                "negatives": group.negatives,
            }
            for group in by_capture
        ],
        "synthetic": {
            "policy": synthetic_policy,
            "min_component_area": synthetic_min_component_area,
            "max_components_per_image": synthetic_max_components,
            "images": len(synthetic_samples),
            "positive_images": sum(
                sample.has_defect for sample in synthetic_samples
            ),
            "yolo_polygons": sum(
                len(sample.label_rows) for sample in synthetic_samples
            ),
            "source_components": sum(
                sample.component_count for sample in synthetic_samples
            ),
            "dropped_components": sum(
                sample.dropped_components for sample in synthetic_samples
            ),
            "by_defect_type": [
                {
                    "defect_type": group.group_id,
                    "images": group.images,
                }
                for group in group_samples(
                    synthetic_samples,
                    lambda sample: sample.defect_type,
                )
            ],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "dataset_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.dataset_dir = args.dataset_dir.resolve()
    args.synthetic_dir = args.synthetic_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    samples = discover_samples(
        args.dataset_dir,
        args.min_component_area,
        args.contour_epsilon,
    )
    synthetic_samples = (
        []
        if args.synthetic_policy == "none"
        else discover_synthetic_samples(
            args.synthetic_dir,
            args.synthetic_min_component_area,
            args.contour_epsilon,
            args.synthetic_max_components,
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_dataset_audit(
        samples,
        synthetic_samples,
        args.synthetic_policy,
        args.min_component_area,
        args.synthetic_min_component_area,
        args.synthetic_max_components,
        args.output_dir,
    )

    folds_by_mode: dict[str, list[dict[str, list[Sample]]]] = {}
    for mode in args.cv_modes:
        real_folds = make_cv_folds(
            samples,
            mode,
            args.n_folds,
            args.beam_width,
        )
        folds = add_synthetic_training_data(
            real_folds,
            synthetic_samples,
            args.synthetic_policy,
        )
        folds_by_mode[mode] = folds
        write_fold_manifest(
            folds,
            args.output_dir / mode / "fold_manifest.csv",
        )
    write_cv_assignment_manifest(
        folds_by_mode,
        args.dataset_dir,
        args.output_dir / "fold_manifest.csv",
    )
    write_synthetic_assignment_manifest(
        folds_by_mode,
        args.synthetic_dir,
        args.synthetic_policy,
        args.output_dir / "synthetic_fold_manifest.csv",
    )
    print_inspection(
        samples,
        synthetic_samples,
        args.synthetic_policy,
        folds_by_mode,
    )

    if args.inspect_only:
        return

    check_training_dependencies()
    final_training_samples = list(samples) + list(synthetic_samples)
    for mode in args.cv_modes:
        run_study(
            mode,
            folds_by_mode[mode],
            final_training_samples,
            args,
        )


if __name__ == "__main__":
    main()
