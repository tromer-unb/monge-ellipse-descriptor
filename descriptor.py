#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monge Porosity Descriptor 2D

Reads binary/segmented rock images from the `structures/` folder,
where the green phase represents pores and the remaining pixels represent
the solid rock matrix, and concatenates one descriptor row per image into
the file `descriptor.csv`.

Expected input:
    structures/
        image_001.png
        image_002.png
        ...

Output:
    descriptor.csv

Dependencies:
    pip install numpy pandas pillow scipy scikit-image

Usage:
    python monge_porosity_descriptor_2d.py

Note:
    The "Monge" descriptor used here is an adaptation for real 2D pores:
    - each pore is detected as a connected component;
    - each pore is approximated by region/ellipse properties;
    - local triplets are formed through Delaunay triangulation;
    - for each triplet, the collinearity deviation of external-homothety-like
      centers is computed using a directional elliptical radius.
"""

from __future__ import annotations

import math
import json
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image

from scipy.spatial import Delaunay
from skimage.measure import label, regionprops


# =========================
# MAIN SETTINGS
# =========================

STRUCTURES_DIR = Path("../poros/poros/")
OUTPUT_CSV = Path("descriptor.csv")

# Segmentation of the green pore phase.
# A pixel is considered pore if G is dominant and sufficiently intense.
GREEN_MIN = 80
GREEN_DOMINANCE = 30

# Removes noise / very small islands.
MIN_PORE_AREA_PX = 20

# Removes pores touching the image border.
# This avoids measuring truncated pores.
REMOVE_BORDER_TOUCHING = True

# Avoids combinatorial explosion.
# Local triplets are extracted from Delaunay triangulation.
MAX_TRIANGLES = 20000

# When two directional radii are almost equal,
# the external homothety point goes very far away.
# These cases are discarded when computing the Monge score.
RADIUS_EPS = 1e-6

# Accepted image extensions.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# =========================
# IMAGE FUNCTIONS
# =========================

def load_rgb_image(path: Path) -> np.ndarray:
    """Loads an image as RGB uint8."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def green_pore_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask of the green pore phase.

    Criterion:
        G >= GREEN_MIN
        G > R + GREEN_DOMINANCE
        G > B + GREEN_DOMINANCE
    """
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    mask = (
        (g >= GREEN_MIN)
        & (g > r + GREEN_DOMINANCE)
        & (g > b + GREEN_DOMINANCE)
    )
    return mask


# =========================
# PORE EXTRACTION
# =========================

def touches_border(bbox: tuple[int, int, int, int], height: int, width: int) -> bool:
    """Checks whether a region touches the image border."""
    min_row, min_col, max_row, max_col = bbox
    return min_row <= 0 or min_col <= 0 or max_row >= height or max_col >= width


def extract_pores(mask: np.ndarray, image_name: str) -> pd.DataFrame:
    """
    Segments connected components and extracts geometric properties.

    Returns a DataFrame with one pore per row.
    """
    height, width = mask.shape
    lab = label(mask, connectivity=2)
    props = regionprops(lab)

    rows = []
    for idx, p in enumerate(props, start=1):
        area = float(p.area)
        if area < MIN_PORE_AREA_PX:
            continue

        if REMOVE_BORDER_TOUCHING and touches_border(p.bbox, height, width):
            continue

        cy, cx = p.centroid

        # regionprops provides major_axis_length and minor_axis_length.
        # Approximate semi-axes:
        a = max(float(p.major_axis_length) / 2.0, 1e-9)
        b = max(float(p.minor_axis_length) / 2.0, 1e-9)

        # Major-axis orientation relative to the vertical image axis, in radians.
        # Converted here for x-y coordinate descriptors.
        # In images, y increases downward; for statistical descriptors this is acceptable,
        # since the important point is angular consistency.
        theta = float(p.orientation)

        equiv_radius = math.sqrt(area / math.pi)

        perimeter = float(getattr(p, "perimeter", 0.0))
        circularity = np.nan
        if perimeter > 0:
            circularity = 4.0 * math.pi * area / (perimeter ** 2)

        rows.append(
            {
                "image": image_name,
                "pore_id": idx,
                "x": float(cx),
                "y": float(cy),
                "area_px": area,
                "equiv_radius_px": float(equiv_radius),
                "major_axis_px": float(p.major_axis_length),
                "minor_axis_px": float(p.minor_axis_length),
                "ellipse_a_px": float(a),
                "ellipse_b_px": float(b),
                "orientation_rad": theta,
                "eccentricity": float(p.eccentricity),
                "solidity": float(p.solidity),
                "circularity": float(circularity) if not np.isnan(circularity) else np.nan,
                "bbox_min_row": int(p.bbox[0]),
                "bbox_min_col": int(p.bbox[1]),
                "bbox_max_row": int(p.bbox[2]),
                "bbox_max_col": int(p.bbox[3]),
            }
        )

    return pd.DataFrame(rows)


# =========================
# MONGE 2D DESCRIPTOR
# =========================

def ellipse_directional_radius(a: float, b: float, theta: float, direction_angle: float) -> float:
    """
    Radius of the ellipse in the direction `direction_angle`.

    Formula:
        r(phi) = 1 / sqrt((cos(phi')/a)^2 + (sin(phi')/b)^2)

    Here, phi' is the angle relative to the major axis of the ellipse.

    Note:
        theta from regionprops follows an image-coordinate convention.
        The goal here is to capture local anisotropy consistently,
        not to reconstruct a perfect CAD geometry.
    """
    phi = direction_angle - theta
    c = math.cos(phi)
    s = math.sin(phi)
    denom = (c / a) ** 2 + (s / b) ** 2
    if denom <= 0:
        return float(min(a, b))
    return float(1.0 / math.sqrt(denom))


def external_homothety_point(
    ci: np.ndarray,
    cj: np.ndarray,
    ri: float,
    rj: float,
) -> np.ndarray | None:
    """
    External homothety center for two circles / directional equivalents.

    H_ij = (ri*Cj - rj*Ci) / (ri - rj)

    Returns None if the radii are practically equal.
    """
    denom = ri - rj
    if abs(denom) < RADIUS_EPS:
        return None
    return (ri * cj - rj * ci) / denom


def point_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point p to the line passing through a-b."""
    ab = b - a
    denom = np.linalg.norm(ab)
    if denom <= 1e-12:
        return float("nan")
    return float(abs(np.cross(ab, p - a)) / denom)


def monge_score_for_triplet(pores: pd.DataFrame, ids: tuple[int, int, int]) -> tuple[float, float] | None:
    """
    Computes the adapted Monge score for a triplet of pores.

    Returns:
        (normalized_score, angle_rad)

    normalized_score:
        distance from the third Monge-like point to the line defined by
        the other two points, normalized by a local scale.

    angle_rad:
        orientation of the main Monge line.
    """
    p0 = pores.iloc[ids[0]]
    p1 = pores.iloc[ids[1]]
    p2 = pores.iloc[ids[2]]

    C = [
        np.array([p0["x"], p0["y"]], dtype=float),
        np.array([p1["x"], p1["y"]], dtype=float),
        np.array([p2["x"], p2["y"]], dtype=float),
    ]

    A = [float(p0["ellipse_a_px"]), float(p1["ellipse_a_px"]), float(p2["ellipse_a_px"])]
    B = [float(p0["ellipse_b_px"]), float(p1["ellipse_b_px"]), float(p2["ellipse_b_px"])]
    T = [float(p0["orientation_rad"]), float(p1["orientation_rad"]), float(p2["orientation_rad"])]

    pairs = [(0, 1), (0, 2), (1, 2)]
    H = []

    for i, j in pairs:
        vec = C[j] - C[i]
        if np.linalg.norm(vec) <= 1e-12:
            return None

        angle_ij = math.atan2(vec[1], vec[0])
        angle_ji = angle_ij + math.pi

        ri = ellipse_directional_radius(A[i], B[i], T[i], angle_ij)
        rj = ellipse_directional_radius(A[j], B[j], T[j], angle_ji)

        hij = external_homothety_point(C[i], C[j], ri, rj)
        if hij is None or not np.all(np.isfinite(hij)):
            return None

        H.append(hij)

    h01, h02, h12 = H

    # Collinearity deviation.
    d = point_line_distance(h12, h01, h02)
    if not np.isfinite(d):
        return None

    # Local scale based on the size of the centroid triangle.
    center_distances = [
        np.linalg.norm(C[1] - C[0]),
        np.linalg.norm(C[2] - C[0]),
        np.linalg.norm(C[2] - C[1]),
    ]
    scale = float(np.mean(center_distances))
    if scale <= 1e-12:
        return None

    score = float(d / scale)

    # Orientation of the main line between h01 and h02.
    v = h02 - h01
    angle = float(math.atan2(v[1], v[0]))
    if not np.isfinite(angle):
        return None

    return score, angle


def local_delaunay_triplets(pores: pd.DataFrame) -> list[tuple[int, int, int]]:
    """
    Generates local triplets using Delaunay triangulation of pore centroids.
    """
    n = len(pores)
    if n < 3:
        return []

    points = pores[["x", "y"]].to_numpy(dtype=float)

    try:
        tri = Delaunay(points)
        triplets = [tuple(map(int, simplex)) for simplex in tri.simplices]
    except Exception:
        # Fallback: uses simple combinations if triangulation fails.
        triplets = list(combinations(range(n), 3))

    if len(triplets) > MAX_TRIANGLES:
        # Deterministic sampling to preserve reproducibility.
        rng = np.random.default_rng(123)
        idx = rng.choice(len(triplets), size=MAX_TRIANGLES, replace=False)
        triplets = [triplets[i] for i in sorted(idx)]

    return triplets


def angular_resultant_strength(angles: np.ndarray) -> float:
    """
    Angular anisotropy index for lines.

    Since lines have 180-degree symmetry, exp(2i theta) is used.
    R close to 0: dispersed orientations.
    R close to 1: dominant orientation.
    """
    if len(angles) == 0:
        return np.nan
    z = np.exp(2j * angles)
    return float(abs(np.mean(z)))


def circular_mean_line_angle(angles: np.ndarray) -> float:
    """
    Mean angle for lines with 180-degree symmetry.
    Returns the value in radians.
    """
    if len(angles) == 0:
        return np.nan
    z = np.mean(np.exp(2j * angles))
    return float(0.5 * np.angle(z))


def summarize_image_descriptors(
    image_path: Path,
    mask: np.ndarray,
    pores: pd.DataFrame,
) -> dict:
    """
    Computes all global descriptors for one image.
    """
    height, width = mask.shape
    total_pixels = int(height * width)
    pore_pixels = int(mask.sum())
    porosity_2d = float(pore_pixels / total_pixels)

    n_pores = int(len(pores))

    summary = {
        "image": image_path.name,
        "width_px": int(width),
        "height_px": int(height),
        "total_pixels": total_pixels,
        "pore_pixels": pore_pixels,
        "porosity_2d": porosity_2d,
        "n_pores": n_pores,
    }

    if n_pores == 0:
        return add_empty_descriptor_fields(summary)

    # Simple size/shape descriptors.
    for col in [
        "area_px",
        "equiv_radius_px",
        "major_axis_px",
        "minor_axis_px",
        "eccentricity",
        "solidity",
        "circularity",
    ]:
        values = pores[col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            summary[f"{col}_mean"] = np.nan
            summary[f"{col}_std"] = np.nan
            summary[f"{col}_median"] = np.nan
            summary[f"{col}_p10"] = np.nan
            summary[f"{col}_p90"] = np.nan
        else:
            summary[f"{col}_mean"] = float(np.mean(values))
            summary[f"{col}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            summary[f"{col}_median"] = float(np.median(values))
            summary[f"{col}_p10"] = float(np.percentile(values, 10))
            summary[f"{col}_p90"] = float(np.percentile(values, 90))

    # Numerical pore density.
    summary["pore_number_density_px2"] = float(n_pores / total_pixels)

    # Local triplets.
    triplets = local_delaunay_triplets(pores)
    scores = []
    angles = []

    for ids in triplets:
        result = monge_score_for_triplet(pores, ids)
        if result is None:
            continue
        score, angle = result
        if np.isfinite(score) and np.isfinite(angle):
            scores.append(score)
            angles.append(angle)

    scores = np.asarray(scores, dtype=float)
    angles = np.asarray(angles, dtype=float)

    summary["n_delaunay_triplets"] = int(len(triplets))
    summary["n_monge_valid_triplets"] = int(len(scores))

    if len(scores) == 0:
        summary.update(monge_empty_fields())
        return summary

    summary["monge_score_mean"] = float(np.mean(scores))
    summary["monge_score_std"] = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    summary["monge_score_median"] = float(np.median(scores))
    summary["monge_score_p10"] = float(np.percentile(scores, 10))
    summary["monge_score_p25"] = float(np.percentile(scores, 25))
    summary["monge_score_p75"] = float(np.percentile(scores, 75))
    summary["monge_score_p90"] = float(np.percentile(scores, 90))
    summary["monge_score_min"] = float(np.min(scores))
    summary["monge_score_max"] = float(np.max(scores))

    # Fractions of almost "Monge-like" triplets.
    summary["monge_frac_score_lt_0_10"] = float(np.mean(scores < 0.10))
    summary["monge_frac_score_lt_0_25"] = float(np.mean(scores < 0.25))
    summary["monge_frac_score_lt_0_50"] = float(np.mean(scores < 0.50))
    summary["monge_frac_score_lt_1_00"] = float(np.mean(scores < 1.00))

    # Anisotropy of Monge lines.
    summary["monge_line_anisotropy_R"] = angular_resultant_strength(angles)
    summary["monge_line_mean_angle_rad"] = circular_mean_line_angle(angles)
    summary["monge_line_mean_angle_deg"] = (
        float(np.degrees(summary["monge_line_mean_angle_rad"]))
        if np.isfinite(summary["monge_line_mean_angle_rad"])
        else np.nan
    )

    return summary


def monge_empty_fields() -> dict:
    """Empty fields when there are no valid Monge triplets."""
    return {
        "monge_score_mean": np.nan,
        "monge_score_std": np.nan,
        "monge_score_median": np.nan,
        "monge_score_p10": np.nan,
        "monge_score_p25": np.nan,
        "monge_score_p75": np.nan,
        "monge_score_p90": np.nan,
        "monge_score_min": np.nan,
        "monge_score_max": np.nan,
        "monge_frac_score_lt_0_10": np.nan,
        "monge_frac_score_lt_0_25": np.nan,
        "monge_frac_score_lt_0_50": np.nan,
        "monge_frac_score_lt_1_00": np.nan,
        "monge_line_anisotropy_R": np.nan,
        "monge_line_mean_angle_rad": np.nan,
        "monge_line_mean_angle_deg": np.nan,
    }


def add_empty_descriptor_fields(summary: dict) -> dict:
    """Fills empty descriptors when no pores are detected."""
    simple_cols = [
        "area_px",
        "equiv_radius_px",
        "major_axis_px",
        "minor_axis_px",
        "eccentricity",
        "solidity",
        "circularity",
    ]
    stats = ["mean", "std", "median", "p10", "p90"]

    for col in simple_cols:
        for st in stats:
            summary[f"{col}_{st}"] = np.nan

    summary["pore_number_density_px2"] = 0.0
    summary["n_delaunay_triplets"] = 0
    summary["n_monge_valid_triplets"] = 0
    summary.update(monge_empty_fields())
    return summary


# =========================
# MAIN LOOP
# =========================

def find_images(folder: Path) -> list[Path]:
    """Lists accepted images inside the folder."""
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder.resolve()}")

    files = [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return files


def process_one_image(path: Path) -> dict:
    """Processes one image and returns one descriptor row."""
    rgb = load_rgb_image(path)
    mask = green_pore_mask(rgb)
    pores = extract_pores(mask, path.name)
    summary = summarize_image_descriptors(path, mask, pores)
    return summary


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    images = find_images(STRUCTURES_DIR)
    if not images:
        raise RuntimeError(
            f"No images found in {STRUCTURES_DIR.resolve()} "
            f"with extensions {sorted(IMAGE_EXTENSIONS)}"
        )

    rows = []
    for i, path in enumerate(images, start=1):
        print(f"[{i}/{len(images)}] Processing: {path.name}")
        try:
            row = process_one_image(path)
            row["status"] = "ok"
            row["error"] = ""
        except Exception as exc:
            row = {
                "image": path.name,
                "status": "error",
                "error": str(exc),
            }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Places main columns at the beginning.
    preferred = [
        "image",
        "status",
        "error",
        "width_px",
        "height_px",
        "porosity_2d",
        "n_pores",
        "pore_pixels",
        "total_pixels",
        "pore_number_density_px2",
        "n_delaunay_triplets",
        "n_monge_valid_triplets",
        "monge_score_mean",
        "monge_score_std",
        "monge_score_median",
        "monge_score_p10",
        "monge_score_p25",
        "monge_score_p75",
        "monge_score_p90",
        "monge_frac_score_lt_0_10",
        "monge_frac_score_lt_0_25",
        "monge_frac_score_lt_0_50",
        "monge_frac_score_lt_1_00",
        "monge_line_anisotropy_R",
        "monge_line_mean_angle_rad",
        "monge_line_mean_angle_deg",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFile saved: {OUTPUT_CSV.resolve()}")
    print(f"Generated rows: {len(df)}")


if __name__ == "__main__":
    main()
