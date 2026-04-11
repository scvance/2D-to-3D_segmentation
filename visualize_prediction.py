from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CONFLICT_COLOR = np.array([255, 95, 0], dtype=np.uint8)
CLASS_COLORS = {
    0: np.array([255, 50, 50], dtype=np.uint8),
    1: np.array([255, 225, 50], dtype=np.uint8),
    2: np.array([109, 255, 50], dtype=np.uint8),
    3: np.array([50, 167, 255], dtype=np.uint8),
}


def derive_sample_name(pred_path: Path) -> str:
    stem = pred_path.stem
    if stem.endswith("_pred"):
        stem = stem[: -len("_pred")]
    if stem.endswith("_labels"):
        stem = stem[: -len("_labels")]
    return stem


def find_point_cloud(sample_name: str, data_root: Path) -> Path:
    matches = sorted(data_root.rglob(f"{sample_name}.csv"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find point cloud CSV for sample '{sample_name}' under {data_root}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple point cloud CSVs for sample '{sample_name}': {matches}"
        )
    return matches[0]


def resolve_prediction_paths(pred: str | None, sample_prefix: str | None, pred_root: Path) -> list[Path]:
    if pred:
        pred_path = Path(pred).expanduser().resolve()
        if not pred_path.is_file():
            raise FileNotFoundError(f"Prediction file not found: {pred_path}")
        return [pred_path]

    if sample_prefix is None:
        raise ValueError("Either --pred or --sample-prefix must be provided.")

    pattern = (
        sample_prefix
        if any(ch in sample_prefix for ch in "*?[]")
        else f"{sample_prefix}*_pred.npy"
    )
    matches = sorted(pred_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not find any prediction files matching '{pattern}' under {pred_root}"
        )
    return matches


def load_prediction_points(
    pred_path: Path, point_cloud_root: Path
) -> tuple[str, Path, np.ndarray, np.ndarray]:
    pred = np.load(pred_path)
    sample_name = derive_sample_name(pred_path)
    point_cloud_path = find_point_cloud(sample_name, point_cloud_root)
    df = pd.read_csv(point_cloud_path, usecols=["x", "y", "z"])
    pc = df[["x", "y", "z"]].to_numpy()
    if pc.shape[0] != pred.shape[0]:
        raise ValueError(
            f"Prediction length ({pred.shape[0]}) does not match point count ({pc.shape[0]}) for {point_cloud_path}"
        )
    return sample_name, point_cloud_path, pc, pred.astype(int)


def point_key(point: np.ndarray, coord_decimals: int) -> tuple[float, float, float]:
    rounded = np.round(point.astype(np.float64), decimals=coord_decimals)
    return tuple(float(value) for value in rounded)


def pred2colors(pred: np.ndarray) -> np.ndarray:
    colors = np.zeros((pred.shape[0], 3), dtype=np.uint8)
    for label, color in CLASS_COLORS.items():
        colors[pred == label] = color
    return colors


def merge_predictions(
    pred_paths: Iterable[Path], point_cloud_root: Path, coord_decimals: int
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    merged: dict[tuple[float, float, float], dict[str, object]] = {}
    partials: list[dict[str, object]] = []
    raw_points = 0

    for pred_path in pred_paths:
        sample_name, point_cloud_path, pc, pred = load_prediction_points(
            pred_path, point_cloud_root
        )
        partials.append(
            {
                "sample_name": sample_name,
                "pred_path": str(pred_path),
                "point_cloud_path": str(point_cloud_path),
                "points": int(pc.shape[0]),
            }
        )
        raw_points += int(pc.shape[0])
        for point, label in zip(pc, pred, strict=True):
            key = point_key(point, coord_decimals)
            entry = merged.get(key)
            if entry is None:
                merged[key] = {
                    "point": point.astype(np.float64),
                    "labels": {int(label)},
                }
            else:
                entry["labels"].add(int(label))

    merged_points = np.empty((len(merged), 3), dtype=np.float64)
    merged_colors = np.empty((len(merged), 3), dtype=np.uint8)
    conflict_count = 0

    for index, entry in enumerate(merged.values()):
        merged_points[index] = entry["point"]
        labels = sorted(entry["labels"])
        if len(labels) == 1:
            merged_colors[index] = pred2colors(np.array(labels, dtype=np.int64))[0]
        else:
            merged_colors[index] = CONFLICT_COLOR
            conflict_count += 1

    stats = {
        "partials": partials,
        "raw_points": raw_points,
        "merged_points": int(merged_points.shape[0]),
        "duplicate_points": int(raw_points - merged_points.shape[0]),
        "conflict_points": conflict_count,
        "conflict_color_rgb": CONFLICT_COLOR.tolist(),
    }
    return merged_points, merged_colors, stats


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors.astype(np.uint8), strict=True):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize or export Pointcept predictions on TomatoWUR partial point clouds."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--pred",
        help="Path to a single *_pred.npy file under exp/.../result",
    )
    input_group.add_argument(
        "--sample-prefix",
        help="Plant prefix such as Harvest_02_PotNr_27 to merge all matching partial predictions",
    )
    parser.add_argument(
        "--pred-root",
        default="exp/ptv3-partial-v1-safe/result",
        help="Root folder containing *_pred.npy files when using --sample-prefix",
    )
    parser.add_argument(
        "--point-cloud-root",
        default="TomatoWUR/data/TomatoWUR/ann_versions/partial-v1/point_clouds",
        help="Root folder containing TomatoWUR partial point cloud CSVs",
    )
    parser.add_argument(
        "--output-ply",
        help="Optional output .ply path for local visualization",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the Polyscope viewer",
    )
    parser.add_argument(
        "--coord-decimals",
        type=int,
        default=6,
        help="Decimal precision used to identify duplicate xyz points when merging partials",
    )
    args = parser.parse_args()

    pred_root = Path(args.pred_root).expanduser().resolve()
    point_cloud_root = Path(args.point_cloud_root).expanduser().resolve()
    pred_paths = resolve_prediction_paths(args.pred, args.sample_prefix, pred_root)

    points, colors, stats = merge_predictions(
        pred_paths, point_cloud_root, coord_decimals=args.coord_decimals
    )

    print(f"Predictions merged: {len(pred_paths)}")
    for partial in stats["partials"]:
        print(
            f"  {partial['sample_name']}: {partial['points']} points "
            f"({partial['pred_path']})"
        )
    print(f"Raw points: {stats['raw_points']}")
    print(f"Merged unique points: {stats['merged_points']}")
    print(f"Duplicate points removed: {stats['duplicate_points']}")
    print(
        f"Conflicting points: {stats['conflict_points']} "
        f"(colored RGB {stats['conflict_color_rgb']})"
    )
    if args.output_ply:
        output_ply = Path(args.output_ply).expanduser().resolve()
        write_ply(output_ply, points, colors)
        print(f"Wrote colored point cloud: {output_ply}")
    if not args.no_show:
        from TomatoWUR.scripts import visualize_examples as ve

        ve.vis(pc=points, colors=colors)


if __name__ == "__main__":
    main()
