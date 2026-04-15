import csv
import json
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose


@dataclass(frozen=True)
class TrajectoryFrame:
    frame_name: str
    frame_index: int
    point_cloud_path: str
    label_path: str


@dataclass(frozen=True)
class TrajectorySample:
    plant: str
    sequence_id: str
    frames: tuple[TrajectoryFrame, ...]


@contextmanager
def preserve_random_state(seed: int):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


@DATASETS.register_module()
class TomatoWURTrajectoryCSV(Dataset):
    def __init__(
        self,
        split="train",
        data_root=None,
        transform=None,
        lr_file=None,
        la_file=None,
        ignore_index=-1,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
        min_rows=3,
        min_voxels=2,
        min_points_after_transform=2,
        consistent_trajectory_transform=True,
        trajectory_retry_count=10,
    ):
        super().__init__()
        if lr_file is None:
            raise ValueError("TomatoWURTrajectoryCSV requires lr_file to be provided.")
        if cache:
            raise NotImplementedError(
                "TomatoWURTrajectoryCSV does not support cache=True."
            )
        if test_mode:
            raise NotImplementedError(
                "TomatoWURTrajectoryCSV currently expects test_mode=False. "
                "Use the trajectory tester/evaluator with regular frame transforms."
            )

        self.data_root = data_root
        self.split = split
        self.transform_cfg = transform
        self.transform = Compose(transform)
        self.cache = cache
        self.min_rows = min_rows
        self.min_voxels = min_voxels
        self.min_points_after_transform = min_points_after_transform
        self.loop = loop
        self.test_mode = test_mode
        self.test_cfg = test_cfg
        self.filter_grid_size = self._infer_grid_size(transform)
        self.consistent_trajectory_transform = consistent_trajectory_transform
        self.trajectory_retry_count = max(1, trajectory_retry_count)
        self.lr_file = str(lr_file)
        self.la = torch.load(la_file) if la_file else None
        self.ignore_index = ignore_index

        (
            self.trajectories,
            invalid_frame_names,
            dropped_trajectories,
        ) = self._load_trajectories(Path(lr_file))

        logger = get_root_logger()
        if invalid_frame_names:
            preview = ", ".join(invalid_frame_names[:5])
            logger.warning(
                "Filtered %s invalid TomatoWUR trajectory frames from %s with min_rows=%s, min_voxels=%s. Examples: %s",
                len(invalid_frame_names),
                lr_file,
                self.min_rows,
                self.min_voxels,
                preview,
            )
        if dropped_trajectories:
            preview = ", ".join(dropped_trajectories[:5])
            logger.warning(
                "Dropped %s empty TomatoWUR trajectories from %s after frame filtering. Examples: %s",
                len(dropped_trajectories),
                lr_file,
                preview,
            )
        logger.info(
            "Totally %s x %s trajectories in %s set.",
            len(self.trajectories),
            self.loop,
            split,
        )
        if len(self.trajectories) == 0:
            raise RuntimeError(f"No valid TomatoWUR trajectories found in {lr_file}")

    def _load_trajectories(
        self, lr_path: Path
    ) -> tuple[list[TrajectorySample], list[str], list[str]]:
        with lr_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        trajectories: list[TrajectorySample] = []
        invalid_frame_names: list[str] = []
        dropped_trajectories: list[str] = []
        for item in data:
            sequence_id = item["sequence_id"]
            plant = item["plant"]
            frames = item.get("frames", [])
            if item.get("num_frames", len(frames)) != len(frames):
                raise RuntimeError(
                    f"Trajectory {sequence_id} in {lr_path} reports num_frames="
                    f"{item.get('num_frames')} but contains {len(frames)} frame entries."
                )
            parsed_frames: list[TrajectoryFrame] = []
            seen_indices: set[int] = set()
            for frame in frames:
                frame_name = frame["frame_name"]
                frame_index = int(frame["frame_index"])
                if frame_index in seen_indices:
                    raise RuntimeError(
                        f"Trajectory {sequence_id} in {lr_path} contains duplicate "
                        f"frame_index={frame_index}."
                    )
                seen_indices.add(frame_index)
                point_cloud_path = str((lr_path.parent / frame["file_name"]).resolve())
                label_path = str((lr_path.parent / frame["sem_seg_file_name"]).resolve())
                if self._sample_is_valid(point_cloud_path, label_path):
                    parsed_frames.append(
                        TrajectoryFrame(
                            frame_name=frame_name,
                            frame_index=frame_index,
                            point_cloud_path=point_cloud_path,
                            label_path=label_path,
                        )
                    )
                else:
                    invalid_frame_names.append(frame_name)
            parsed_frames.sort(key=lambda frame: (frame.frame_index, frame.frame_name))
            if parsed_frames:
                trajectories.append(
                    TrajectorySample(
                        plant=plant,
                        sequence_id=sequence_id,
                        frames=tuple(parsed_frames),
                    )
                )
            else:
                dropped_trajectories.append(sequence_id)
        return trajectories, invalid_frame_names, dropped_trajectories

    def _sample_is_valid(self, pc_path: str, seg_path: str) -> bool:
        return self._csv_meets_minimum(
            seg_path, min_rows=self.min_rows
        ) and self._csv_meets_minimum(
            pc_path,
            min_rows=self.min_rows,
            min_voxels=self.min_voxels,
            grid_size=self.filter_grid_size,
        )

    @staticmethod
    def _infer_grid_size(transform_cfg):
        if not transform_cfg:
            return None
        for item in transform_cfg:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "GridSample" and "grid_size" in item:
                return item["grid_size"]
        return None

    @staticmethod
    def _csv_meets_minimum(path, min_rows=1, min_voxels=1, grid_size=None):
        row_count = 0
        voxels = set()
        try:
            with open(path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    row_count += 1
                    if grid_size is not None and min_voxels > 1:
                        voxels.add(
                            (
                                math.floor(float(row["x"]) / grid_size),
                                math.floor(float(row["y"]) / grid_size),
                                math.floor(float(row["z"]) / grid_size),
                            )
                        )
                    if row_count >= min_rows and len(voxels) >= min_voxels:
                        return True
        except OSError:
            return False
        if grid_size is None or min_voxels <= 1:
            return row_count >= min_rows
        return row_count >= min_rows and len(voxels) >= min_voxels

    def _load_frame_data(self, frame: TrajectoryFrame) -> dict:
        semantic = pd.read_csv(frame.label_path, usecols=["semantic"])
        point_cloud = pd.read_csv(
            frame.point_cloud_path,
            usecols=["x", "y", "z", "blue", "green", "red", "nx", "ny", "nz"],
        )
        data = pd.concat([semantic, point_cloud], axis=1)
        coord = data[["x", "y", "z"]].values
        color = data[["red", "green", "blue"]].values
        normal = data[["nx", "ny", "nz"]].values
        segment = data["semantic"].astype(int).values.reshape([-1]) - 1
        instance = np.ones(coord.shape[0]) * -1
        return dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=frame.frame_name,
        )

    def _prepare_transformed_frame(
        self,
        trajectory: TrajectorySample,
        frame: TrajectoryFrame,
        frame_position: int,
        transform_seed: int | None,
    ) -> dict:
        data_dict = self._load_frame_data(frame)
        if transform_seed is None:
            data_dict = self.transform(data_dict)
        else:
            with preserve_random_state(transform_seed):
                data_dict = self.transform(data_dict)
        if data_dict["coord"].shape[0] < self.min_points_after_transform:
            raise RuntimeError(
                f"Trajectory frame {frame.frame_name} collapsed below "
                f"min_points_after_transform={self.min_points_after_transform}."
            )
        data_dict["plant"] = trajectory.plant
        data_dict["sequence_id"] = trajectory.sequence_id
        data_dict["frame_name"] = frame.frame_name
        data_dict["frame_index"] = frame.frame_index
        data_dict["frame_position"] = frame_position
        data_dict["num_frames"] = len(trajectory.frames)
        data_dict["is_first_frame"] = frame_position == 0
        data_dict["is_last_frame"] = frame_position == len(trajectory.frames) - 1
        data_dict["reset_state"] = frame_position == 0
        return data_dict

    def prepare_trajectory_data(self, idx):
        trajectory = self.trajectories[idx % len(self.trajectories)]
        max_retries = self.trajectory_retry_count if self.split == "train" else 1
        for _ in range(max_retries):
            transform_seed = None
            if self.split == "train" and self.consistent_trajectory_transform:
                transform_seed = random.randrange(2**31)
            frames: list[dict] = []
            try:
                for frame_position, frame in enumerate(trajectory.frames):
                    frames.append(
                        self._prepare_transformed_frame(
                            trajectory=trajectory,
                            frame=frame,
                            frame_position=frame_position,
                            transform_seed=transform_seed,
                        )
                    )
            except ValueError as exc:
                if "zero-size array to reduction operation" not in str(exc):
                    raise
                continue
            except RuntimeError:
                if self.split != "train":
                    raise
                continue
            return {
                "plant": trajectory.plant,
                "sequence_id": trajectory.sequence_id,
                "num_frames": len(frames),
                "frame_names": [frame.frame_name for frame in trajectory.frames],
                "frames": frames,
                "reset_state_at_start": True,
            }
        raise RuntimeError(
            "Failed to prepare a sufficiently large TomatoWUR trajectory after retries "
            f"for {trajectory.sequence_id}."
        )

    def __getitem__(self, idx):
        return self.prepare_trajectory_data(idx)

    def __len__(self):
        return len(self.trajectories) * self.loop
