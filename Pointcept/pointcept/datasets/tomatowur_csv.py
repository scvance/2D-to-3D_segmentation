"""
ScanNet20 / ScanNet200 / ScanNet Data Efficient Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import csv
import math
import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .transform import Compose, TRANSFORMS

import json
from pathlib import Path
import pandas as pd

# from .preprocessing.scannet.meta_data.scannet200_constants import (
#     VALID_CLASS_IDS_20,
#     VALID_CLASS_IDS_200,
# )


# ScanNet Benchmark constants
# VALID_CLASS_IDS_20 = (
#     1,
#     2,
#     3,
#     4,
# )

# CLASS_LABELS_20 = (
#     "leaves",
#     "main_stem",
#     "pole",
#     "side_stem",
# )

import os

@DATASETS.register_module()
class TomatoWURCSV(Dataset):
    # class2id = np.array(VALID_CLASS_IDS_20)

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
    ):
        super(TomatoWURCSV, self).__init__()
        self.data_root = data_root
        self.split = split
        self.filter_grid_size = self._infer_grid_size(transform)
        self.transform = Compose(transform)
        self.cache = cache
        self.min_rows = min_rows if not test_mode else 1
        self.min_voxels = min_voxels if not test_mode else 1
        self.min_points_after_transform = min_points_after_transform if not test_mode else 1
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        if lr_file:
            with open(lr_file, "r") as f:
                data = json.load(f)
            print("USING NEW APPROACH")
            pairs = [
                (
                    str(Path(lr_file).parent / item["file_name"]),
                    str(Path(lr_file).parent / item["sem_seg_file_name"]),
                )
                for item in data
            ]
            valid_pairs = []
            invalid_pairs = []
            for pc_path, seg_path in pairs:
                if self._sample_is_valid(pc_path, seg_path):
                    valid_pairs.append((pc_path, seg_path))
                else:
                    invalid_pairs.append((pc_path, seg_path))
            self.pc_list = [pc_path for pc_path, _ in valid_pairs]
            self.data_list = [seg_path for _, seg_path in valid_pairs]
        else:
            self.data_list = self.get_data_list()
        self.la = torch.load(la_file) if la_file else None
        self.ignore_index = ignore_index
        logger = get_root_logger()
        if lr_file and invalid_pairs:
            preview = ", ".join(Path(seg_path).name for _, seg_path in invalid_pairs[:5])
            logger.warning(
                "Filtered %s empty or too-small TomatoWUR CSV samples from %s with min_rows=%s, min_voxels=%s. Examples: %s",
                len(invalid_pairs),
                lr_file,
                self.min_rows,
                self.min_voxels,
                preview,
            )
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )
        if len(self.data_list) == 0:
            raise RuntimeError(
                f"No valid TomatoWUR samples found in {lr_file or self.data_root}"
            )

    def _sample_is_valid(self, pc_path, seg_path):
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
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
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

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        semantic_column = "semantic"

        if not self.cache:
            # data = torch.load(data_path)
            fields = ["semantic"]
            data = pd.read_csv(data_path, usecols=fields)
            fields = ["x", "y", "z", "blue", "green", "red", "nx", "ny", "nz"]
            data_pc = pd.read_csv(self.pc_list[idx % len(self.pc_list)], usecols=fields)
            # Stack data and data_pc
            data = pd.concat([data, data_pc], axis=1)
        else:
            data_name = data_path.replace(os.path.dirname(self.data_root), "").split(
                "."
            )[0]
            cache_name = "pointcept" + data_name.replace(os.path.sep, "-")
            data = shared_dict(cache_name)
        coord = data[["x", "y", "z"]].values
        color = data[["red", "green", "blue"]].values ## Based on pre-trained scannet
        normal = data[["nx", "ny", "nz"]].values
        scene_id = Path(data_path).stem
        if semantic_column in data.columns:
            segment =data[semantic_column].astype(int).values.reshape([-1]) -1 
            # print(np.unique(segment), data["scene_id"])
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            mask = np.ones_like(segment).astype(np.bool)
            mask[sampled_index] = False
            segment[mask] = self.ignore_index
            data_dict["segment"] = segment
            data_dict["sampled_index"] = sampled_index
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def prepare_train_data(self, idx):
        # Retry a few times in case a transform collapses a sample unexpectedly.
        for retry in range(min(len(self.data_list), 10)):
            data_dict = self.get_data((idx + retry) % len(self.data_list))
            if data_dict["coord"].shape[0] < self.min_points_after_transform:
                continue
            try:
                data_dict = self.transform(data_dict)
            except ValueError as exc:
                if "zero-size array to reduction operation" in str(exc):
                    continue
                raise
            if data_dict["coord"].shape[0] >= self.min_points_after_transform:
                return data_dict
        raise RuntimeError(
            "Failed to prepare a sufficiently large TomatoWUR training sample after retries."
        )

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        segment = data_dict.pop("segment")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            for data_part in data_part_list:
                if self.test_crop:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                input_dict_list += data_part

        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        data_dict = dict(
            fragment_list=input_dict_list, segment=segment, name=self.get_data_name(idx)
        )
        return data_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


# @DATASETS.register_module()
# class ScanNet200Dataset(ScanNetDataset):
#     class2id = np.array(VALID_CLASS_IDS_200)

#     def get_data(self, idx):
#         data = torch.load(self.data_list[idx % len(self.data_list)])
#         coord = data["coord"]
#         color = data["color"]
#         normal = data["normal"]
#         scene_id = data["scene_id"]
#         if "semantic0" in data.keys():
#             segment = data["semantic0"].reshape([-1])
#         else:
#             segment = np.ones(coord.shape[0]) * -1
#         if "instance_gt" in data.keys():
#             instance = data["instance_gt"].reshape([-1])
#         else:
#             instance = np.ones(coord.shape[0]) * -1
#         data_dict = dict(
#             coord=coord,
#             normal=normal,
#             color=color,
#             segment=segment,
#             instance=instance,
#             scene_id=scene_id,
#         )
#         if self.la:
#             sampled_index = self.la[self.get_data_name(idx)]
#             segment[sampled_index] = self.ignore_index
#             data_dict["segment"] = segment
#             data_dict["sampled_index"] = sampled_index
#         return data_dict
