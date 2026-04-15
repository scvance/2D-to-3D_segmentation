import torch.nn as nn

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        seg_logits = self.seg_head(point.feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class TrajectorySegmentorV2(DefaultSegmentorV2):
    TRAJECTORY_META_KEYS = {
        "plant",
        "sequence_id",
        "frame_name",
        "frame_index",
        "frame_position",
        "num_frames",
        "is_first_frame",
        "is_last_frame",
        "reset_state",
    }

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__(
            num_classes=num_classes,
            backbone_out_channels=backbone_out_channels,
            backbone=backbone,
            criteria=criteria,
        )
        self.hidden_state = None
        self.current_sequence_id = None

    def reset_sequence_state(self):
        self.hidden_state = None
        self.current_sequence_id = None

    def _frame_input_dict(self, input_dict):
        return {
            key: value
            for key, value in input_dict.items()
            if key not in self.TRAJECTORY_META_KEYS
        }

    def _update_sequence_state(self, input_dict):
        sequence_id = input_dict.get("sequence_id")
        reset_state = bool(input_dict.get("reset_state", False))
        if reset_state or (
            sequence_id is not None and sequence_id != self.current_sequence_id
        ):
            self.hidden_state = None
            self.current_sequence_id = sequence_id
        elif sequence_id is not None and self.current_sequence_id is None:
            self.current_sequence_id = sequence_id

    def forward(self, input_dict):
        self._update_sequence_state(input_dict)
        frame_input = self._frame_input_dict(input_dict)
        point = Point(frame_input)
        point = self.backbone(point)
        seg_logits = self.seg_head(point.feat)
        if self.training:
            output_dict = dict(loss=self.criteria(seg_logits, frame_input["segment"]))
        elif "segment" in frame_input.keys():
            loss = self.criteria(seg_logits, frame_input["segment"])
            output_dict = dict(loss=loss, seg_logits=seg_logits)
        else:
            output_dict = dict(seg_logits=seg_logits)
        if bool(input_dict.get("is_last_frame", False)):
            self.reset_sequence_state()
        return output_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        feat = self.backbone(input_dict)
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
