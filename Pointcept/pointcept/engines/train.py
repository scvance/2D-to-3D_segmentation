"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
import torch
import torch.nn as nn
import torch.utils.data
from functools import partial

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import (
    build_dataset,
    point_collate_fn,
    collate_fn,
    trajectory_collate_fn,
)
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage
from pointcept.utils.registry import Registry


TRAINERS = Registry("trainers")


def move_data_to_cuda(data):
    if isinstance(data, torch.Tensor):
        return data.cuda(non_blocking=True)
    if isinstance(data, dict):
        return {key: move_data_to_cuda(value) for key, value in data.items()}
    if isinstance(data, list):
        return [move_data_to_cuda(value) for value in data]
    if isinstance(data, tuple):
        return tuple(move_data_to_cuda(value) for value in data)
    return data


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                # TODO: optimize to iteration based
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def run_step(self):
        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
        self.comm_info["model_output_dict"] = output_dict
        if not torch.isfinite(loss):
            self.logger.warning(
                "Skipping epoch %s iter %s due to non-finite loss.",
                self.epoch + 1,
                self.comm_info["iter"] + 1,
            )
            if "iter_info" in self.comm_info.keys():
                self.comm_info["iter_info"] += "SkipNonFiniteLoss 1 "
            if self.cfg.empty_cache:
                torch.cuda.empty_cache()
            return

        grad_max_norm = getattr(self.cfg, "grad_max_norm", None)
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            if grad_max_norm is not None:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), grad_max_norm
                )
                if not torch.isfinite(grad_norm).all():
                    self.logger.warning(
                        "Skipping epoch %s iter %s due to non-finite grad norm.",
                        self.epoch + 1,
                        self.comm_info["iter"] + 1,
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    if "iter_info" in self.comm_info.keys():
                        self.comm_info["iter_info"] += "SkipNonFiniteGrad 1 "
                    if self.cfg.empty_cache:
                        torch.cuda.empty_cache()
                    return
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            if grad_max_norm is not None:
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), grad_max_norm
                )
                if not torch.isfinite(grad_norm).all():
                    self.logger.warning(
                        "Skipping epoch %s iter %s due to non-finite grad norm.",
                        self.epoch + 1,
                        self.comm_info["iter"] + 1,
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    if "iter_info" in self.comm_info.keys():
                        self.comm_info["iter_info"] += "SkipNonFiniteGrad 1 "
                    if self.cfg.empty_cache:
                        torch.cuda.empty_cache()
                    return
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        scaler = torch.cuda.amp.GradScaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader


@TRAINERS.register_module("TrajectoryTrainer")
class TrajectoryTrainer(Trainer):
    def _reset_sequence_state(self):
        model = unwrap_model(self.model)
        if hasattr(model, "reset_sequence_state"):
            model.reset_sequence_state()

    def _build_init_fn(self):
        return (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

    def build_train_loader(self):
        if self.cfg.batch_size_per_gpu != 1:
            raise ValueError(
                "TrajectoryTrainer currently requires batch_size_per_gpu=1, got "
                f"{self.cfg.batch_size_per_gpu}."
            )
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=trajectory_collate_fn,
            pin_memory=True,
            worker_init_fn=self._build_init_fn(),
            drop_last=True,
            persistent_workers=self.cfg.num_worker_per_gpu > 0,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            if self.cfg.batch_size_val_per_gpu != 1:
                raise ValueError(
                    "TrajectoryTrainer currently requires batch_size_val_per_gpu=1, got "
                    f"{self.cfg.batch_size_val_per_gpu}."
                )
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=trajectory_collate_fn,
                persistent_workers=self.cfg.num_worker_per_gpu > 0,
            )
        return val_loader

    def run_step(self):
        trajectory_dict = self.comm_info["input_dict"]
        frames = trajectory_dict["frames"]
        if not frames:
            raise RuntimeError(
                f"Trajectory {trajectory_dict.get('sequence_id', '<unknown>')} has no frames."
            )

        self._reset_sequence_state()
        self.optimizer.zero_grad(set_to_none=True)
        loss_terms = []
        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            for frame in frames:
                frame_input = move_data_to_cuda(frame)
                output_dict = self.model(frame_input)
                loss_terms.append(output_dict["loss"])
            loss = torch.stack(loss_terms).mean()
        self.comm_info["model_output_dict"] = {"loss": loss.detach()}

        if not torch.isfinite(loss):
            self.logger.warning(
                "Skipping epoch %s iter %s sequence %s due to non-finite loss.",
                self.epoch + 1,
                self.comm_info["iter"] + 1,
                trajectory_dict.get("sequence_id", "<unknown>"),
            )
            if "iter_info" in self.comm_info.keys():
                self.comm_info["iter_info"] += "SkipNonFiniteLoss 1 "
            self.optimizer.zero_grad(set_to_none=True)
            self._reset_sequence_state()
            if self.cfg.empty_cache:
                torch.cuda.empty_cache()
            return

        grad_max_norm = getattr(self.cfg, "grad_max_norm", None)
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            if grad_max_norm is not None:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), grad_max_norm
                )
                if not torch.isfinite(grad_norm).all():
                    self.logger.warning(
                        "Skipping epoch %s iter %s sequence %s due to non-finite grad norm.",
                        self.epoch + 1,
                        self.comm_info["iter"] + 1,
                        trajectory_dict.get("sequence_id", "<unknown>"),
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    if "iter_info" in self.comm_info.keys():
                        self.comm_info["iter_info"] += "SkipNonFiniteGrad 1 "
                    self._reset_sequence_state()
                    if self.cfg.empty_cache:
                        torch.cuda.empty_cache()
                    return
            self.scaler.step(self.optimizer)
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            if grad_max_norm is not None:
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), grad_max_norm
                )
                if not torch.isfinite(grad_norm).all():
                    self.logger.warning(
                        "Skipping epoch %s iter %s sequence %s due to non-finite grad norm.",
                        self.epoch + 1,
                        self.comm_info["iter"] + 1,
                        trajectory_dict.get("sequence_id", "<unknown>"),
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    if "iter_info" in self.comm_info.keys():
                        self.comm_info["iter_info"] += "SkipNonFiniteGrad 1 "
                    self._reset_sequence_state()
                    if self.cfg.empty_cache:
                        torch.cuda.empty_cache()
                    return
            self.optimizer.step()
            self.scheduler.step()

        self._reset_sequence_state()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
