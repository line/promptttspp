# Copyright 2024 LY Corporation

# LY Corporation licenses this file to you under the Apache License,
# version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at:

#   https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.utils import instantiate
from omegaconf import OmegaConf
from promptttspp.datasets.utils import ShuffleBatchSampler, batch_by_size
from promptttspp.utils import Tracker, seed_everything
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class TTSTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "65535"

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            mp.spawn(self._train, nprocs=num_gpus, args=(self.cfg,))
        else:
            self._train(0, self.cfg)

    def _train(self, rank, cfg):
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            init_process_group(
                "nccl", init_method="env://", world_size=num_gpus, rank=rank
            )

        seed_everything(cfg.train.seed)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        if rank == 0:
            output_dir = Path(cfg.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(cfg, output_dir / "config.yaml")
            ckpt_dir = output_dir / "ckpt"
            log_dir = output_dir / "logs"
            tb_dir = log_dir / "tensorboard"
            [d.mkdir(parents=True, exist_ok=True) for d in [ckpt_dir, log_dir, tb_dir]]
            logger = logging.getLogger(str(log_dir))
            logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s"  # noqa
            )
            h = logging.FileHandler(log_dir / "train.log")
            h.setLevel(logging.DEBUG)
            h.setFormatter(formatter)
            logger.addHandler(h)

            writer = SummaryWriter(log_dir=tb_dir)

        model = instantiate(cfg.model).to(device)
        if rank == 0:
            logger.info(
                f"model parameter : {sum(p.numel() for p in model.parameters())}"
            )
        optimizer = instantiate(cfg.optimizer, params=model.parameters())
        if "lr_scheduler" in cfg.train:
            lr_scheduler = instantiate(cfg.train.lr_scheduler, optimizer=optimizer)
        else:
            lr_scheduler = None
        per_epoch_scheduler = cfg.train.get("per_epoch_scheduler", True)
        scaler = GradScaler(enabled=cfg.train.fp16)

        start_epoch = 1
        if "pretrained" in cfg:
            print("Using pretrained")
            try:
                ckpt = torch.load(cfg.pretrained)
                missing_keys = model.load_state_dict(ckpt["model"], strict=False)
                print(missing_keys)
                params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"model parameter : {params}")
            except:  # noqa
                print("Failed loading pretrained.")
        if cfg.ckpt_path is not None:
            try:
                ckpt = torch.load(cfg.ckpt_path)
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                if "lr_scheduler" in ckpt:
                    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
                start_epoch = ckpt["epoch"] + 1
            except:  # noqa
                print("Failed loading checkpoint.")

        if num_gpus > 1:
            model = DDP(model, device_ids=[rank]).to(device)

        to_mel = instantiate(cfg.transforms)
        collator = instantiate(cfg.dataset.collator)
        train_ds = instantiate(cfg.dataset.train, to_mel=to_mel)
        if "dynamic_batch" in cfg.dataset and cfg.dataset.dynamic_batch:
            if dist.is_initialized():
                required_batch_size_multiple = dist.get_world_size()
            else:
                required_batch_size_multiple = 1
            indices = train_ds.ordered_indices()
            max_tokens = (
                cfg.dataset.max_tokens if "max_tokens" in cfg.dataset else 30000
            )
            batches = batch_by_size(
                indices,
                train_ds.num_tokens,
                max_tokens=max_tokens,
                required_batch_size_multiple=required_batch_size_multiple,
            )

            if dist.is_initialized():
                num_replicas = dist.get_world_size()
                batches = [
                    x[rank::num_replicas] for x in batches if len(x) % num_replicas == 0
                ]
            batch_sampler = ShuffleBatchSampler(batches, drop_last=True, shuffle=True)
            train_dl = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                pin_memory=True,
                collate_fn=collator,
                num_workers=cfg.train.num_workers,
            )
        else:
            train_sampler = (
                DistributedSampler(train_ds, shuffle=True, drop_last=True)
                if num_gpus > 1
                else None
            )
            if dist.is_initialized():
                batch_size = cfg.train.batch_size * dist.get_world_size()
            else:
                batch_size = cfg.train.batch_size
            train_dl = DataLoader(
                train_ds,
                batch_size=batch_size,
                sampler=train_sampler,
                shuffle=train_sampler is None,
                pin_memory=True,
                drop_last=True,
                collate_fn=collator,
                num_workers=cfg.train.num_workers,
            )

        if rank == 0:
            valid_ds = instantiate(cfg.dataset.valid, to_mel=to_mel)
            valid_dl = DataLoader(
                valid_ds,
                batch_size=cfg.train.batch_size,
                shuffle=False,
                pin_memory=True,
                collate_fn=collator,
                num_workers=cfg.train.num_workers,
            )
        global_step = (start_epoch - 1) * len(train_dl) + 1

        def to_device(*args):
            # TODO: handle args for general settings
            return [
                a.to(device, non_blocking=True) if isinstance(a, torch.Tensor) else a
                for a in args
            ][2:]

        if rank == 0:
            mode = "a" if cfg.ckpt_path is not None else "w"
            tracker = Tracker(log_dir / "loss.csv", mode=mode)
        for epoch in range(start_epoch, cfg.train.num_epochs + 1):
            if num_gpus > 1:
                train_sampler.set_epoch(epoch)
            bar = tqdm(
                train_dl, total=len(train_dl), desc=f"Epoch: {epoch}", disable=rank != 0
            )
            model.train()
            for batch in bar:
                batch = to_device(*batch)
                with autocast(device_type="cuda", enabled=cfg.train.fp16):
                    loss_dict = model(batch)
                    loss = loss_dict["loss"]
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if rank == 0:
                    record = {f"train/{k}": v.item() for k, v in loss_dict.items()}
                    tracker.update(**record)
                    s = ", ".join(
                        f'{k.split("/")[1]}: {v.mean():.5f}' for k, v in tracker.items()
                    )
                    bar.set_postfix_str(s)
                    global_step += 1
                if not per_epoch_scheduler and lr_scheduler is not None:
                    lr_scheduler.step()

                # break
            if rank == 0:
                for k, v in record.items():
                    writer.add_scalar(k, v, global_step)
                logger.info(f"Train: {epoch}, {s}")

            if rank == 0:
                model.eval()
                for batch in tqdm(valid_dl, total=len(valid_dl), leave=False):
                    batch = to_device(*batch)
                    with torch.no_grad():
                        loss_dict = model(batch)
                    record = {f"valid/{k}": v.item() for k, v in loss_dict.items()}
                    tracker.update(**record)
                    for k, v in record.items():
                        writer.add_scalar(k, v, global_step)
                s = ", ".join(
                    f'{k.split("/")[1]}: {v.mean():.5f}'
                    for k, v in tracker.items()
                    if k.startswith("valid")
                )
                logger.info(f"Valid: {epoch}, {s}")
                save_obj = {
                    "epoch": epoch,
                    "model": (model.module if num_gpus > 1 else model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if lr_scheduler is not None:
                    save_obj["lr_scheduler"] = lr_scheduler.state_dict()
                torch.save(save_obj, ckpt_dir / "last.ckpt")
                if epoch % cfg.train.save_interval == 0:
                    torch.save(save_obj, ckpt_dir / f"epoch-{epoch}.ckpt")
                tracker.write(epoch, clear=True)

            if per_epoch_scheduler and lr_scheduler is not None:
                lr_scheduler.step()
