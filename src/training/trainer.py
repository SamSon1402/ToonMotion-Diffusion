"""
Training Loop
==============
Data loading, forward/backward, EMA, checkpointing, logging, evaluation.
"""

import torch
from torch.utils.data import DataLoader
import os
import time
import logging
from typing import Optional

from ..models.toonmotion import ToonMotionDiffusion, ToonMotionConfig
from ..data.dataset import ToonMotionDataset
from .ema import EMA
from .scheduler import WarmupCosineScheduler

logger = logging.getLogger("ToonMotion")


class Trainer:
    def __init__(self, model, config, device="cpu", output_dir="checkpoints"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        self.ema = EMA(model, decay=config.ema_decay)
        total_steps = config.num_epochs * 100
        self.scheduler = WarmupCosineScheduler(self.optimizer, warmup_steps=1000, total_steps=total_steps)
        self.global_step = 0
        self.best_loss = float("inf")

    def train(self, dataset, val_dataset=None):
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=0, drop_last=True)
        logger.info(f"Training: {self.config.num_epochs} epochs, {len(dataloader)} batches/epoch")

        self.model.train()
        for epoch in range(self.config.num_epochs):
            epoch_loss = self._train_epoch(dataloader, epoch)
            if (epoch + 1) % 50 == 0:
                self._save_checkpoint(epoch, epoch_loss)
            if val_dataset and (epoch + 1) % 25 == 0:
                val_loss = self._validate(val_dataset)
                logger.info(f"  Val loss: {val_loss:.4f}")

        self._save_checkpoint(self.config.num_epochs - 1, epoch_loss, is_final=True)

    def _train_epoch(self, dataloader, epoch):
        total_loss, n = 0, 0
        t0 = time.time()

        for batch in dataloader:
            motion = batch["motion"].to(self.device)
            char_ids = batch["character_id"].to(self.device)
            losses = self.model.compute_loss(motion, batch["text"], char_ids)

            self.optimizer.zero_grad()
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.ema.update(self.model)

            total_loss += losses["total_loss"].item()
            n += 1
            self.global_step += 1

        avg = total_loss / max(n, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} | Loss: {avg:.4f} | Time: {time.time()-t0:.1f}s")
        return avg

    @torch.no_grad()
    def _validate(self, dataset):
        self.model.eval()
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        total, n = 0, 0
        for batch in loader:
            losses = self.model.compute_loss(
                batch["motion"].to(self.device), batch["text"], batch["character_id"].to(self.device)
            )
            total += losses["total_loss"].item()
            n += 1
        self.model.train()
        return total / max(n, 1)

    def _save_checkpoint(self, epoch, loss, is_final=False):
        name = "final.pt" if is_final else f"epoch_{epoch+1}.pt"
        path = os.path.join(self.output_dir, name)
        torch.save({
            "epoch": epoch, "model_state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss, "global_step": self.global_step, "config": self.config,
        }, path)
        logger.info(f"Saved: {path}")
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(torch.load(path), os.path.join(self.output_dir, "best.pt"))

    @staticmethod
    def load_checkpoint(path, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        config = ckpt["config"]
        model = ToonMotionDiffusion(config, device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model, config