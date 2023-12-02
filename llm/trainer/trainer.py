import random
from pathlib import Path
from random import shuffle

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from llm.base import BaseTrainer
from llm.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True
    ):
        super().__init__(model, criterion, optimizer, config, device)

        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss", "grad norm", writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", writer=self.writer
        )
        self.fine_tune = config["trainer"].get("finetune", False)
        self.grad_accum_iters = config["trainer"].get("grad_accum_iters", 1)
        self.eval_start_iter = config["trainer"].get("eval_start_iter", 0)
        self.scheduler_config = config["trainer"].get("scheduler", None)
        if self.scheduler_config is not None:
            self.scheduler_config["requires_loss"] = self.scheduler_config.get("requires_loss", False)
            self.scheduler_config["epoch_based"] = self.scheduler_config.get("epoch_based", False)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
                checkpoint["config"]["optimizer"] != self.config["optimizer"] or
                (self.lr_scheduler is not None and checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"])
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in config file is different "
                "from that of checkpoint. Optimizer parameters not being resumed."
            )
        elif not self.fine_tune:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["sentence"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )
    
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            stop = False
            try:
                batch = self.process_batch(
                    batch,
                    batch_idx,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    raise e
                    continue
                else:
                    raise e
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                if self.lr_scheduler is not None:
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )
                self._log_predictions(**batch, is_train=True)
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                stop = True
                break
            if stop:
                break
        log = last_train_metrics
        
        if epoch * self.len_epoch >= self.eval_start_iter:
            for part, dataloader in self.evaluation_dataloaders.items():
                val_log = self._evaluation_epoch(epoch, part, dataloader)
                log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        if self.lr_scheduler is not None and self.scheduler_config is not None and self.scheduler_config["epoch_based"]:
            if self.scheduler_config["requires_loss"]:
                if "val_loss" in log:
                    self.lr_scheduler.step(log["val_loss"])
            else:
                self.lr_scheduler.step()

        return log

    def process_batch(self, batch, batch_idx, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        outputs = self.model(**batch)
        batch.update(outputs)
        batch["loss"] = self.criterion(**batch) / self.grad_accum_iters
        if is_train:
            batch["loss"].backward()
            if (batch_idx + 1) % self.grad_accum_iters == 0 or (batch_idx + 1) == self.len_epoch:
                self._clip_grad_norm()
                self.optimizer.step()
                if self.lr_scheduler is not None and self.scheduler_config is not None and not self.scheduler_config["epoch_based"]:
                    self.lr_scheduler.step()
                self.train_metrics.update("grad norm", self.get_grad_norm())
                self.optimizer.zero_grad()

        metrics.update("loss", batch["loss"].item())
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    batch_idx,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(**batch, is_train=False)

        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            examples_to_log=10,
            *args,
            **kwargs,
    ):
        if self.writer is None:
            return

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]

        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))