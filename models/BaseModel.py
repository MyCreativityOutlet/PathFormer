"""
PathFormer
Copyright (C) 2026  Liam Brydon
Contact at: liam.brydon@auckland.ac.nz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import lightning as L
import abc
from argparse import Namespace
from torch import nn, Tensor
from torch.utils.data import DataLoader
from utils.utils import save_train_metrics


class BaseModel(L.LightningModule, abc.ABC):
    def __init__(self, args: Namespace):
        self.args = args
        self.pre_trained_weights = None
        super().__init__()

    def _init_params(self):
        """
        Apply Xavier uniform initialisation of learnable weights
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @abc.abstractmethod
    def encode_data(self, smiles) -> tuple[DataLoader, DataLoader, DataLoader]:
        raise NotImplementedError

    @abc.abstractmethod
    def encode_data_kfold(self, smiles) -> list:
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, src, trg) -> Tensor:
        raise NotImplementedError

    def on_train_start(self) -> None:
        if self.trainer.early_stopping_callback.mode == "max":
            self.trainer.early_stopping_callback.best_score *= 1e-10
        else:
            self.trainer.early_stopping_callback.best_score *= 1e10


    @abc.abstractmethod
    def x_step(self, batch: tuple[Tensor, Tensor], step_type: str, save_output: bool = False) -> Tensor:
        raise NotImplementedError

    def training_step(self, batch: tuple[Tensor, Tensor], batch_id: int) -> Tensor:
        return self.x_step(batch, "train")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_id: int) -> Tensor:
        return self.x_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        steps_per_epoch = {
            "train": self.trainer.num_training_batches if self.trainer.num_training_batches < float('inf') else 1,
            "val": self.trainer.num_val_batches[0],
            "test": self.trainer.num_val_batches[0]}
        save_train_metrics(self.metric_history, self.save_path, steps_per_epoch)

    @abc.abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError

    @abc.abstractmethod
    def reset_model(self):
        raise NotImplementedError
