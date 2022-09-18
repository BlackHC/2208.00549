__all__ = ['TrainEvalModel', 'TrainSelfDistillationEvalModel', 'TrainRandomLabelEvalModel', 'TrainExplicitEvalModel']


from dataclasses import dataclass
from typing import Optional, Type

import torch
import torch.nn
import torch.utils.data
from torch import nn

from consistent_mc_dropout import get_log_mean_probs
from dataset_operations import (
    RandomLabelsDataset,
    ReplaceTargetsDataset,
)
from trained_model import ModelTrainer, TrainedModel


class TrainEvalModel:
    def __call__(
        self,
        *,
        model_trainer: ModelTrainer,
        training_dataset: torch.utils.data.Dataset,
        train_augmentations: nn.Module,
        eval_dataset: torch.utils.data.Dataset,
        validation_loader: torch.utils.data.DataLoader,
        trained_model: TrainedModel,
        device: Optional,
        storage_device: Optional,
        training_log,
        wandb_key_path: str
    ) -> TrainedModel:
        raise NotImplementedError()


@dataclass
class TrainSelfDistillationEvalModel(TrainEvalModel):
    num_pool_samples: int

    def __call__(
        self,
        *,
        model_trainer: ModelTrainer,
        training_dataset: torch.utils.data.Dataset,
        train_augmentations: nn.Module,
        eval_dataset: torch.utils.data.Dataset,
        validation_loader: torch.utils.data.DataLoader,
        trained_model: TrainedModel,
        device: Optional,
        storage_device: Optional,
        training_log,
        wandb_key_path: str
    ):
        train_eval_dataset = torch.utils.data.ConcatDataset([training_dataset, eval_dataset])
        train_eval_loader = model_trainer.get_evaluation_dataloader(train_eval_dataset)

        eval_log_probs_N_C = get_log_mean_probs(
            trained_model.get_log_probs_N_K_C(
                train_eval_loader, num_samples=self.num_pool_samples, device=device, storage_device=storage_device
            )
        )

        eval_self_distillation_dataset = ReplaceTargetsDataset(dataset=train_eval_dataset, targets=eval_log_probs_N_C)

        train_eval_self_distillation_loader = model_trainer.get_train_dataloader(eval_self_distillation_dataset)

        trained_model = model_trainer.get_distilled(
            prediction_loader=train_eval_self_distillation_loader,
            train_augmentations=train_augmentations,
            validation_loader=validation_loader,
            log=training_log,
            wandb_key_path=wandb_key_path,
        )

        return trained_model


@dataclass
class TrainRandomLabelEvalModel(TrainEvalModel):
    def __call__(
        self,
        *,
        model_trainer: ModelTrainer,
        training_dataset: torch.utils.data.Dataset,
        train_augmentations: nn.Module,
        eval_dataset: torch.utils.data.Dataset,
        validation_loader: torch.utils.data.DataLoader,
        trained_model: TrainedModel,
        device: Optional,
        storage_device: Optional,
        training_log,
        wandb_key_path: str
    ):
        # TODO: support one_hot!
        # TODO: different seed needed!
        train_eval_dataset = torch.utils.data.ConcatDataset(
            [training_dataset, RandomLabelsDataset(eval_dataset, seed=0, device=storage_device)]
        )
        train_eval_loader = model_trainer.get_train_dataloader(train_eval_dataset)

        trained_model = model_trainer.get_trained(
            train_loader=train_eval_loader,
            train_augmentations=train_augmentations,
            validation_loader=validation_loader,
            log=training_log,
            wandb_key_path=wandb_key_path,
        )

        return trained_model


@dataclass
class TrainExplicitEvalModel(TrainEvalModel):
    cache_explicit_eval_model: bool = False
    _fully_trained_model: TrainedModel = None

    def __call__(
        self,
        *,
        model_trainer: ModelTrainer,
        training_dataset: torch.utils.data.Dataset,
        train_augmentations: nn.Module,
        eval_dataset: torch.utils.data.Dataset,
        validation_loader: torch.utils.data.DataLoader,
        trained_model: TrainedModel,
        device: Optional,
        storage_device: Optional,
        training_log,
        wandb_key_path: str
    ):
                        # TODO: support one_hot!? For this we need to change the eval_dataset to also have one_hot applied in
                        #  ExperimentData?
        if not self._fully_trained_model:
            train_eval_dataset = torch.utils.data.ConcatDataset([training_dataset, eval_dataset])
            train_eval_loader = model_trainer.get_train_dataloader(train_eval_dataset)

            trained_model = model_trainer.get_trained(
                train_loader=train_eval_loader,
                train_augmentations=train_augmentations,
                validation_loader=validation_loader,
                log=training_log,
                wandb_key_path=wandb_key_path,
            )
            if self.cache_explicit_eval_model:
                self._fully_trained_model = trained_model
        else:
            print("Using cached fully trained model!")
            trained_model = self._fully_trained_model

        return trained_model