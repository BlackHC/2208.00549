__all__ = ['MNISTCNN', 'BayesianMNISTCNN', 'BayesianMNISTCNN_EBM', 'MnistOptimizerFactory', 'MnistModelTrainer', 'BayesianMnistModelTrainer']

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn
import torch.optim
from torch import nn as nn
from torch.nn import functional as F, Module
from torch.utils.data import DataLoader, Dataset

from active_learning import RandomFixedLengthSampler
from black_box_model_training import train, train_simple
from consistent_mc_dropout import (
    GradEmbeddingType,
    BayesianModule,
    ConsistentMCDropout,
    ConsistentMCDropout2d,
    freeze_encoder_context
)

from model_optimizer_factory import ModelOptimizer, ModelOptimizerFactory
from trained_model import ModelTrainer, TrainedModel, TrainedBayesianModel


class MNISTCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop = nn.Dropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc1_drop = nn.Dropout()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input: torch.Tensor):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
        input = input.view(-1, 1024)
        input = F.relu(self.fc1_drop(self.fc1(input)))

        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)

        return input


class BayesianMNISTCNN(BayesianModule):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop = ConsistentMCDropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = ConsistentMCDropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc1_drop = ConsistentMCDropout()
        self.fc2 = nn.Linear(128, num_classes)

    def mc_forward_impl(self, input: torch.Tensor, freeze_encoder: bool):
        with freeze_encoder_context(freeze_encoder):
            input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
            input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
            input = input.view(-1, 1024)
            input = F.relu(self.fc1_drop(self.fc1(input)))

        embedding = input
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)

        return input, embedding


class BayesianMNISTCNN_EBM(BayesianModule):
    """Without Softmax."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop = ConsistentMCDropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = ConsistentMCDropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc1_drop = ConsistentMCDropout()
        self.fc2 = nn.Linear(128, num_classes)

    def mc_forward_impl(self, input: torch.Tensor, freeze_encoder: bool):
        with freeze_encoder_context(freeze_encoder):
            input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
            input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
            input = input.view(-1, 1024)
            input = F.relu(self.fc1_drop(self.fc1(input)))

        embedding = input
        input = self.fc2(input)

        return input, embedding


class MnistOptimizerFactory(ModelOptimizerFactory):
    def create_model_optimizer(self) -> ModelOptimizer:
        model = BayesianMNISTCNN()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4)
        return ModelOptimizer(model=model, optimizer=optimizer)



@dataclass
class MnistModelTrainer:
    device: str

    num_patience_epochs: int = 20
    max_training_epochs: int = 120

    min_samples_per_epoch: int = 1024
    num_training_batch_size: int = 64
    num_evaluation_batch_size: int = 128

    @staticmethod
    def create_model_optimizer() -> ModelOptimizer:
        model = MNISTCNN()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4)
        return ModelOptimizer(model=model, optimizer=optimizer)

    def get_train_dataloader(self, dataset: Dataset):
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.num_training_batch_size,
            sampler=RandomFixedLengthSampler(dataset, self.min_samples_per_epoch),
            drop_last=True,
        )
        return train_loader

    def get_evaluation_dataloader(self, dataset: Dataset):
        evaluation_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.num_evaluation_batch_size, drop_last=False, shuffle=False
        )
        return evaluation_loader

    def get_trained(self, *, train_loader: DataLoader, train_augmentations: Optional[Module],
                    validation_loader: DataLoader, log, wandb_key_path:str, loss=None, validation_loss=None) -> MNISTCNN:
        model_optimizer = self.create_model_optimizer()

        if loss is None:
            loss = torch.nn.NLLLoss()
        if validation_loss is None:
            validation_loss = torch.nn.NLLLoss()

        train_simple(
            model=model_optimizer.model,
            optimizer=model_optimizer.optimizer,
            train_loader=train_loader,
            train_augmentations=train_augmentations,
            validation_loader=validation_loader,
            patience=self.num_patience_epochs,
            max_epochs=self.max_training_epochs,
            loss=loss,
            validation_loss=validation_loss,
            device=self.device,
            training_log=log,
            wandb_key_path=wandb_key_path,
        )

        return model_optimizer.model


@dataclass
class BayesianMnistModelTrainer(ModelTrainer):
    device: str

    num_training_samples: int = 1
    num_validation_samples: int = 20
    num_patience_epochs: int = 20
    max_training_epochs: int = 120

    min_samples_per_epoch: int = 1024
    num_training_batch_size: int = 64
    num_evaluation_batch_size: int = 128

    @staticmethod
    def create_model_optimizer() -> ModelOptimizer:
        model = BayesianMNISTCNN()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4)
        return ModelOptimizer(model=model, optimizer=optimizer)

    def get_train_dataloader(self, dataset: Dataset):
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.num_training_batch_size,
            sampler=RandomFixedLengthSampler(dataset, self.min_samples_per_epoch),
            drop_last=True,
        )
        return train_loader

    def get_evaluation_dataloader(self, dataset: Dataset):
        evaluation_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.num_evaluation_batch_size, drop_last=False, shuffle=False
        )
        return evaluation_loader

    def get_trained(self, *, train_loader: DataLoader, train_augmentations: Optional[Module],
                    validation_loader: DataLoader, log, wandb_key_path:str, loss=None, validation_loss=None) -> TrainedModel:
        model_optimizer = self.create_model_optimizer()

        if loss is None:
            loss = torch.nn.NLLLoss()
        if validation_loss is None:
            validation_loss = torch.nn.NLLLoss()

        train(
            model=model_optimizer.model,
            optimizer=model_optimizer.optimizer,
            training_samples=self.num_training_samples,
            validation_samples=self.num_validation_samples,
            train_loader=train_loader,
            train_augmentations=train_augmentations,
            validation_loader=validation_loader,
            patience=self.num_patience_epochs,
            max_epochs=self.max_training_epochs,
            loss=loss,
            validation_loss=validation_loss,
            device=self.device,
            training_log=log,
            wandb_key_path=wandb_key_path,
        )

        return TrainedBayesianModel(model_optimizer.model)