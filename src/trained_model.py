__all__ = ['TrainedModel', 'TrainedBayesianModel', 'TrainedBayesianEnsemble', 'ModelTrainer',
           'BayesianEnsembleModelTrainer']

from dataclasses import dataclass
from typing import List, Optional

import torch.nn
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from consistent_mc_dropout import BayesianModule, GradEmbeddingType


@dataclass
class TrainedModel:
    """Evaluate a trained model."""

    def get_log_probs_N_K_C_labels_N(
        self, loader: DataLoader, num_samples: int, device: object, storage_device: object
    ):
        raise NotImplementedError()

    def get_log_probs_N_K_C(self, loader: DataLoader, num_samples: int, device: object, storage_device: object):
        log_probs_N_K_C, labels = self.get_log_probs_N_K_C_labels_N(loader, num_samples, device, storage_device)
        return log_probs_N_K_C

    def get_grad_embeddings(
        self,
        loader: DataLoader,
        num_samples: int,
        loss,
        grad_embedding_type: GradEmbeddingType,
        model_labels: bool,
        device: object,
        storage_device: object,
    ):
        raise NotImplementedError()


@dataclass
class TrainedBayesianModel(TrainedModel):
    model: BayesianModule

    def get_log_probs_N_K_C_labels_N(
        self, loader: DataLoader, num_samples: int, device: object, storage_device: object
    ):
        log_probs_N_K_C, labels_B = self.model.get_predictions_labels(
            num_samples=num_samples, loader=loader, device=device, storage_device=storage_device
        )

        # NOTE: this wastes memory bandwidth, but is needed for ensembles where more than one model might not fit
        # into memory.
        self.model.to("cpu")

        return log_probs_N_K_C, labels_B

    def get_grad_embeddings(
        self,
        loader: DataLoader,
        num_samples: int,
        loss,
        grad_embedding_type: GradEmbeddingType,
        model_labels: bool,
        device: object,
        storage_device: object,
    ):
        grad_embeddings_N_K_E = self.model.get_grad_embeddings(
            num_samples=num_samples,
            loader=loader,
            loss=loss,
            grad_embedding_type=grad_embedding_type,
            model_labels=model_labels,
            device=device,
            storage_device=storage_device,
        )
        return grad_embeddings_N_K_E

    def get_embeddings(
        self,
        loader: DataLoader,
        num_samples: int,
        device: object,
        storage_device: object,
    ):
        embeddings_N_K_E = self.model.get_grad_embeddings(
            num_samples=num_samples,
            loader=loader,
            device=device,
            storage_device=storage_device,
        )
        return embeddings_N_K_E

@dataclass
class TrainedBayesianEnsemble(TrainedModel):
    models: List[TrainedModel]

    def get_log_probs_N_K_C_labels_N(
        self, loader: DataLoader, num_samples: int, device: object, storage_device: object
    ):
        ensemble_size = len(self.models)
        member_num_samples = (num_samples + ensemble_size - 1) // ensemble_size

        ensemble_log_probs_N_K_C = []
        ensemble_labels_B = None

        for model in self.models:
            log_probs_N_K_C, labels_B = model.get_log_probs_N_K_C_labels_N(
                loader=loader, num_samples=member_num_samples, device=device, storage_device=storage_device
            )

            ensemble_log_probs_N_K_C += [log_probs_N_K_C]
            if ensemble_labels_B is not None:
                assert torch.all(ensemble_labels_B == labels_B)
            else:
                ensemble_labels_B = labels_B

        ensemble_log_probs_N_K_C = torch.cat(ensemble_log_probs_N_K_C, dim=1)
        return ensemble_log_probs_N_K_C, ensemble_labels_B


    def get_grad_embeddings(
        self,
        loader: DataLoader,
        num_samples: int,
        loss,
        grad_embedding_type: GradEmbeddingType,
        model_labels: bool,
        device: object,
        storage_device: object,
    ):
        ensemble_size = len(self.models)
        member_num_samples = (num_samples + ensemble_size - 1) // ensemble_size

        ensemble_grad_embeddings_N_K_E = []

        for model in self.models:
            grad_embeddings_N_K_E = model.get_grad_embeddings(
                    num_samples=member_num_samples,
                    loader=loader,
                    loss=loss,
                    grad_embedding_type=grad_embedding_type,
                    model_labels=model_labels,
                    device=device,
                    storage_device=storage_device,
                )

            ensemble_grad_embeddings_N_K_E += [grad_embeddings_N_K_E]

        ensemble_grad_embeddings_N_K_E = torch.cat(ensemble_grad_embeddings_N_K_E, dim=1)
        return ensemble_grad_embeddings_N_K_E


    def get_embeddings(
        self,
        loader: DataLoader,
        num_samples: int,
        device: object,
        storage_device: object,
    ):
        ensemble_size = len(self.models)
        member_num_samples = (num_samples + ensemble_size - 1) // ensemble_size

        ensemble_embeddings_N_K_E = []

        for model in self.models:
            embeddings_N_K_E = model.get_embeddings(
                    num_samples=member_num_samples,
                    loader=loader,
                    device=device,
                    storage_device=storage_device,
                )

            ensemble_embeddings_N_K_E += [embeddings_N_K_E]

        ensemble_embeddings_N_K_E = torch.cat(ensemble_embeddings_N_K_E, dim=1)
        return ensemble_embeddings_N_K_E


class ModelTrainer:
    train_batch_size: int
    evaluation_batch_size: int

    def get_train_dataloader(self, dataset: Dataset):
        raise NotImplementedError

    # test|validation|evaluation
    def get_evaluation_dataloader(self, dataset: Dataset):
        raise NotImplementedError

    def get_trained(
        self,
        *,
        train_loader: DataLoader,
        train_augmentations: Optional[Module],
        validation_loader: DataLoader,
        log,
        wandb_key_path: str,
        loss=None,
        validation_loss=None
    ) -> TrainedModel:
        raise NotImplementedError

    def get_distilled(
        self,
        *,
        prediction_loader: DataLoader,
        train_augmentations: Optional[Module],
        validation_loader: DataLoader,
        log,
        wandb_key_path:str
    ) -> TrainedModel:
        loss = torch.nn.KLDivLoss(log_target=True, reduction="batchmean")
        validation_loss = torch.nn.NLLLoss()
        return self.get_trained(
            train_loader=prediction_loader,
            train_augmentations=train_augmentations,
            validation_loader=validation_loader,
            loss=loss,
            validation_loss=validation_loss,
            log=log,
            wandb_key_path=wandb_key_path,
        )


@dataclass
class BayesianEnsembleModelTrainer(ModelTrainer):
    model_trainer: ModelTrainer
    ensemble_size: int

    def get_train_dataloader(self, dataset: Dataset):
        return self.model_trainer.get_train_dataloader(dataset)

    # test|validation|evaluation
    def get_evaluation_dataloader(self, dataset: Dataset):
        return self.model_trainer.get_evaluation_dataloader(dataset)

    def get_trained(
        self,
        *,
        train_loader: DataLoader,
        train_augmentations: Optional[Module],
        validation_loader: DataLoader,
        log,
        wandb_key_path: str,
        loss=None,
        validation_loss=None
    ) -> TrainedBayesianEnsemble:
        models = []

        log["ensemble"] = []
        for i in range(self.ensemble_size):
            log["ensemble"].append({})
            model = self.model_trainer.get_trained(
                train_loader=train_loader,
                train_augmentations=train_augmentations,
                validation_loader=validation_loader,
                log=log["ensemble"][-1],
                wandb_key_path=wandb_key_path + f"/{i}/",
                loss=loss,
                validation_loss=validation_loss,
            )
            models += [model]

        return TrainedBayesianEnsemble(models)


    def get_distilled(
        self,
        *,
        prediction_loader: DataLoader,
        train_augmentations: Optional[Module],
        validation_loader: DataLoader,
        log,
        wandb_key_path:str,
    ) -> TrainedModel:
        models = []

        log["ensemble"] = []
        for i in range(self.ensemble_size):
            log["ensemble"].append({})
            model = self.model_trainer.get_distilled(
                prediction_loader=prediction_loader,
                train_augmentations=train_augmentations,
                validation_loader=validation_loader,
                log=log["ensemble"][-1],
                wandb_key_path=wandb_key_path + f"/{i}/",
            )
            models += [model]

        return TrainedBayesianEnsemble(models)
