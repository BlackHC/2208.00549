__all__ = ['conv3x3', 'conv1x1', 'BasicBlock', 'Bottleneck', 'BayesianResNet', 'bayesian_resnet18', 'bayesian_resnet34',
           'model_urls', 'Cifar10BayesianResnetFactory', 'Cifar10DeterministicResnetFactory',
           'Cifar10ModelWorkshopPaperTrainer', 'Cifar10ModelTrainer']

from dataclasses import dataclass
from typing import Optional, Callable, Type, Union, List, Any

import torch
import torch.optim
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from active_learning import RandomFixedLengthSampler
from black_box_model_training import (
    train_with_cosine_annealing,
    train_with_schedule,
)
from consistent_mc_dropout import (
    BayesianModule,
    ConsistentMCDropout,
    freeze_encoder_context,
)
from model_optimizer_factory import (
    ModelOptimizer,
    ModelOptimizerFactory,
)

from torch import Tensor
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F

from trained_model import (
    ModelTrainer,
    TrainedBayesianModel,
    TrainedModel,
)

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BayesianResNet(BayesianModule):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        dropout_head: bool,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # This gets reset when cifar_mod=True below
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # This gets reset when cifar_mod=True
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if not dropout_head:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.fc = nn.Sequential(
                ConsistentMCDropout(),
                nn.Linear(512 * block.expansion, 256 * block.expansion),
                nn.ReLU(),
                ConsistentMCDropout(),
                nn.Linear(256 * block.expansion, 256 * block.expansion),
                nn.Linear(256, num_classes),
            )
        # self.fc = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(512 * block.expansion, 256 * block.expansion),
        #     ConsistentMCDropout(),
        #     nn.Linear(256 * block.expansion, 128 * block.expansion),
        #     ConsistentMCDropout(),
        #     nn.Linear(128 * block.expansion, 128 * block.expansion),
        #     nn.ReLU(),
        #     nn.Linear(128 * block.expansion, num_classes),
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def deterministic_forward_impl(self, x: Tensor, freeze_encoder: bool):
        with freeze_encoder_context(freeze_encoder):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.reshape(x.size(0), -1)

        return x

    def mc_forward_impl(self, x: Tensor, freeze_encoder: bool):
        embedding = x

        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)

        return x, embedding


def _bayesian_resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    cifar_mod: bool,
    dropout_head: bool,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> BayesianResNet:
    model = BayesianResNet(block=block, layers=layers, dropout_head=dropout_head, **kwargs)
    if cifar_mod:
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()

    if pretrained:
        assert not cifar_mod
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)

    return model


def bayesian_resnet18(*, cifar_mod=False, dropout_head=False, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _bayesian_resnet("resnet18", BasicBlock, [2, 2, 2, 2], cifar_mod, dropout_head, pretrained, progress, **kwargs)


def bayesian_resnet34(*, cifar_mod=False, dropout_head=False, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _bayesian_resnet("resnet34", BasicBlock, [3, 4, 6, 3], cifar_mod, dropout_head, pretrained, progress, **kwargs)


class Cifar10BayesianResnetFactory(ModelOptimizerFactory):
    def create_model_optimizer(self) -> ModelOptimizer:
        model = bayesian_resnet18(cifar_mod=True, dropout_head=True, num_classes=10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        return ModelOptimizer(model=model, optimizer=optimizer)


class Cifar10DeterministicResnetFactory(ModelOptimizerFactory):
    def create_model_optimizer(self) -> ModelOptimizer:
        model = bayesian_resnet18(cifar_mod=True, dropout_head=False, num_classes=10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        return ModelOptimizer(model=model, optimizer=optimizer)


_dataloader_kwargs = dict(num_workers=4, pin_memory=True)


@dataclass
class Cifar10ModelWorkshopPaperTrainer(ModelTrainer):
    device: str

    num_training_samples: int = 1
    num_validation_samples: int = 20
    max_training_epochs: int = 105
    #patience_schedule: [int] = (20,)
    # Added
    patience_schedule: [int] = (20, 10)
    factor_schedule: [int] = (0.1,)

    min_samples_per_epoch: int = 1024
    num_training_batch_size: int = 128
    num_evaluation_batch_size: int = 512

    resnet18_dropout_head: bool = True

    def create_model_optimizer(self) -> ModelOptimizer:
        model = bayesian_resnet18(cifar_mod=True, dropout_head=self.resnet18_dropout_head, num_classes=10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        return ModelOptimizer(model=model, optimizer=optimizer)

    def get_train_dataloader(self, dataset: Dataset):
        train_loader = DataLoader(
            dataset,
            batch_size=self.num_training_batch_size,
            sampler=RandomFixedLengthSampler(dataset, self.min_samples_per_epoch),
            drop_last=True,
            **_dataloader_kwargs
        )
        return train_loader

    def get_evaluation_dataloader(self, dataset: Dataset):
        evaluation_loader = DataLoader(
            dataset, batch_size=self.num_evaluation_batch_size, drop_last=False, shuffle=False, **_dataloader_kwargs
        )
        return evaluation_loader

    def get_trained(
        self,
        *,
        train_loader: DataLoader,
        train_augmentations: Optional[Module],
        validation_loader: DataLoader,
        log,
        wandb_key_path:str,
        loss=None,
        validation_loss=None
    ) -> TrainedModel:
        model_optimizer = self.create_model_optimizer()

        if loss is None:
            loss = torch.nn.NLLLoss()
        if validation_loss is None:
            validation_loss = torch.nn.NLLLoss()

        print("NeurIPS Workshop Style")
        train_with_schedule(
            model=model_optimizer.model,
            optimizer=model_optimizer.optimizer,
            training_samples=self.num_training_samples,
            validation_samples=self.num_validation_samples,
            train_loader=train_loader,
            train_augmentations=train_augmentations,
            validation_loader=validation_loader,
            max_epochs=self.max_training_epochs,
            patience_schedule=self.patience_schedule,
            factor_schedule=self.factor_schedule,
            loss=loss,
            validation_loss=validation_loss,
            device=self.device,
            training_log=log,
            wandb_key_path=wandb_key_path,
        )

        return TrainedBayesianModel(model_optimizer.model)


@dataclass
class Cifar10ModelTrainer(ModelTrainer):
    device: str

    num_training_samples: int = 1
    num_validation_samples: int = 20
    max_training_epochs: int = 300
    patience_schedule: [int] = (10, 10, 5)
    factor_schedule: [int] = (0.1,)

    min_samples_per_epoch: int = 5056
    num_training_batch_size: int = 128
    num_evaluation_batch_size: int = 512

    resnet18_dropout_head: bool = True


    def create_model_optimizer(self) -> ModelOptimizer:
        model = bayesian_resnet18(cifar_mod=True, dropout_head=self.resnet18_dropout_head, num_classes=10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        return ModelOptimizer(model=model, optimizer=optimizer)

    def get_train_dataloader(self, dataset: Dataset):
        train_loader = DataLoader(
            dataset,
            batch_size=self.num_training_batch_size,
            sampler=RandomFixedLengthSampler(dataset, self.min_samples_per_epoch),
            drop_last=True,
            **_dataloader_kwargs
        )
        return train_loader

    def get_evaluation_dataloader(self, dataset: Dataset):
        evaluation_loader = DataLoader(
            dataset, batch_size=self.num_evaluation_batch_size, drop_last=False, shuffle=False, **_dataloader_kwargs
        )
        return evaluation_loader

    def get_trained(
        self,
        *,
        train_loader: DataLoader,
        train_augmentations: Optional[Module],
        validation_loader: DataLoader,
        log,
        wandb_key_path:str,
        loss=None,
        validation_loss=None
    ) -> TrainedModel:
        model_optimizer = self.create_model_optimizer()

        if loss is None:
            loss = torch.nn.NLLLoss()
        if validation_loss is None:
            validation_loss = torch.nn.NLLLoss()

        if False:
            print("With Patience")
            train_with_schedule(
                model=model_optimizer.model,
                optimizer=model_optimizer.optimizer,
                training_samples=self.num_training_samples,
                validation_samples=self.num_validation_samples,
                train_loader=train_loader,
                train_augmentations=train_augmentations,
                validation_loader=validation_loader,
                patience_schedule=self.patience_schedule,
                factor_schedule=self.factor_schedule,
                max_epochs=self.max_training_epochs,
                loss=loss,
                validation_loss=validation_loss,
                device=self.device,
                training_log=log,
                wandb_key_path=wandb_key_path,
            )
        else:
            print("Cosine Annealing")
            train_with_cosine_annealing(
                model=model_optimizer.model,
                optimizer=model_optimizer.optimizer,
                training_samples=self.num_training_samples,
                validation_samples=self.num_validation_samples,
                train_loader=train_loader,
                train_augmentations=train_augmentations,
                validation_loader=validation_loader,
                max_epochs=self.max_training_epochs,
                loss=loss,
                validation_loss=validation_loss,
                device=self.device,
                training_log=log,
                wandb_key_path=wandb_key_path,
            )

        return TrainedBayesianModel(model_optimizer.model)