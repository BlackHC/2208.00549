__all__ = ['ModelOptimizer', 'ModelOptimizerFactory']


from dataclasses import dataclass
from typing import Union

import torch.optim

from consistent_mc_dropout import BayesianModule


@dataclass
class ModelOptimizer:
    model: Union[torch.nn.Module, BayesianModule]
    optimizer: torch.optim.Optimizer


class ModelOptimizerFactory:
    def create_model_optimizer(self) -> ModelOptimizer:
        raise NotImplementedError
