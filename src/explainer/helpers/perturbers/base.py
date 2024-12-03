import random
import torch

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import List

from torch_geometric.data import Data

from src.dataset.instances.graph import GraphInstance
from src.dataset.utils.dataset_torch import TorchGeometricDataset


class GraphPerturber(metaclass=ABCMeta):

    def __init__(self, instances: List[GraphInstance]):
        self.instances: List[GraphInstance] = instances

    @abstractmethod
    def perturb(self) -> List[GraphInstance]:
        pass