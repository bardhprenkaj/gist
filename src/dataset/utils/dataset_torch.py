import numpy as np

from typing import List, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GeometricDataset
from torch.utils.data import Dataset

from src.dataset.instances.graph import GraphInstance
from src.utils.utils import pad_adj_matrix
    
class TorchGeometricDataset(GeometricDataset):
  
  def __init__(self, instances: List[GraphInstance]):
    super(TorchGeometricDataset, self).__init__()    
    self.instances = []
    self._process(instances)
    
  def len(self):
    return len(self.instances)
  
  def get(self, idx):
    return self.instances[idx]
  
  def _process(self, instances: List[GraphInstance]):
    self.instances = [self.to_geometric(inst, label=inst.label) for inst in instances]
      
  @classmethod
  def to_geometric(self, instance: GraphInstance, label=0) -> Data:   
    adj = torch.from_numpy(instance.data).double()
    x = torch.from_numpy(instance.node_features).double()
    a = torch.nonzero(adj).int()
    w = torch.from_numpy(instance.edge_weights).double()
    label = torch.tensor(label).long()
    return Data(x=x, y=label, edge_index=a.T, edge_attr=w)
  
  @classmethod
  def to_gretel(self, data: Data) -> GraphInstance:
    node_features = data.x.double().numpy()
    adj_matrix = torch.zeros((node_features.shape[0], node_features.shape[0]), dtype=torch.float)
    edge_index = data.edge_index
    adj_matrix[edge_index[0,:], edge_index[1,:]] = 1.0
    edge_weights = None
    try:
       if hasattr(data, 'edge_attr') and data.edge_attr.size(0):
          edge_weights = data.edge_attr.double().numpy()
    except:
       edge_weights = None

    return GraphInstance(id=-1, label=data.y.numpy(),
                        data=adj_matrix.numpy(), node_features=node_features,
                        edge_weights=edge_weights) 
 
  
class TorchDataset(Dataset):
  
    def __init__(self, instances: List[GraphInstance], max_nodes=10):
        super(TorchDataset, self).__init__()
        self.instances = instances
        self.max_nodes = max_nodes
        self._process(instances)
      
    def _process(self, instances: List[GraphInstance]):
        for i, instance in enumerate(instances):
            padded_adj = pad_adj_matrix(instance.data, self.max_nodes)
            # create a new instance
            new_instance = GraphInstance(id=instance.id,
                                        label=instance.label,
                                        data=padded_adj,
                                        dataset=instance._dataset)
            # redo the manipulators
            instance._dataset.manipulate(new_instance)
            instances[i] = new_instance
        
        self.instances = [self.to_geometric(inst, label=inst.label) for inst in instances]
 
    @classmethod
    def to_geometric(self, instance: GraphInstance, label=0):   
        adj = torch.from_numpy(instance.data).double()
        x = torch.from_numpy(instance.node_features).double()
        label = torch.tensor(label).long()
        return adj, x, label
    
    def __getitem__(self, index):
        return self.instances(index)
    

class ZippedGraphDataset(GeometricDataset):
    
    def __init__(self, dataset1: List[Data], dataset2: List[Data]):
        """
        A dataset that zips two TorchGeometricDatasets together.
        Args:
            dataset1 (List[Data]): First dataset (e.g., train_graphs).
            dataset2 (List[Data]): Second dataset (e.g., overshot_graphs).
        """
        super(ZippedGraphDataset, self).__init__()
        assert len(dataset1) == len(dataset2), "Datasets must be of the same length"
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def len(self):
      return len(self.dataset1)
  
    def get(self, idx) -> Tuple[Data, Data]:
      return self.dataset1[idx], self.dataset2[idx]