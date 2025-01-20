import networkx as nx
import numpy as np
import torch
from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance
from torch_geometric.datasets.graph_generator.ba_graph import BAGraph
from torch_geometric.datasets import ExplainerDataset

class BAShapes(Generator):
        
    def init(self):        
        self.num_instances = self.local_config['parameters']['num_instances']
        self.num_nodes_per_instance = self.local_config['parameters']['num_nodes_per_instance']
        self.num_edges = self.local_config['parameters']['num_edges']
        self.num_motives = self.local_config['parameters']['num_motives']
        self.generate_dataset()
        
    def check_configuration(self):
        super().check_configuration
        local_config=self.local_config

        # set defaults
        local_config['parameters']['num_instances'] = local_config['parameters'].get('num_instances', 1000)
        local_config['parameters']['num_nodes_per_instance'] = local_config['parameters'].get('num_nodes_per_instance', 300)
        local_config['parameters']['num_motives'] = local_config['parameters'].get('num_motives', 80)
        local_config['parameters']['num_edges'] = local_config['parameters'].get('num_edges', 5)
    
    def generate_dataset(self):
        dataset_house = ExplainerDataset(
            graph_generator=BAGraph(num_nodes=self.num_nodes_per_instance, num_edges=self.num_edges),
            motif_generator='house',
            num_motifs=self.num_motives,
            num_graphs=self.num_instances // 2
        )
        dataset_grid = ExplainerDataset(
            graph_generator=BAGraph(num_nodes=self.num_nodes_per_instance, num_edges=self.num_edges),
            motif_generator='grid',
            num_motifs=self.num_motives,
            num_graphs=self.num_instances // 2
        )

        self.__populate_dataset(dataset_house, label=0)
        self.__populate_dataset(dataset_grid, label=1)

    def __populate_dataset(self, dataset, label=0):
         for i in range(len(dataset)):
            adj_matrix = torch.zeros(dataset[i].y.size(0), dataset[i].y.size(0))
            adj_matrix[dataset[i].edge_index[0], dataset[i].edge_index[1]] = 1.0
            adj_matrix[dataset[i].edge_index[1], dataset[i].edge_index[0]] = 1.0

            self.dataset.instances.append(
                GraphInstance(id=i,
                              label=label,
                              data=adj_matrix.numpy())
            )
    