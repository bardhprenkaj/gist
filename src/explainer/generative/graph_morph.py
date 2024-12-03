import numpy as np

import os

import tqdm 

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torch_geometric.nn import TransformerConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import degree, to_scipy_sparse_matrix, to_dense_adj

from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import ArpackNoConvergence
from src.core.explainer_base import Explainer
from src.core.factory_base import get_instance_kvargs
from src.core.trainable_base import Trainable
from src.dataset.instances.graph import GraphInstance
from src.dataset.utils.dataset_torch import TorchGeometricDataset, ZippedGraphDataset
from src.explainer.helpers.perturbers.dce import DCEPerturber
from src.utils.cfg_utils import init_dflts_to_of
from src.utils.logger import GLogger

class GraphMorphExplainer(Trainable, Explainer):

    def init(self):
        super().init()        
        self.batch_size = self.local_config['parameters']['batch_size']
        self.input_dim = self.local_config['parameters']['input_dim']
        self.hidden_dim = self.local_config['parameters']['hidden_dim']
        self.heads = self.local_config['parameters']['heads']
        self.epochs = self.local_config['parameters']['epochs']

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = GraphMorph(input_dim=self.input_dim, 
                                hidden_dim=self.hidden_dim, 
                                output_dim=self.input_dim).to(self.device).double()
        
        #self.graph_perturber = SimulatedAnnealingPerturber(oracle=self.oracle, instances=None, max_iterations=500, cooling_rate=0.95, T_0=1.0)
        self.graph_perturber = DCEPerturber(oracle=self.oracle, instances=self.dataset.get_instances_by_indices(fold_id=self.fold_id))
        self.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                             {'params':self.model.parameters(), 
                                              **self.local_config['parameters']['optimizer']['parameters']})

        self._logger = GLogger.getLogger()
        
    def explain(self, instance):
        overshot_graph: GraphInstance = self.graph_perturber.perturb_instance(instance)
        org_label = self.oracle.predict(instance)
        new_label = self.oracle.predict(overshot_graph)
        if org_label != new_label:
            x, edges = self.__inference(TorchGeometricDataset.to_geometric(overshot_graph, label=new_label))
            overshot_graph = TorchGeometricDataset.to_gretel(Data(x=x, y=torch.tensor([new_label]), edge_index=edges))

        return overshot_graph  
        
    def real_fit(self):
        train_loader: DataLoader = self.__preprocess()
        loss_fn = torch.nn.L1Loss()
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for G_style, G_prime in tqdm.tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.epochs} [Training]'):
                train_loss += self.fwd(G_style, G_prime, loss_fn)

            self._logger.info(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss / len(train_loader):.4f}")

    def fwd(self, G_style: Data, G_prime: Data, loss_fn):
        self.optimizer.zero_grad()
        edge_index = G_prime.edge_index.to(torch.int64)
        num_nodes = G_prime.x.size(0)
        # Create a dense adjacency matrix from edge_index
        adj_matrix = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0) 
        # Generate all possible edges
        all_possible_edges = torch.cartesian_prod(torch.arange(num_nodes), torch.arange(num_nodes))
        # Separate positive and negative edges
        pos_edges = edge_index.t()  # Ground truth edges
        neg_edges = all_possible_edges[
            (adj_matrix[all_possible_edges[:, 0], all_possible_edges[:, 1]] == 0)
            & (all_possible_edges[:, 0] != all_possible_edges[:, 1])
        ]  # Non-existent edges, excluding self-loops

        # Combine positive and negative edges
        all_edges = torch.cat([pos_edges, neg_edges], dim=0)

        labels = torch.cat([torch.ones(pos_edges.size(0)), torch.zeros(neg_edges.size(0))], dim=0)
        # Forward pass through the model
        x_styled, edge_probs, selected_edge_index = self.model(G_prime.x, edge_index, edge_pairs=all_edges.t())
        # Loss calculations
        selected_edge_index = selected_edge_index.to(torch.int64)
        G_style.edge_index = G_style.edge_index.to(torch.int64)
        s_loss = self.__style_loss(x_styled, selected_edge_index, G_style.x, G_style.edge_index)
        c_loss = loss_fn(x_styled, G_prime.x) + F.binary_cross_entropy(edge_probs.squeeze().double(), labels.double())  # Example content loss
        loss = s_loss + c_loss
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def __inference(self, data: Data):
        """Apply the trained StyleTransferGNN for inference."""
        self.model.eval()
        with torch.no_grad():
            edge_index = data.edge_index.to(torch.int64)
            num_nodes = data.x.size(0)
            # Create a dense adjacency matrix from edge_index
            adj_matrix = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
            # Generate all possible edges
            all_possible_edges = torch.cartesian_prod(torch.arange(num_nodes), torch.arange(num_nodes))
            # Separate positive and negative edges
            pos_edges = edge_index.t()  # Ground truth edges
            neg_edges = all_possible_edges[
                (adj_matrix[all_possible_edges[:, 0], all_possible_edges[:, 1]] == 0)
                & (all_possible_edges[:, 0] != all_possible_edges[:, 1])
            ]  # Non-existent edges, excluding self-loops
            # Combine positive and negative edges
            all_edges = torch.cat([pos_edges, neg_edges], dim=0)

            x_styled, _, selected_edge_index = self.model(data.x, edge_index, edge_pairs=all_edges.t())

        return x_styled, selected_edge_index
    
    def __style_loss(self, G1_x, G1_edge_index, G2_x, G2_edge_index):
        """
        Compute the total loss between two graphs G1 and G2.

        Args:
            G1 (Data): First PyTorch Geometric Data object.
            G2 (Data): Second PyTorch Geometric Data object.

        Returns:
            torch.Tensor: Total loss combining all components.
        """

        # === 1. Degree Distribution Loss ===


        def degree_distribution(G1_x, G1_edge_index, G2_x, G2_edge_index, eps=1e-8):
            """
            Compute the KL divergence between the degree distributions of two graphs.

            Args:
                G1_x (torch.Tensor): Node features of graph 1.
                G1_edge_index (torch.Tensor): Edge index of graph 1.
                G2_x (torch.Tensor): Node features of graph 2.
                G2_edge_index (torch.Tensor): Edge index of graph 2.
                eps (float): Small value to avoid log(0).

            Returns:
                torch.Tensor: KL divergence loss.
            """
            # Compute degree distributions
            degrees1 = degree(G1_edge_index[0], num_nodes=G1_x.size(0)).float()
            degrees2 = degree(G2_edge_index[0], num_nodes=G2_x.size(0)).float()

            # Normalize degree distributions
            degrees1 = degrees1 / (degrees1.sum() + eps)
            degrees2 = degrees2 / (degrees2.sum() + eps)

            # Add epsilon to avoid log(0) or division by zero
            degrees1 = degrees1 + eps
            degrees2 = degrees2 + eps

            # Compute KL divergence
            kl_loss = -F.kl_div(degrees1.log(), degrees2, reduction='batchmean')
            return kl_loss


        # === 2. Clustering Coefficient Loss ===
        def clustering_coefficient_loss(G1, G2):
            from networkx.algorithms.cluster import clustering
            import networkx as nx

            # Convert edge_index to NetworkX graphs
            G1_nx = nx.Graph()
            G1_nx.add_edges_from(G1.t().tolist())

            G2_nx = nx.Graph()
            G2_nx.add_edges_from(G2.t().tolist())

            # Compute clustering coefficients
            cc1 = torch.tensor(list(clustering(G1_nx).values()), dtype=torch.float)
            cc2 = torch.tensor(list(clustering(G2_nx).values()), dtype=torch.float)

            # MSE loss between clustering coefficients
            return F.mse_loss(cc1, cc2)

        # === 3. Spectral Properties Loss ===
        def spectral_loss(G1, G2):
            # Compute Laplacian matrices
            L1 = to_scipy_sparse_matrix(G1).astype(float)
            L2 = to_scipy_sparse_matrix(G2).astype(float)
            # Compute eigenvalues of Laplacians
            # Increased k, ncv, and maxiter for better convergence
            k_val = min(5, L1.shape[0] - 2)  # Reduced k to be well below matrix size
            ncv_val = min(2 * k_val + 1, L1.shape[0]) # Increased ncv
            # Added try-except block to handle convergence issues
            try:
                if k_val > 0:  # Ensure k is positive
                    eigenvalues1 = eigsh(L1, k=k_val, return_eigenvectors=False, ncv=ncv_val, maxiter=1000, tol=1e-5)  # Increased maxiter and set tolerance
                    eigenvalues2 = eigsh(L2, k=k_val, return_eigenvectors=False, ncv=ncv_val, maxiter=1000, tol=1e-5)  # Increased maxiter and set tolerance
                    # Convert to PyTorch tensors
                    eigenvalues1 = torch.tensor(eigenvalues1, dtype=torch.float)
                    eigenvalues2 = torch.tensor(eigenvalues2, dtype=torch.float)
                    # MSE loss between eigenvalues
                    return F.l1_loss(eigenvalues1, eigenvalues2)
                else:
                    return torch.tensor(0.0, requires_grad=True)  # Handle cases where k is 0
            except ArpackNoConvergence as e:
                print(f"Warning: ARPACK did not converge: {e}")
                # Return a default loss or handle the non-convergence gracefully
                return torch.tensor(0.0, requires_grad=True)

        # === 4. Node Feature Loss ===
        def node_feature_loss(G1, G2):
            if G1 is None or G2 is None:
                return torch.tensor(0.0)  # No node features to compare
            # Ensure both graphs have the same number of nodes
            assert G1.size(0) == G2.size(0), "Graphs must have the same number of nodes for feature loss"
            # MSE loss between node features
            return F.l1_loss(G1, G2)

        # === Compute Total Loss ===
        #loss_deg = degree_distribution(G1_x, G1_edge_index, G2_x, G2_edge_index)
        #loss_cc = clustering_coefficient_loss(G1_edge_index, G2_edge_index)
        loss_spec = spectral_loss(G1_edge_index, G2_edge_index)
        #loss_feat = node_feature_loss(G1_x, G2_x)

        total_loss = loss_spec
        #print(f"Degree Loss: {loss_deg.item():.4f}")
        #print(f"Clustering Coefficient Loss: {loss_cc.item():.4f}")
        #print(f"Spectral Loss: {loss_spec.item():.4f}")
        #print(f"Node Feature Loss: {loss_feat.item():.4f}")
        #print(f"Total Loss: {total_loss.item():.4f}")

        return total_loss
        
    

    def __preprocess(self) -> DataLoader:
        train_graphs: List[GraphInstance] = self.dataset.get_instances_by_indices(fold_id=self.fold_id)
        self.graph_perturber.instances = train_graphs
        #self.graph_perturber = GeneticGraphPerturber(oracle=self.oracle, instances=train_graphs)
        train_graphs = TorchGeometricDataset(train_graphs)
        
        overshot_graphs: List[Data] = []
        if os.path.exists(os.path.join(self.context.dataset_store_path, f'Perturbed_{self.dataset.name}_fold={self.fold_id}.pt')):
            overshot_graphs: TorchGeometricDataset = torch.load(os.path.join(self.context.dataset_store_path, f'Perturbed_{self.dataset.name}_fold={self.fold_id}.pt'))
        else:
            overshot_graphs: TorchGeometricDataset = TorchGeometricDataset(self.graph_perturber.perturb())
            torch.save(overshot_graphs, os.path.join(self.context.dataset_store_path, f'Perturbed_{self.dataset.name}_fold={self.fold_id}.pt'))
            self._logger.info(f'Finished perturbing the dataset')

        # filter only those graphs that have been correctly overshot
        # we don't need to train on wrong pairs of data
        correct_training_graphs: List[Data] = []
        correct_overshot_graphs: List[Data] = []
        for i, graph in enumerate(train_graphs):
            graph_instance = TorchGeometricDataset.to_gretel(graph)
            overshot_instance = TorchGeometricDataset.to_gretel(overshot_graphs.get(i))
            if self.oracle.predict(graph_instance) != self.oracle.predict(overshot_instance):
                correct_training_graphs.append(graph)
                correct_overshot_graphs.append(overshot_graphs.get(i))

        dataset = ZippedGraphDataset(correct_training_graphs, correct_overshot_graphs)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
    
    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['batch_size'] =  self.local_config['parameters'].get('batch_size', 8)
        self.local_config['parameters']['input_dim'] =  self.dataset.num_node_features()
        self.local_config['parameters']['hidden_dim'] =  self.local_config['parameters'].get('hidden_dim', 16)
        self.local_config['parameters']['heads'] =  self.local_config['parameters'].get('heads', 2)
        self.local_config['parameters']['epochs'] =  self.local_config['parameters'].get('epochs', 200)
        init_dflts_to_of(self.local_config, 'optimizer','torch.optim.Adam',lr=1e-3, weight_decay=1e-5)



class GraphMorph(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, heads=2):
        """
        GNN model using a graph transformer for node embeddings and edge probabilities.

        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Dimension of hidden embeddings.
            output_dim (int): Dimension of output node embeddings.
            heads (int): Number of attention heads in the transformer layer.
        """
        super(GraphMorph, self).__init__()
        # Node embedding layers using TransformerConv
        self.trans1 = TransformerConv(input_dim, hidden_dim // heads, heads=heads, dropout=0.1)
        self.trans2 = TransformerConv(hidden_dim, output_dim, heads=1, dropout=0.1)

        # Edge probability prediction layer
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Outputs probabilities between 0 and 1
        )

    def forward(self, x, edge_index, edge_pairs=None, eps=1e-8, temperature=1.0):
        """
        Forward pass for node embeddings and edge probabilities with detailed debugging.
        """
        # Validate edge_index
        if edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape (2, num_edges), but got {edge_index.shape}")
        if edge_index.max() >= x.size(0):
            raise ValueError(f"edge_index contains out-of-bounds indices: max {edge_index.max()}, num_nodes {x.size(0)}")
        if edge_index.size(1) == 0:
            raise ValueError("edge_index is empty.")

        # Node embeddings
        x = self.trans1(x, edge_index)
        x = F.relu(x)

        node_embeddings = self.trans2(x, edge_index)

        # Edge probabilities
        if edge_pairs is None:
            edge_pairs = edge_index

        row, col = edge_pairs
        edge_features = torch.cat([node_embeddings[row], node_embeddings[col]], dim=1)
        edge_probs = self.edge_predictor(edge_features)

        # Differentiable sampling using Gumbel-Softmax
        u = torch.rand_like(edge_probs)
        gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
        logits = torch.log(edge_probs + eps) - torch.log(1 - edge_probs + eps)
        soft_samples = torch.sigmoid((logits + gumbel_noise) / temperature)

        sampled_mask = torch.bernoulli(soft_samples).bool().squeeze()
        sampled_edge_index = edge_pairs[:, sampled_mask]

        return node_embeddings, soft_samples, sampled_edge_index