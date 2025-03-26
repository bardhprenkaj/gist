import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.trainable_base import Trainable

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class UGFormer(nn.Module, Trainable):

    def __init__(self, node_features, ff_hidden_size, num_self_att_layers, dropout, num_gnn_layers, nhead=1, n_classes=2):
        super(UGFormer, self).__init__()
        self.node_features = node_features
        self.ff_hidden_size = ff_hidden_size
        self.n_classes = n_classes
        self.num_self_att_layers = num_self_att_layers #Each layer consists of a number of self-attention layers
        self.num_gnn_layers = num_gnn_layers
        self.nhead = nhead
        self.lst_gnn = torch.nn.ModuleList()
        #
        self.ugformer_layers = torch.nn.ModuleList()
        for _layer in range(self.num_gnn_layers):
            encoder_layers = TransformerEncoderLayer(d_model=self.node_features,
                                                     nhead=self.nhead,
                                                     dim_feedforward=self.ff_hidden_size,
                                                     dropout=0.5)
            self.ugformer_layers.append(TransformerEncoder(encoder_layers, self.num_self_att_layers))
            self.lst_gnn.append(GraphConvolution(self.node_features, self.node_features, act=torch.relu))

        # Linear function
        self.predictions = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            self.predictions.append(nn.Linear(self.node_features, self.n_classes))
            self.dropouts.append(nn.Dropout(dropout))

        self.prediction = nn.Linear(self.node_features, self.n_classes)
        self.dropout = nn.Dropout(dropout)

    def init(self):
        return super().init()
    
    def real_fit(self):
        return
        
    def forward(self, Adj_block, node_features):
        input_Tr = node_features
        for layer_idx in range(self.num_gnn_layers):
            # self-attention over all nodes
            input_Tr = torch.unsqueeze(input_Tr, 1)  #[seq_length, batch_size=1, dim] for pytorch transformer
            input_Tr = self.ugformer_layers[layer_idx](input_Tr)
            input_Tr = torch.squeeze(input_Tr, 1)
            # take a sum over neighbors followed by a linear transformation and an activation function --> similar to GCN
            input_Tr = self.lst_gnn[layer_idx](input_Tr, Adj_block)
            # take a sum over all node representations to get graph representations
            graph_embedding = torch.sum(input_Tr, dim=0)
        graph_embedding = self.dropout(graph_embedding)
        # # Produce the final scores
        return self.prediction(graph_embedding)

    def fwd(self, x, edge_index, edge_weights, batch, labels, loss_fn):
        Adj_block = self.get_Adj_matrix(x, edge_index)
        pred = self(Adj_block, x)
        return pred, loss_fn(torch.unsqueeze(pred, 0), labels)

    def get_Adj_matrix(self, node_features, edge_index):
        Adj_block_idx = torch.LongTensor(edge_index)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        num_node = len(node_features)
        self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
        elem = torch.ones(num_node)
        Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
        Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([num_node, num_node]))

        return Adj_block.to(self.device) # can implement and tune for the re-normalized adjacency matrix D^-1/2AD^-1/2 or D^-1A like in GCN/SGC ???


""" GCN layer, similar to https://arxiv.org/abs/1609.02907 """
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, act=torch.relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.act = act
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        output = self.bn(output)
        return self.act(output)
