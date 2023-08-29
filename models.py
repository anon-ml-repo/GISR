import random
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import torch_geometric.transforms as T
from torch_geometric.nn.conv import HANConv, RGCNConv, RGATConv
from torch_geometric.nn.norm import HeteroBatchNorm


########################################################################################################################
# GNNs
########################################################################################################################

class HAN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int, metadata, negative_slope: float = 0.2, dropout: float = 0):
        super(HAN, self).__init__()
        self.conv1 = HANConv(in_channels, hidden_channels, metadata, heads, negative_slope=negative_slope, dropout=dropout)
        self.conv2 = HANConv(hidden_channels, hidden_channels, metadata, heads, negative_slope=negative_slope, dropout=dropout)
        
    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: emb.relu() for key, emb in x.items()}
        x = self.conv2(x, edge_index_dict)
        return x


class RGCN(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_relations):
        super(RGCN, self).__init__()
        # self.node_emb = Parameter(torch.empty(num_nodes, hidden_channels))
        print('input channels', input_channels)
        self.conv1 = RGCNConv(input_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.conv3 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_type)
        x = F.relu(x)
        return x
    

class RGAT(nn.Module):
    def __init__(self, hidden_channels, num_relations):
        super(RGAT, self).__init__()
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_type, edge_attr):
        x = self.conv1(x, edge_index, edge_type, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type, edge_attr)
        x = F.relu(x)
        return x
    

########################################################################################################################
# Drug pair encoder
########################################################################################################################

class ComboMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ComboMLP, self).__init__()
        
        self.bilinear = torch.nn.Bilinear(input_dim, input_dim, input_dim)
        
        self.combo_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )
        
    def forward(self, x_a, x_b):
        x_ab = self.bilinear(x_a, x_b).relu()
        x_ab = self.combo_mlp(x_ab)
        return x_ab


class DrugPairEncoder(nn.Module):
    def __init__(self, input_dim, gnn_dim, embed_dim, num_nodes, num_relations, num_node_types):
        super(DrugPairEncoder, self).__init__()

        self.gnn_dim = gnn_dim
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.num_node_types = num_node_types

        # Heterogeneous GNN
        self.kg_encoder = RGCN(input_channels=input_dim,
                              hidden_channels=gnn_dim, 
                              num_relations=num_relations)
        
        
        # self.kg_encoder = RGAT(gnn_dim, num_relations)

        # Bilinear MLP for drug pair embeddings
        # self.bilinear = torch.nn.Bilinear(gnn_dim, gnn_dim, gnn_dim)
        self.combo_mlp = ComboMLP(gnn_dim, embed_dim)

    def forward(self, kg_data, edge_filter):
        kg_out = self.kg_encoder(kg_data.x,
                                kg_data.edge_index, 
                                kg_data.edge_type)

        # Get drug node embeddings
        drug_embeds = kg_out[torch.where(kg_data.node_type == 0)[0]]
       
        # Filter for drug pairs that have combo response data
        features_A = drug_embeds[edge_filter[0, :]] 
        features_B = drug_embeds[edge_filter[1, :]]
   
        # Get drug pair embeddings
        pair_embeds = self.combo_mlp(features_A, features_B)
        return pair_embeds
    

########################################################################################################################
# ReplayBuffer
########################################################################################################################

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, idx, state, action, reward, next_state):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (idx, state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


########################################################################################################################
# Clustering algorithms
########################################################################################################################

class DiffClusterer(torch.nn.Module):
    def __init__(self, num_clusters, init_stiffness=1.0, stiffness_inc=0.0):
        super(DiffClusterer, self).__init__()
        self.num_clusters = num_clusters
        self.stiffness = init_stiffness
        self.stiffness_inc = stiffness_inc
        self.centroids = Parameter(torch.empty(num_clusters, num_clusters))
        torch.nn.init.xavier_normal_(self.centroids)

    def forward(self, x, return_centers=True):
        dists = torch.cdist(x, self.centroids)
        assignments = F.softmax(-self.stiffness * dists, dim=1)

        self.stiffness += self.stiffness_inc
        
        if return_centers:
            return assignments, self.centroids
        return assignments


