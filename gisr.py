import faulthandler
faulthandler.enable()

import os
import time
import argparse
import json
from inspect import signature

from datetime import timedelta, datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.filterwarnings("ignore", message=".*try using `pyenv`.*")
warnings.filterwarnings("ignore", message=".*Matplotlib 3.3.*")

import sympy as sp
import pysr
from pysr import PySRRegressor
pysr.julia_helpers.init_julia()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import FakeHeteroDataset
from torch_geometric.transforms import RandomLinkSplit

from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN, AgglomerativeClustering

from models import *
from sr_utils import expr_similarity


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/', help="Path to directory for loading and saving data.")
parser.add_argument('--k', type=int, default=3, help="Number of clusters to search for.")
parser.add_argument('--max_iters', type=int, default=100, help="Maximum number of GISR training iterations.")
parser.add_argument('--classif_epochs', type=int, default=100, help="Number of classifier training epochs per iteration.")
parser.add_argument('--learning_rate', type=float, default=0.001, help="Initial learning rate for classifier training.")
parser.add_argument('--hidden_dim', type=int, default=32, help="Hidden layer dimension for classifier.")
parser.add_argument('--dataset', type=str, default='synthetic', choices=['synthetic', 'real'], help="Name of dataset.")
parser.add_argument('--cluster_init', type=str, default='random', choices=['random', 'similar'], help="Type of label initialization.")
parser.add_argument('--noise_std', type=float, default=0.0, help="Standard deviation of added Gaussian noise to the synthetic dataset.")
parser.add_argument('--test_size', type=float, default=0.1, help="Portion of dataset that is in the test set.")


def generate_GDE(feature_dim, noise_std, num_leaves = 20):
    def generate_sr_data(func, N, noise_std, range_min=0, range_max=1):
        """Generates synthetic [X | Y] datasets with noise."""
        x_dim = len(signature(func).parameters) # Number of input variables
        x = (range_max - range_min) * torch.rand([N, x_dim]) + range_min
        y = torch.tensor([[func(*x_i)] for x_i in x])
        data = torch.cat([x, y], dim=1) 
        return data + noise_std * torch.randn(data.shape) 

    # Replace these with target functions of interest
    def f1(x0, x1):
        return x0 + x1 

    def f2(x0, x1):
        return x0 * x1 

    def f3(x0, x1):
        return 2*x0 - x1 

    kg_data = HeteroData()

    kg_data['drug'].x = torch.randn(num_leaves, feature_dim)
    kg_data['protein'].x = torch.randn(2,2)

    drug_idxs = torch.arange(num_leaves).unsqueeze(0)
    protein_idxs = torch.randint(2, (1, num_leaves))
    kg_data['drug'].x = torch.cat([protein_idxs, 1-protein_idxs], dim=0).T
    kg_data['drug', 'targets', 'protein'].edge_index = torch.cat([drug_idxs, protein_idxs], dim=0) # [2, num_edges]
    kg_data['protein', 'rev_targets', 'drug'].edge_index = torch.cat([protein_idxs, drug_idxs], dim=0)
    kg_data['protein', 'interacts', 'protein'].edge_index = torch.tensor([[0], [1]])
    kg_data['protein', 'rev_interacts', 'protein'].edge_index = torch.tensor([[1], [0]])
    
    kg_data = kg_data.to_homogeneous()

    x0, x1 = sp.symbols('x0, x1')
    edges = []
    sr_data_stack = []
    expressions = []
    for drug_A_idx in range(num_leaves):
        for drug_B_idx in range(num_leaves):
            edge = sorted([drug_A_idx, drug_B_idx])
            if drug_A_idx != drug_B_idx and edge not in edges:
                edges.append(edge)
                protein_of_A = kg_data.edge_index[1, drug_A_idx]
                protein_of_B = kg_data.edge_index[1, drug_B_idx]

                if protein_of_A == protein_of_B:
                    if protein_of_A == num_leaves: 
                        sr_data_stack.append(generate_sr_data(f1, 9, noise_std))
                        expressions.append(f1(x0, x1))
                    else:
                        sr_data_stack.append(generate_sr_data(f2, 9, noise_std))
                        expressions.append(f2(x0, x1))
                else:
                    sr_data_stack.append(generate_sr_data(f3, 9, noise_std))
                    expressions.append(f3(x0, x1))
    sr_data = torch.stack(sr_data_stack, dim=0)
    edge_filter = torch.tensor(edges).T

    return kg_data, edge_filter, sr_data, expressions


def load_data(data_path, config):
    dataset_name = config.dataset
    if dataset_name == 'synthetic':
        feature_dim = config.hidden_dim
        noise_std = config.noise_std
        
        return generate_GDE(feature_dim, noise_std)
        
    else:
        all_pair_data = pd.read_json(os.path.join(data_path, "drug_combs.json"))

        kg_data = torch.load(os.path.join(data_path, 'kg_subgraph.pt'))
        
        with open(os.path.join(data_path, 'kg_subgraph_node_map.json')) as f:
            mapping = json.load(f)
        
        edge_filter = all_pair_data[['kg_idx_A', 'kg_idx_B']].values.astype(int)
        for original, replacement in mapping.items():
            edge_filter[edge_filter == int(original)] = replacement
        edge_filter = torch.tensor(edge_filter).T

        concs = torch.tensor(np.stack(all_pair_data['concs'].values)).float()
        inhibs = torch.tensor(np.stack(all_pair_data['inhibs'].values)).float()
        sr_data = torch.cat((concs, inhibs), 2)
        
        expressions = None # No ground truth

    return kg_data, edge_filter, sr_data, expressions



def init_clusters(expressions, X_dim, k, dataset_name):
    if dataset_name == 'synthetic':
        n = len(expressions)
        expr_similarities = np.zeros((n, n))
        X_dim = 2

        for i in range(n):
            for j in range(n):
                expr_similarities[i, j] = expr_similarity(expressions[i], expressions[j], X_dim)
    clustering = AgglomerativeClustering(n_clusters = k).fit(expr_similarities)
    return torch.tensor(clustering.labels_)


def init_sr_models(k, config):
    sr_models = []
    result_dir = "results/"
    eq_file = config.dataset + '_' + str(config.cluster_init) + \
                                '_gnn-dim-'+str(config.hidden_dim) + \
                                '_lr-'+str(config.learning_rate) + \
                                '_epochs-'+str(config.classif_epochs) + \
                                '_split-'+str(config.test_size) 
    if config.dataset == 'synthetic':
        eq_file += '_noise-'+str(config.noise_std) 
    if k is not None:
        eq_file += '_k-'+str(config.k) 

    for c in range(k):
        sr_model = PySRRegressor(
            niterations=40,
            binary_operators=["+", "-", "*", "/", "maxim(x,y)=max(x,y)"],
            # unary_operators=["square", "exp", "log", "sqrt"],
            extra_sympy_mappings={"maxim": lambda x, y: sp.Piecewise((x, x > y), (y, True))},
            equation_file = os.path.join(result_dir, eq_file + '_cluster-%d.csv' % c),
            progress=False,
            verbosity=0,
            warm_start=True,
            should_optimize_constants=False,
            procs = 0,
            maxdepth=4,
            turbo=True,
            constraints={"maxim": (1, 1)},
            complexity_of_operators = {'maxim': 5},
        )
        sr_models.append(sr_model)
    return sr_models


def run_sr(sr_data, sr_models, labels):
    expressions = []
    for c, sr_model in enumerate(sr_models):
        start_time = time.time()

        cluster_idxs = (labels == c).nonzero()
        if sum(cluster_idxs) != 0: # Non-empty cluster
            # Update SR model
            sr_data_c = torch.index_select(sr_data, 0, cluster_idxs.squeeze())
            sr_data_c = torch.flatten(sr_data_c, end_dim=-2)
            X_train_c = sr_data_c[:, :-1].detach().cpu().numpy()
            y_train_c = sr_data_c[:, -1:].squeeze().detach().cpu().numpy()

            sr_model.fit(X_train_c, y_train_c)

        with torch.no_grad():
            expr = str(sr_model.sympy())
            cluster_time = time.time() - start_time
            print('-- Expression %d: %s\t| Time elapsed: %f' % (c, expr, cluster_time))
            expressions.append(str(expr))
    
    return sr_models, expressions


def get_pseudo_labels(X, y, sr_models, device):
    errs = []
    for sr_model in sr_models:
        y_pred = torch.tensor(sr_model.predict(X)).reshape(y.shape).to(device) #N x d x 1
        avg_errs = (y_pred - y).pow(2).mean(dim=1).squeeze()
        errs.append(avg_errs)
    errs = torch.stack(errs)
    return torch.argmin(errs, dim=0)


def avg_sr_mse(labels, sr_data, sr_models):
    errs = []
    for c, sr_model in enumerate(sr_models):
        cluster_idxs = (labels == c).nonzero()
        if sum(cluster_idxs) != 0: # Non-empty cluster
            sr_data_c = torch.index_select(sr_data, 0, cluster_idxs.squeeze())
            sr_data_c = torch.flatten(sr_data_c, end_dim=-2)
            X_c = sr_data_c[:, :-1].detach().cpu().numpy()
            y_c = sr_data_c[:, -1:].squeeze().detach().cpu().numpy()

            y_pred = sr_model.predict(X_c)
            
            mse = ((y_pred - y_c) ** 2).mean()
            errs.append(mse)
    return np.mean(errs)


def train(config):
    """ Train GISR."""
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Data preparation
    data_path = config.data_dir
    kg_data, edge_filter, sr_data, all_expressions = load_data(data_path, config)
    kg_data = kg_data.to(device)
    edge_filter = edge_filter.to(device).T
    sr_data = sr_data.to(device)

    # Train-test split
    n_samples = sr_data.shape[0]
    test_size = int(config.test_size * n_samples)
    train_size = n_samples - test_size
    indices = torch.randperm(n_samples)
    train_indices = indices[:train_size]

    sr_data_train = sr_data[train_indices]
    X_train = sr_data_train[:, :, :-1]
    X_train = torch.flatten(X_train, end_dim=-2).detach().cpu().numpy()
    y_train = sr_data_train[:, :, -1:]

    if config.test_size > 0:
        test_indices = indices[train_size:]
        sr_data_test = sr_data[test_indices]
        X_test = sr_data_test[:, :, :-1]
        X_test = torch.flatten(X_test, end_dim=-2).detach().cpu().numpy()
        y_test = sr_data_test[:, :, -1:]


    # Drug pair encoder using RGCN + Bilinear MLP
    pair_encoder = DrugPairEncoder(kg_data.x.shape[1],
                                    config.hidden_dim,
                                    config.k,
                                    kg_data.num_nodes,
                                    kg_data.num_edge_types,
                                    kg_data.num_node_types).to(device)
    optimizer = torch.optim.Adam(pair_encoder.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Initialize labels
    print('Initializing clusters...')
    K = config.k
    if config.cluster_init == 'random':
        classif_train_labels = torch.randint(K, size=(len(sr_data_train),)).to(device)
    elif config.cluster_init == 'similar' and config.dataset == 'synthetic':
        train_expressions = [all_expressions[i] for i in train_indices.tolist()]
        classif_train_labels = init_clusters(train_expressions, X_train.shape[1], K, config.dataset).to(device)
    else:
        raise NotImplementedError("Cluster initialization based on expression similarity is only implemented for synthetic data.")
    counts = torch.bincount(classif_train_labels)
    print('-- Initial cluster distribution: ' + str(counts.tolist()))

    # Initialize SR models
    sr_models = init_sr_models(K, config)

    for iteration in range(config.max_iters):
        print('Iteration: %d' % iteration)
        start_iter_time = time.time()

        #########################################################

        print('[Phase 1] Running symbolic regression...')
        sr_models, expressions = run_sr(sr_data_train, sr_models, classif_train_labels)

        # Calculate the pseudo-labels based on the regression errors
        pseudo_train_labels = get_pseudo_labels(X_train, y_train, sr_models, device)
        pseudo_train_mse = avg_sr_mse(pseudo_train_labels, sr_data_train, sr_models)
        print('-- Average train SR MSE based on pseudo-labels: %f' % (pseudo_train_mse))

        if config.test_size > 0:
            pseudo_test_labels = get_pseudo_labels(X_test, y_test, sr_models, device)
            pseudo_test_mse = avg_sr_mse(pseudo_test_labels, sr_data_test, sr_models)
            print('-- Average test SR MSE based on pseudo-labels: %f' % (pseudo_test_mse))

        pseudo_label_counts = torch.bincount(pseudo_train_labels)
        print('-- Cluster distribution: ' + str(pseudo_label_counts.tolist()))
        
        #########################################################

        print('[Phase 2] Update classifier...')
        for epoch in range(config.classif_epochs):
            # Calculate the classification loss and backpropagate 
            pair_embeds = pair_encoder(kg_data, edge_filter[train_indices].T)

            optimizer.zero_grad()
            ce_train_loss = criterion(pair_embeds, pseudo_train_labels)
            ce_train_loss.backward()
            optimizer.step()

        # Update the labels based on the classifier predictions
        with torch.no_grad():
            pair_train_embeds = pair_encoder(kg_data, edge_filter[train_indices].T)
            classif_train_labels = pair_train_embeds.argmax(dim=1)

            classif_train_mse = avg_sr_mse(classif_train_labels, sr_data_train, sr_models)
            print('-- Average train SR MSE based on predicted classification: %f' % (classif_train_mse))

            if config.test_size > 0:
                pair_test_embeds = pair_encoder(kg_data, edge_filter[test_indices].T)
                classif_test_labels = pair_test_embeds.argmax(dim=1)
                ce_test_loss = criterion(pair_test_embeds, pseudo_test_labels)

                classif_test_mse = avg_sr_mse(classif_test_labels, sr_data_test, sr_models)
                print('-- Average test SR MSE based on predicted classification: %f' % (classif_test_mse))

        classif_label_counts = torch.bincount(classif_train_labels)
        print('-- Cluster distribution: ' + str(classif_label_counts.tolist()))

        iter_time = time.time() - start_iter_time
        print('Iteration: %d\tTime elapsed: %f' % (iteration, iter_time))
        print('Classifier Train SR MSE: %f\tPseudo-label Train SR MSE: %f\tCE Train Loss: %f' % (classif_train_mse, 
                                                                                                pseudo_train_mse,
                                                                                                ce_train_loss.item()))
        if config.test_size > 0:
            print('Classifier Test SR MSE: %f\tPseudo-label Test SR MSE: %f\tCE Test Loss: %f' % (classif_test_mse, 
                                                                                                pseudo_test_mse,
                                                                                                ce_test_loss.item()))
        
        print('Expressions:', expressions)
        print('===================================================================================================')
        print(' ')


if __name__ == '__main__':
    np.random.seed(7)
    torch.manual_seed(7)

    config = parser.parse_args()
    train(config)