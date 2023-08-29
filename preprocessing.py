import argparse
import os
import json

from tqdm import tqdm

from itertools import product

import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.utils import is_undirected, to_undirected, to_dense_adj, k_hop_subgraph, coalesce


import numpy as np
import pandas as pd

import reservoir as rsv

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/', help="Path to directory for loading and saving data.")
parser.add_argument('--combo_measurements', type=int, default=9, help="Retrieve all experiment blocks that have this many combo measurements.")
parser.add_argument('--mono_row_measurements', type=int, default=4, help="Retrieve all experiment blocks that have this many mono measurements for the row drug.")
parser.add_argument('--mono_col_measurements', type=int, default=4, help="Retrieve all experiment blocks that have this many mono measurements for the column drug.")
parser.add_argument('--cell_line', type=str, default='MCF7', help="Retrieve all experiment blocks from this cell line.")
parser.add_argument('--study', type=str, default=None, help="Retrieve all experiment blocks from this study.")


class DrugCombSet(Dataset):
    '''Drug combo Dataset object '''
    
    def __init__(self, X, y=None):
        width = X.shape[1]//2
        self.X_a = X[:, :width]
        self.X_b = X[:, width:]
        self.y_mono = y[:, [0, 1]]
        self.y = y[:, [2]]
        
    def __len__(self):
        return len(self.X_a)
    
    def __getitem__(self, index):
        return self.X_a[index], self.X_b[index], self.y[index], self.y_mono[index] 

        
class DrugPair:
    '''Drug pair data object '''
    
    def __init__(self, smile_A, smile_B):
        self.smile_A = smile_A 
        self.smile_B = smile_B 
        
    def set_inhibs(self, inhibs):
        self.inhibs = inhibs
        
    def set_concs(self, concs):
        self.concs = concs
        

class DrugCombLoader:
    '''Drug combination data object for data processing and feature generation'''
    
    def __init__(self, data_dir, cell_line=None, study=None, combo_measurements=9, mono_row_measurements=4, mono_col_measurements=4):
        self.data_dir = data_dir
        self.cell_line = cell_line
        self.study = study
        self.combo_measurements = combo_measurements
        self.mono_row_measurements = mono_row_measurements
        self.mono_col_measurements = mono_col_measurements

        #Load DrugComb data
        blocks = rsv.get_specific_drug_combo_blocks(combo_measurements=combo_measurements, 
                                                    mono_row_measurements=mono_row_measurements, 
                                                    mono_col_measurements=mono_col_measurements,
                                                    cell_line_name=cell_line,
                                                    study_name=study)
        self.mono_df = rsv.get_drug_combo_data_monos(block_ids=blocks['block_id'])
        self.combo_df = rsv.get_drug_combo_data_combos(block_ids=blocks['block_id'])

        # Load PrimeKG 
        self.kg_hetero = torch.load(os.path.join(self.data_dir, "kg_hetero.pt"))
        self.kg_drug_node_lookup = pd.read_csv(os.path.join(self.data_dir, "kg_drug_node_lookup.csv"))

  
    def get_pair_data(self, filtered_df, block_id):
        '''
        Args: 
            filtered_df (DataFrame): DrugComb dataset filtered by cell line
            block_id (str): ID of singular drug combo experiment in filtered_df
        Return: 
            drug_pair (DrugPair): Drug pair object
        '''
        combo_row = filtered_df[filtered_df.block_id == block_id]
        
        # Get smiles strings
        smile_pair = combo_row[['drug_row_smiles', 'drug_col_smiles']].values
        smile_A = smile_pair[0][0]
        smile_B = smile_pair[0][1]

        smiles_set = set(self.kg_drug_node_lookup['canonical_smiles'])
        if (smile_A not in smiles_set) or (smile_B not in smiles_set):
            return None
    
        # Get combined inhibition data (responses)
        combo_inhib_list = list(set(combo_row.inhibitions.values[0]))
        combo_inhibs = np.array(combo_inhib_list).reshape(-1, 1)

        # Get combined concentration data (responses)
        combo_conc_arr = np.array(combo_row.concentration_pairs.values[0])
        combo_conc_list = [i[np.sort(np.unique(i, axis=0, return_index=True)[1])] for i in combo_conc_arr] # Remove duplicates

        if len(combo_inhib_list) != len(combo_conc_list):
            combo_conc_list = combo_conc_list[::2]

        new_combo_conc_list = []
        for pair in combo_conc_list:
            # If concentrations are the same, there is only 1 entry in the array
            if len(pair) < 2:
                new_combo_conc_list.append(np.array([pair.item(), pair.item()]))
            else:
                new_combo_conc_list.append(pair)

        combo_concs = np.array(new_combo_conc_list)

        # Get individual responses 
        top_mono_row = self.mono_df[self.mono_df.block_id == block_id]
        mono_inhibs_arr = top_mono_row[['inhibition_r', 'inhibition_c']].values[0]

        # Remove duplicates
        mono_inhibs_A = list(set(mono_inhibs_arr[0][1:])) 
        mono_inhibs_B = list(set(mono_inhibs_arr[1][1:]))

        # Some combo data has extra values for concentrations of 0
        if len(mono_inhibs_A)**2 > len(combo_inhib_list):
            mono_inhibs_A = mono_inhibs_A[1:]
            mono_inhibs_B = mono_inhibs_B[1:]

            if len(mono_inhibs_A)*len(mono_inhibs_B) != len(combo_inhib_list):
                return None

        # Cartesian product of non-zero concentration combos
        mono_inhibs = np.array(list(product(mono_inhibs_A, mono_inhibs_B)))

        # Convert inhibition percentages to value between [-1, 1]
        concat_inhibs = 1/100 * np.concatenate([mono_inhibs, combo_inhibs], axis=1)
        
        drug_pair = DrugPair(smile_A, smile_B)
        drug_pair.set_inhibs(concat_inhibs)
        drug_pair.set_concs(combo_concs)
        drug_pair.synergy_zip = combo_row['synergy_zip'].item()
        drug_pair.synergy_bliss = combo_row['synergy_bliss'].item()
        drug_pair.synergy_loewe = combo_row['synergy_loewe'].item()
        drug_pair.synergy_hsa = combo_row['synergy_hsa'].item()
        return drug_pair
        

    def process_drugcomb_data(self):
        '''
        Generates and saves the drug interaction data ensemble using DrugComb 

        Output file:
            'drug_combs.json': JSON formatted dataframe of drug combination data
        '''
        combo_df = self.combo_df
        drug_combs_dataset = [] # Rows of drug_comb_df

        for block_id in tqdm(list(combo_df.block_id), desc='Querying selected drug pairs...'):
            drug_pair = self.get_pair_data(combo_df, block_id)
            if drug_pair is not None:
                pair_data = {
                    'block_id': block_id,
                    'kg_idx_A': self.kg_drug_node_lookup.loc[self.kg_drug_node_lookup['canonical_smiles'] == drug_pair.smile_A, 'kg_idx'].item(),
                    'kg_idx_B': self.kg_drug_node_lookup.loc[self.kg_drug_node_lookup['canonical_smiles'] == drug_pair.smile_B, 'kg_idx'].item(),
                    'smile_A': drug_pair.smile_A,
                    'smile_B': drug_pair.smile_B,
                    'concs': drug_pair.concs.tolist(),
                    'inhibs': drug_pair.inhibs.tolist(),
                    'synergy_zip':drug_pair.synergy_zip,
                    'synergy_bliss':drug_pair.synergy_bliss,
                    'synergy_loewe':drug_pair.synergy_loewe,
                    'synergy_hsa':drug_pair.synergy_hsa,
                }

                drug_combs_dataset.append(pair_data)
        
        with open(os.path.join(self.data_dir, 'processed/drug_combs.json'), 'w') as f:
            json.dump(drug_combs_dataset, f)

    
    def process_kg_data(self):
        '''
        Generates and saves the subgraph of PrimeKG that corresponds to the DrugComb data

        Output files:
            'kg_subgraph_node_map.json': Mapping from DrugComb node IDs to PrimeKG subgraph indices
            'kg_subgraph.pt': PyTorch Geometric HeteroData object for filtered PrimeKG subgraph
        '''
        drug_combs_df = pd.read_json(os.path.join(self.data_dir, 'processed/drug_combs.json'))
        kg_hetero = self.kg_hetero

        edge_types = [
        ('drug', 'contraindication', 'disease'),
        ('disease', 'contraindication', 'drug'),
        ('gene/protein', 'drug_protein', 'drug'),
        ('drug', 'drug_protein', 'gene/protein'),
        ('drug', 'indication', 'disease'),
        ('disease', 'indication', 'drug'),
        ('drug', 'off-label use', 'disease'),
        ('disease', 'off-label use', 'drug')
        ]
        kg_subgraph = kg_hetero.edge_type_subgraph(edge_types)

        node_filter = torch.tensor(np.unique(drug_combs_df[['kg_idx_A', 'kg_idx_B']].values.astype(int)))
        mapping = {x.item(): i for i, x in enumerate(node_filter)}
        with open(os.path.join(self.data_dir, 'processed/kg_subgraph_node_map.json'), 'w') as f:
            json.dump(mapping, f)
        
        kg_subgraph = kg_subgraph.to_homogeneous()

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_filter, 
                                                                1, 
                                                                kg_subgraph.edge_index,
                                                                directed=False,
                                                                relabel_nodes=True)

        kg_subgraph.x = kg_subgraph.x[subset]
        kg_subgraph.node_type = kg_subgraph.node_type[subset]
        kg_subgraph.edge_index = edge_index
        kg_subgraph.edge_type = kg_subgraph.edge_type[edge_mask]

        torch.save(kg_subgraph, os.path.join(self.data_dir, "processed/kg_subgraph.pt"))
                   
    
if __name__ == '__main__':
    args = parser.parse_args()

    drugCombLoader = DrugCombLoader(args.data_dir, 
                                    args.cell_line, 
                                    args.study,
                                    args.combo_measurements, 
                                    args.mono_row_measurements, 
                                    args.mono_col_measurements)
    drugCombLoader.process_drugcomb_data()
    drugCombLoader.process_kg_data()

   