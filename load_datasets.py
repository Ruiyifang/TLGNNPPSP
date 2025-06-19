# -*- coding: utf-8 -*-
import json
import random
from random import shuffle
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch_scatter import scatter
from torch_geometric.data import Data, InMemoryDataset, Batch
#from utils import one_of_k_encoding, random_scaffold_split, seed_torch
from torch.nn.functional import one_hot


import rdkit
import sqlite3
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, MolFromSmiles
from rdkit.Chem.rdchem import HybridizationType as HT
from rdkit.Chem.rdchem import BondType as BT
from rdkit import RDLogger


def get_traindata():

    dataset_source = []
    dataset_target = []


    df_source = pd.DataFrame(pd.read_csv('solvent_HSP_hz0828.csv'))
    smiles_list_v=df_source['SMILES'].tolist()
    
    smiles_list=df_source['SMILES'].tolist()
    labels_D=df_source['D'].tolist()
    labels_P=df_source['P'].tolist()
    labels_H=df_source['H'].tolist()

    #D拟合的好？

    labels=[]

    smiles_list_v=[]
    labels_D_v=[]
    labels_P_v=[]
    labels_H_v=[]

    for D,P,H,smi in tqdm(zip(labels_D, labels_P,labels_H,smiles_list)):
        if not is_valid_smiles(smi): continue

        smiles_list_v.append(smi)
        labels_D_v.append(D)
        labels_P_v.append(P)
        labels_H_v.append(H)

        #labels.append(D)

    labels=[]
    for D,P,H in tqdm(zip(labels_D_v, labels_P_v,labels_H_v)):
        labels.append([D,P,H])



    for smi, label in tqdm(zip(smiles_list_v, labels)):
        x, edge_index, edge_attr = get_mol_nodes_edges(Chem.MolFromSmiles(smi))
        dataset_source.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smi=smi, y=torch.tensor([label])))






    df_target = pd.DataFrame(pd.read_csv('polyinfo_HSP_hz0827.csv'))

    smiles_list=df_target['SMILES of monomer'].tolist()
    labels_D=df_target['d'].tolist()
    labels_P=df_target['p'].tolist()
    labels_H=df_target['h'].tolist()

    #D拟合的好？

    labels=[]

    smiles_list_v=[]
    labels_D_v=[]
    labels_P_v=[]
    labels_H_v=[]

    for D,P,H,smi in tqdm(zip(labels_D, labels_P,labels_H,smiles_list)):
        if not is_valid_smiles(smi): continue

        smiles_list_v.append(smi)
        labels_D_v.append(D)
        labels_P_v.append(P)
        labels_H_v.append(H)

        #labels.append(D)

    
    labels=[]
    for D,P,H in tqdm(zip(labels_D_v, labels_P_v,labels_H_v)):
        labels.append([D,P,H])
    #labels=[0,1]
    




    for smi, label in tqdm(zip(smiles_list_v, labels)):
        x, edge_index, edge_attr = get_mol_nodes_edges(Chem.MolFromSmiles(smi))
        dataset_target.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smi=smi, y=torch.tensor([label])))
    

    return dataset_source, dataset_target

def get_testdata(name):

    dataset = []


    df = pd.DataFrame(pd.read_csv('polyinfo_HSP_hz0827.csv'))

    smiles_list=df['SMILES of monomer'].tolist()
    labels_D=df['d'].tolist()
    labels_P=df['p'].tolist()
    labels_H=df['h'].tolist()

    #D拟合的好？

    labels=[]

    smiles_list_v=[]
    labels_D_v=[]
    labels_P_v=[]
    labels_H_v=[]

    for D,P,H,smi in tqdm(zip(labels_D, labels_P,labels_H,smiles_list)):
        if not is_valid_smiles(smi): continue

        smiles_list_v.append(smi)
        labels_D_v.append(D)
        labels_P_v.append(P)
        labels_H_v.append(H)

        #labels.append(D)

    
    labels=[]
    for D,P,H in tqdm(zip(labels_D_v, labels_P_v,labels_H_v)):
        labels.append([D,P,H])
    #labels=[0,1]
    
    labels=labels_D_v
    print(labels_P_v)


    for smi, label in tqdm(zip(smiles_list_v, labels)):
        x, edge_index, edge_attr = get_mol_nodes_edges(Chem.MolFromSmiles(smi))
        dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smi=smi, y=torch.tensor([label])))

    return dataset

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        pass
        # raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def get_mol_nodes_edges(mol):
    gma = get_mol_attr(mol)
    #gma.show_str()
    gma.get_eles()
    gma.get_ring()
    gma.cal_mass()

    # Read node features
    N = mol.GetNumAtoms()
    #print(N)
    x_att_o=[gma.mass, gma.num_ring, gma.congj_ring_num, gma.non_congj_ring_num, gma.fuse_ring_num, gma.non_fuse_ring_num]
    x_att_all=[]
    for i in range(N):
        x_att_all.append(x_att_o)
    atom_type = []
    atomic_number = []
    aromatic = []
    hybridization = []
    # num_hs = []
    for atom in mol.GetAtoms():
        atom_type.append(atom.GetSymbol())
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization.append(atom.GetHybridization())

    # Read edge features
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bond.GetBondType()]
    edge_index = torch.LongTensor([row, col])
    edge_type = [one_of_k_encoding(t, [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]) for t in edge_type]
    edge_attr = torch.FloatTensor(edge_type)
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]
    row, col = edge_index

    # Concat node fetures
    hs = (torch.tensor(atomic_number, dtype=torch.long) == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N).tolist()
    x_atom_type = [one_of_k_encoding(t, ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']) for t in atom_type]
    x_hybridization = [one_of_k_encoding(h, [HT.SP, HT.SP2, HT.SP3]) for h in hybridization]
    x2 = torch.tensor([atomic_number, aromatic, num_hs], dtype=torch.float).t().contiguous()
    #print(x2)
    #print(x_att_all)
    
    #x = torch.cat([torch.FloatTensor(x_atom_type), torch.FloatTensor(x_hybridization), x2, torch.FloatTensor(x_att_all)], dim=-1)
    x = torch.cat([torch.FloatTensor(x_atom_type), torch.FloatTensor(x_hybridization), x2], dim=-1)

    return x, edge_index, edge_attr




def is_valid_smiles(smi):
    try:
        Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
    except:
        print("not successfully processed smiles: ", smi)
        return False
    return True


