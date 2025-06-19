import os
import model
import load_datasets
import random
import numpy as np
import torch
import torch.functional as F
from torch import nn
import torch.nn.functional as F
#import itertools
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import itertools
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
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





from torch_geometric.datasets import MoleculeNet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import load_datasets
from torch_geometric.loader import DataLoader



import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from torch_geometric.nn import global_mean_pool





def is_valid_smiles(smi):
    try:
        Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
    except:
        print("not successfully processed smiles: ", smi)
        return False
    return True






df_source_2 = pd.DataFrame(pd.read_csv('HSPiP预测数据N=1.csv'))

labels_D_1=df_source_2['D'].tolist()
labels_P_1=df_source_2['P'].tolist()
labels_H_1=df_source_2['H'].tolist()



df_source_1 = pd.DataFrame(pd.read_csv('test.csv'))
smiles_list_v=df_source_1['SMILES'].tolist()

smiles_list=df_source_1['SMILES'].tolist()
labels_D=df_source_1['D'].tolist()
labels_P=df_source_1['P'].tolist()
labels_H=df_source_1['H'].tolist()
new_smiles_list=[]
for i in smiles_list:
    i_new=i.replace('X','')
    new_smiles_list.append(i_new)

labels=[]

smiles_list_v=[]
labels_D_v=[]
labels_P_v=[]
labels_H_v=[]

labels_D_v_1=[]
labels_P_v_1=[]
labels_H_v_1=[]


for D,P,H,smi,D_v,P_v,H_v in tqdm(zip(labels_D, labels_P,labels_H,new_smiles_list,labels_D_1, labels_P_1,labels_H_1)):
    if not is_valid_smiles(smi): continue
    if D_v=='-': continue
    if P_v=='-': continue
    if H_v=='-': continue
    smiles_list_v.append(smi)
    labels_D_v.append(D)
    labels_P_v.append(P)
    labels_H_v.append(H)
    labels_D_v_1.append(float(D_v))
    labels_P_v_1.append(float(P_v))
    labels_H_v_1.append(float(H_v))
    #print(H_v)
    #labels.append(D)

labels=[]
for D,P,H in tqdm(zip(labels_D_v, labels_P_v,labels_H_v)):
    labels.append([D,P,H])

data=[]
for D,P,H in tqdm(zip(labels_D_v_1, labels_P_v_1,labels_H_v_1)):
    data.append([D,P,H])
print(labels)
rmse = torch.sqrt(torch.tensor(mean_squared_error(labels, data,multioutput='raw_values')))
print(rmse)