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


#dataset.num_node_features=torch.Tensor(15)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hidden_channels = 64

class GNN(nn.Module):
    
    def __init__(self):
        # 初始化Pytorch父类
        super().__init__()
        
        #self.conv1=GCNConv(21, hidden_channels)
        self.conv1=GCNConv(15, hidden_channels)
        #print(dataset.num_node_features)
        self.conv2=GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        #self.out = nn.Linear(hidden_channels, 1)
        
        # 创建损失函数，使用RMSE均方误差
        #self.loss_function = nn.MSELoss()

    
    # 前向传播函数
    def forward(self, x, edge_index,batch):
        
        x=x.to(device)
        edge_index=edge_index.to(device)
        batch=batch.to(device)

        x=self.conv1(x, edge_index)
        x=x.relu()
        x=self.conv2(x, edge_index)
        x=x.relu()
        x=self.conv3(x, edge_index)
        x=x.relu()
        x=self.conv4(x, edge_index)
        x=x.relu()

        # 全局池化
        x = global_mean_pool(x, batch)  # [x, batch]

        out=self.out(x)
        return out