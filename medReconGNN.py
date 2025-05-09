import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
import numpy as np
import random
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, RGCNConv
import pandas as pd


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def similarity_loss(embeddings, similar_pairs, dissimilar_pairs, margin=1.0):
    """
    Compute similarity loss: aiming for smaller loss for similiar drugs and bigger loss for dissimilar drugs

    params:
    - embeddings: for drug
    - similar_pairs: [(i, j), ...] list of similar pair
    - dissimilar_pairs: [(i, j), ...] list of disimilar pair
    - margin: minimum distance between disimilar drug pair

    return:
    - loss: L2
    """
    loss = 0
    # similar drug should be close
    for i, j in similar_pairs:
        loss += torch.norm(embeddings[i] - embeddings[j], p=2)

    # dissimilar drug should be far away
    for i, j in dissimilar_pairs:
        loss += F.relu(margin - torch.norm(embeddings[i] - embeddings[j], p=2))
    
    return loss / (len(similar_pairs) + len(dissimilar_pairs))


class DrugDiseaseHeteroGAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        
        # Use HeteroConv to assign different conv layer to different relationships
        self.convs = HeteroConv({
            ('DrugProfile', 'TREATS', 'DiseaseProfile'): GATConv((-1,-1), hidden_channels, add_self_loops=False),
            ('DiseaseProfile', 'TREATED_BY', 'DrugProfile'): GATConv((-1,-1), hidden_channels, add_self_loops=False),
            ('DrugProfile', 'HAS_CONTRAIND', 'DiseaseProfile'): GATConv((-1,-1), hidden_channels, add_self_loops=False),
            ('DiseaseProfile', 'CONTRAIND_BY', 'DrugProfile'): GATConv((-1,-1), hidden_channels, add_self_loops=False),
            ('DiseaseProfile', 'HAS_PARENTCODE', 'DiseaseProfile'): GATConv((-1,-1), hidden_channels, add_self_loops=False),
            ('DiseaseProfile', 'HAS_CHILDCODE', 'DiseaseProfile'): GATConv((-1,-1), hidden_channels, add_self_loops=False),
            ('DrugProfile', 'INTERACTS', 'DrugProfile'): GATConv(-1, hidden_channels, add_self_loops=False)
        }, aggr='sum')  # aggregator can also be 'mean' or 'max'

        self.convs2 = HeteroConv({
            ('DrugProfile', 'TREATS', 'DiseaseProfile'): GATConv((-1,-1), hidden_channels, add_self_loops=False),
            ('DiseaseProfile', 'TREATED_BY', 'DrugProfile'): GATConv((-1,-1), hidden_channels, add_self_loops=False),
            ('DrugProfile', 'HAS_CONTRAIND', 'DiseaseProfile'): GATConv((-1,-1), hidden_channels, add_self_loops=False),
            ('DiseaseProfile', 'CONTRAIND_BY', 'DrugProfile'): GATConv((-1,-1), hidden_channels, add_self_loops=False),
            ('DiseaseProfile', 'HAS_PARENTCODE', 'DiseaseProfile'): GATConv((-1,-1), hidden_channels, add_self_loops=False),
            ('DiseaseProfile', 'HAS_CHILDCODE', 'DiseaseProfile'): GATConv((-1,-1), hidden_channels, add_self_loops=False),
            ('DrugProfile', 'INTERACTS', 'DrugProfile'): GATConv(-1, hidden_channels, add_self_loops=False)
        }, aggr='sum')  # aggregator can also be 'mean' or 'max'

        # output layer
        self.lin_drug = torch.nn.Linear(hidden_channels, out_channels)
        self.lin_disease = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # use HeteroConv to feed forward
        x_dict = self.convs(x_dict, edge_index_dict)
        x_dict = {key: self.dropout(F.relu(x)) for key, x in x_dict.items()}
        x_dict = self.convs2(x_dict, edge_index_dict)
        # linear + ReLU activation 
        x_dict['DrugProfile'] = F.relu(self.lin_drug(x_dict['DrugProfile']))
        x_dict['DiseaseProfile'] = F.relu(self.lin_disease(x_dict['DiseaseProfile']))
        
        return x_dict


class DrugDiseaseHeteroSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        
        # Use HeteroConv to assign different conv layer to different relationships
        self.convs = HeteroConv({
            ('DrugProfile', 'TREATS', 'DiseaseProfile'): SAGEConv((-1,-1), hidden_channels),
            ('DiseaseProfile', 'TREATED_BY', 'DrugProfile'): SAGEConv((-1,-1), hidden_channels),
            ('DrugProfile', 'HAS_CONTRAIND', 'DiseaseProfile'): SAGEConv((-1,-1), hidden_channels),
            ('DiseaseProfile', 'CONTRAIND_BY', 'DrugProfile'): SAGEConv((-1,-1), hidden_channels),
            ('DrugProfile', 'HAS_PARENTCODE', 'DrugProfile'): SAGEConv((-1,-1), hidden_channels),
            ('DrugProfile', 'HAS_CHILDCODE', 'DrugProfile'): SAGEConv((-1,-1), hidden_channels),
            ('DiseaseProfile', 'HAS_PARENTCODE', 'DiseaseProfile'): SAGEConv((-1,-1), hidden_channels),
            ('DiseaseProfile', 'HAS_CHILDCODE', 'DiseaseProfile'): SAGEConv((-1,-1), hidden_channels),
            ('DrugProfile', 'INTERACTS', 'DrugProfile'): SAGEConv(-1, hidden_channels)
        }, aggr='sum')  # aggregator can also be 'mean' or 'max'

        self.convs2 = HeteroConv({
            ('DrugProfile', 'TREATS', 'DiseaseProfile'): SAGEConv((-1,-1), hidden_channels),
            ('DiseaseProfile', 'TREATED_BY', 'DrugProfile'): SAGEConv((-1,-1), hidden_channels),
            ('DrugProfile', 'HAS_CONTRAIND', 'DiseaseProfile'): SAGEConv((-1,-1), hidden_channels),
            ('DiseaseProfile', 'CONTRAIND_BY', 'DrugProfile'): SAGEConv((-1,-1), hidden_channels),
            ('DrugProfile', 'HAS_PARENTCODE', 'DrugProfile'): SAGEConv((-1,-1), hidden_channels),
            ('DrugProfile', 'HAS_CHILDCODE', 'DrugProfile'): SAGEConv((-1,-1), hidden_channels),
            ('DiseaseProfile', 'HAS_PARENTCODE', 'DiseaseProfile'): SAGEConv((-1,-1), hidden_channels),
            ('DiseaseProfile', 'HAS_CHILDCODE', 'DiseaseProfile'): SAGEConv((-1,-1), hidden_channels),
            ('DrugProfile', 'INTERACTS', 'DrugProfile'): SAGEConv(-1, hidden_channels)
        }, aggr='sum')  # aggregator can also be 'mean' or 'max'

        # output layer
        self.lin_drug = torch.nn.Linear(hidden_channels, out_channels)
        self.lin_disease = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # use HeteroConv to feed forward
        x_dict = self.convs(x_dict, edge_index_dict)
        x_dict = {key: self.dropout(F.relu(x)) for key, x in x_dict.items()}
        x_dict = self.convs2(x_dict, edge_index_dict)
        # linear + ReLU activation 
        x_dict['DrugProfile'] = F.relu(self.lin_drug(x_dict['DrugProfile']))
        x_dict['DiseaseProfile'] = F.relu(self.lin_disease(x_dict['DiseaseProfile']))
        
        return x_dict

if __name__ == 'main':
    # Initialization
    model = DrugDiseaseHeteroGAT(hidden_channels=64, out_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    with open('data/node_namp_map.pkl','rb') as f:
        node_name_map = pickle.load(f)
    
    data = torch.load('data/hetero_data.pt')

    # Define similar and dissimilar pairs
    similar_pairs = [(node_name_map['Hydrochlorothiazide'], node_name_map['Labetalol']),  # similar treatment 
                    (node_name_map['Acetylsalicylic acid'], node_name_map['Clopidogrel'])]           # similar treatment 
    dissimilar_pairs = [(node_name_map['Hydrochlorothiazide'], node_name_map['Acetylsalicylic acid']),  # interaction
                        (node_name_map['Salbutamol'], node_name_map['Clopidogrel'])]       # conindication

    # set_seed(42)
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        out = model(data.x_dict, data.edge_index_dict)
        drug_embeddings = out['DrugProfile']

        loss = similarity_loss(drug_embeddings, similar_pairs, dissimilar_pairs)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
