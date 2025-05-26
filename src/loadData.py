# import gzip
# import json
# import torch
# from torch_geometric.data import Dataset, Data
# import os
# from tqdm import tqdm 
# from torch_geometric.loader import DataLoader

# class GraphDataset(Dataset):
#     def __init__(self, filename, transform=None, pre_transform=None):
#         self.raw = filename
#         self.graphs = self.loadGraphs(self.raw)
#         super().__init__(None, transform, pre_transform)

#     def len(self):
#         return len(self.graphs)

#     def get(self, idx):
#         return self.graphs[idx]

#     @staticmethod
#     def loadGraphs(path):
#         print(f"Loading graphs from {path}...")
#         print("This may take a few minutes, please wait...")
#         with gzip.open(path, "rt", encoding="utf-8") as f:
#             graphs_dicts = json.load(f)
#         graphs = []
#         for graph_dict in tqdm(graphs_dicts, desc="Processing graphs", unit="graph"):
#             graphs.append(dictToGraphObject(graph_dict))
#         return graphs



# def dictToGraphObject(graph_dict):
#     edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
#     edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float) if graph_dict["edge_attr"] else None
#     num_nodes = graph_dict["num_nodes"]
#     y = torch.tensor(graph_dict["y"][0], dtype=torch.long) if graph_dict["y"] is not None else None
#     return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)



import gzip
import json
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.model_selection import KFold, train_test_split


def load_data(files, round, n_folds, train_folds_to_use, test_size=0.2, seed=42):
    files = files.split(" ")
    train_graphs, val_graphs = [], []
    for file in files:
        with gzip.open(file, "rt", encoding="utf-8") as f:
            graphs = json.load(f)
            train_indices, val_indices = train_test_split(
                np.arange(len(graphs)), test_size=test_size, random_state=seed, shuffle=True
            )
            
            train_set = [graphs[i] for i in train_indices]
            val_set = [graphs[i] for i in val_indices]

            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            folds = list(kf.split(train_set))
            
            selected_folds = [(round + i) % n_folds for i in range(train_folds_to_use)]
            train_indices = np.concatenate([folds[i][0] for i in selected_folds])
            train_set = [train_set[i] for i in train_indices]

            train_graphs.extend(train_set)
            val_graphs.extend(val_set)
    return train_set, val_set

class PreloadedGraphDataset(Dataset):
    def __init__(self, graphs, transform=None, pre_transform=None, seed=42):
        self.num_graphs, self.graphs_dicts = len(graphs), graphs 
        super().__init__(None, transform, pre_transform)
        np.random.seed(seed)

    def len(self):
        return self.num_graphs  
    
    def get(self, idx):
        return dictToGraphObject(self.graphs_dicts[idx])


class GraphDataset(Dataset):
    def __init__(self, train_path, transform=None, pre_transform=None):
        self.files = train_path.split(" ")
        self.num_graphs, self.graphs_dicts = self._count_graphs() 
        super().__init__(None, transform, pre_transform)
        np.random.seed(42)

    def len(self):
        return self.num_graphs  
    
    def get(self, idx):
        return dictToGraphObject(self.graphs_dicts[idx])
    
    def pre_filter(self, data):
        if len(self.files > 1):
            return np.random.random() > 0.7
        return True
    
    def _count_graphs(self):
        graphs_dicts = []
        for file in self.files:
            with gzip.open(file, "rt", encoding="utf-8") as f:
                graphs_dicts.extend(json.load(f)) 
        return len(graphs_dicts),graphs_dicts 

def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float) if graph_dict["edge_attr"] else None
    num_nodes = graph_dict["num_nodes"]
    y = torch.tensor(graph_dict["y"][0], dtype=torch.long) if graph_dict["y"] is not None else None
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)






