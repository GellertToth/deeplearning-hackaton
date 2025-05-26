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

class GraphDatasetDownsample(Dataset):
    def __init__(self, train_path, round, subset_ratio=0.6, transform=None, pre_transform=None, seed=42):
        self.files = train_path.split(" ")
        self.round = round
        self.subset_ratio = subset_ratio
        np.random.seed(seed)
        self.num_graphs, self.graphs_dicts = self._count_graphs() 
        super().__init__(None, transform, pre_transform)

    def len(self):
        return self.num_graphs  
    
    def get(self, idx):
        return dictToGraphObject(self.graphs_dicts[idx])
    
    def _count_graphs(self):
        graphs_dicts = []
        for file in self.files:
            with gzip.open(file, "rt", encoding="utf-8") as f:
                graphs = json.load(f) 
                indices = np.random.permutation(len(graphs))
                leave_out = int(len(graphs) * (1-self.subset_ratio))
                start, end = leave_out * self.round, leave_out * self.round + int(len(graphs) * self.subset_ratio)
                keep = [i%len(graphs) for i in range(start, end)]
                print(len(keep), len(graphs), start, end)
                graphs = [graphs[indices[i]] for i in keep]
                graphs_dicts.extend(graphs)
        return len(graphs_dicts),graphs_dicts 

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






