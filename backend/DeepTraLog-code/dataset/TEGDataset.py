"""
@Time ： 2023/2/13 下午5:13
@Auth ： Sek Hiunam
@File ：TEGDataset.py
@Desc ：
"""
from collections import defaultdict

import torch
# from torch.utils.data import Dataset
from torch_geometric.data import Dataset

class TEGDataset(Dataset):
    def __init__(self, tegs, id2emb=None):
        self.tegs = tegs
        self.id2emb = id2emb

    def __len__(self):
        return len(self.tegs)

    def __getitem__(self, item):
        graph = self.tegs[item]
        data  = graph["x"]
        edge_index = graph["edge_index"]
        return data,edge_index

    def collate_fn(self, batch):

        output = defaultdict(list)
        for graph in batch:
            x =  graph[0]
            edge_index = graph[1]

            output["x"].append(x)
            output["edge_index"].append(edge_index)

        output["x"] = torch.tensor(output["x"], dtype=torch.float)
        output["edge_index"] = torch.tensor(output["edge_index"], dtype=torch.long)


        return output
