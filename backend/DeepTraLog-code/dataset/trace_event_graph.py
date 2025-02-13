"""
@Time ： 2023/2/13 下午5:17
@Auth ： Sek Hiunam
@File ：trace_event_graph.py
@Desc ：
"""
from datetime import datetime

import pandas as pd
import numpy as np
import pickle
import torch
import json

from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import os
import pickle as plk
from collections import defaultdict
from sklearn.decomposition import PCA


event_type = {"LogEvent": 0, "ServerRequest": 1, "ServerResponse": 2, "ClientRequest": 3, "ClientResponse": 4,
              "Consumer": 5, "Producer": 6}
edge_type={'Sequence':0,'SynRequest':1,'SynResponse':2,'AsynRequest':3}

class Node:
    def __init__(self, node_id,node_info):
        self.data = {'node_id':node_id,'node_info':node_info}

    def get(self):
        return self.data

class TEGraph:
    def __init__(self,event_type,edge_type,graph_corpus,log_corpus,span_corpus,out_dir=None,embed_dir=None):
        self.event_type = event_type
        self.edge_type = edge_type

        self.log_corpus = log_corpus
        self.span_corpus = span_corpus
        self.graph_corpus = graph_corpus

        self.out_dir = out_dir
        self.embed_dir = embed_dir

    def load_graphs_from_jsons(self,filepath,is_Train=True,test_ratio=0.3):
        jsonList = []

        i = 0
        with open(filepath) as f:
            for graph in f:
                if i % 5000 == 0:
                    print("graph loading:" + str(i))
                i += 1
                teg = json.loads(graph)
                jsonList.append(teg)

        normal,abnormal = [],[]

        # node_info = []
        # print(np.array(node_info).shape) # [num_nodes, num_node_features] ,(1513, 300)
        # pca = PCA(n_components=embed_dim)
        # node_info = pca.fit_transform(node_info)
        # tem2Emed = torch.tensor(node_info,dtype=torch.float) # torch.Size([1513, 10])

        i = 0
        for data in jsonList:
            # x (torch.Tensor, optional) – Node feature matrix with shape [num_nodes, num_node_features]
            # edge_index (LongTensor, optional) – Graph connectivity in COO format with shape [2, num_edges]
            # edge_attr (torch.Tensor, optional) – Edge feature matrix with shape [num_edges, num_edge_features]
            graph = Data(x=torch.tensor(data['node_info']),
                         edge_index=torch.tensor(data['edge_index']).t().contiguous(),
                         edge_attr=torch.tensor(data['edge_attr']).reshape(-1, 1))
            if not data['trace_bool']: # trace_bool==True means error
                normal.append(graph)
            else:
                abnormal.append(graph)

        train_normal, test_normal = train_test_split(normal, train_size=1-test_ratio, test_size=test_ratio, random_state=1234)


        if is_Train: # just adopt normal samples for training
            return train_normal, test_normal, abnormal
        else:
            return test_normal, abnormal