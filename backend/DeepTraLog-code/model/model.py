"""
@Time ： 2023/2/14 下午6:10
@Auth ： Sek Hiunam
@File ：model.py
@Desc ：
"""
import torch
import torch.nn as nn
from torch_geometric.nn.conv import GatedGraphConv
from torch_geometric.nn import GlobalAttention,AttentionalAggregation
from torch_geometric.nn import global_mean_pool
# from torch_geometric.utils import softmax
from torch.nn.functional import softmax

class GGNN(nn.Module):
    def __init__(self,out_channels,hidden,num_layers,device):
        super(GGNN, self).__init__()
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.hidden = hidden

        self.ggnn = GatedGraphConv(out_channels=out_channels,num_layers=num_layers).to(device)
        self.soft_attention = AttentionalAggregation(gate_nn=nn.Linear(out_channels, 1), nn=None).to(device)
        self.ggnn_2 = GatedGraphConv(out_channels=out_channels,num_layers=num_layers).to(device)
        self.tanh = nn.Tanh().to(device)
        self.linear_1 = torch.nn.Linear(hidden, out_channels).to(device)

        # self.soft_attention = nn.Sequential(
        #     nn.Linear(self.out_channels, self.out_channels),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.out_channels, 1),
        #     nn.Sigmoid()
        # )
        # self.out = nn.Sequential(
        #     nn.Linear(self.out_channels, self.out_channels),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.out_channels, 1),
        #     nn.Tanh(),
        # )

        self.device = device

    def forward(self,data):
        # x,edge_index = x["x"],x["edge_index"]
        x, edge_index,batch = data.x.to(self.device), data.edge_index.to(self.device),data.batch.to(self.device)

        # x_ggnn = torch.nn.functional.pad(x, (0, self.hidden), "constant", 0) # x.shape[-1] == self.out_channel + self.hidden

        x_ggnn = self.ggnn(x,edge_index).to(self.device) # output: node features (|V|,out_channel)

        # Step 3: catenate the GGNN output with the original input
        # x = torch.cat((x_ggnn, x), -1)

        # Step 4: pass this through the attention layer : torch_geometric.nn.aggr.AttentionalAggregation
        output = self.soft_attention(x_ggnn, batch).to(self.device)


        # x_ggnn_2 = self.tanh(x_ggnn_2).to(self.device)
        x_ggnn_2 = self.tanh(self.linear_1(x_ggnn)).to(self.device)
        batch_sum = torch.zeros(batch.max()+1, x_ggnn_2.shape[1]).to(self.device)
        batch_sum.scatter_add_(0, batch.repeat(x_ggnn_2.shape[1], 1).t(), x_ggnn_2)
        output_2 = batch_sum

        output = output * output_2
        output = self.tanh(output).to(self.device)
        # print(output.shape) # [batch_size,output_dim]

        # shall we add a tanh layer?

        return output



