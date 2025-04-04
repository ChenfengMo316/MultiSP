import torch
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn as nn


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output=torch.mm(adj, support)
        return output
    

class GAN_Discriminator(torch.nn.Module):
    def __init__(self, int_dim):
        super(GAN_Discriminator, self).__init__()
        self.nn=nn.Sequential(nn.Linear(int_dim, 2)
                              )
    def forward(self, x):
        z = self.nn(x)
        output=z
        return output
    

