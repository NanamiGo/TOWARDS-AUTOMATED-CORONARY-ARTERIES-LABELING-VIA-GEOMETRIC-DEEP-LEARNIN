
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv,GINConv
from pygcn.layers import GraphConvolution

class GCNConv(torch.nn.Module):
    def __init__(self, feature, hidden, classes):
        super(GraphSAGE, self).__init__()
        self.GCN1 = GCNConv(feature, hidden)
        self.GCN2 = GCNConv(hidden, classes)

    def forward(self, features, edges):
        features = self.GCN1(features, edges)
        features = F.relu(features)
        features = F.dropout(features, training=self.training)
        features = self.GCN2(features, edges)
        return F.log_softmax(features, dim=1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, feature, hidden, classes):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(feature, hidden)
        self.sage2 = SAGEConv(hidden, classes)

    def forward(self, features, edges):
        features = self.sage1(features, edges)
        features = F.relu(features)
        features = F.dropout(features, training=self.training)
        features = self.sage2(features, edges)
        return F.log_softmax(features, dim=1)

class GIN(torch.nn.Module):
    def __init__(self, feature, hidden, classes):
        super(GIN, self).__init__()
        self.GIN1 = GINConv(feature, hidden)
        self.GIN2 = GINConv(hidden, classes)

    def forward(self, features, edges):
        features = self.GIN1(features, edges)
        features = F.relu(features)
        features = F.dropout(features, training=self.training)
        features = self.GIN2(features, edges)
        return F.log_softmax(features, dim=1)

class GATNet(torch.nn.Module):
    def __init__(self, feature, hidden, classes):
        super(GATNet, self).__init__()
        self.GAT1 = GATConv(feature, hidden)
        self.GAT2 = GATConv(hidden, classes)

    def forward(self, features, edges):
        features = self.GAT1(features, edges)
        features = F.relu(features)
        features = F.dropout(features, training=self.training)
        features = self.GAT2(features, edges)
        return F.log_softmax(features, dim=1)

