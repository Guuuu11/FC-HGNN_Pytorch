from torch_geometric.nn import ChebConv,TransformerConv
from dataload import dataloader
from opt import *
import torch.nn.functional as F
from torch import nn
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils.subgraph import subgraph
from torch_geometric.nn.dense.diff_pool import dense_diff_pool
from torch_geometric.nn import SAGPooling
opt = OptInit().initialize()


class Brain_connectomic_graph(torch.nn.Module):

    def __init__(self):
        super(Brain_connectomic_graph, self).__init__()
        self._setup()
    def _setup(self):
        self.graph_convolution_l_1 = GCNConv(111,64)
        self.graph_convolution_r_1 = GCNConv(111,64)

        self.graph_convolution_l_2 = GCNConv(64,20)
        self.graph_convolution_r_2 = GCNConv(64,20)

        self.graph_convolution_g_1 = GCNConv(20,20)

        self.pooling_1 = SAGPooling(20, opt.k1)
        self.socre_gcn = ChebConv(20, int(opt.k2*112), K=3, normalization='sym')
        self.pooling_2= dense_diff_pool

        self.weight = nn.Parameter(torch.FloatTensor(64, 20)).to(opt.device)
        self.bns=nn.BatchNorm1d(20).to(opt.device)
        nn.init.xavier_normal_(self.weight)

    def forward(self, data):
        edges, features = data.edge_index, data.x
        edges, features = edges.to(opt.device), features.to(opt.device)

        edge_attr = data.edge_attr
        edge_attr = edge_attr.to(opt.device).to(torch.float32)

        adj=data.adj
        adj=torch.tensor(adj)
        adj=adj.float()
        adj = adj.to(opt.device)

        # --The graph convolutional neural network that integrates intrahemispheric and interhemispheric information (IIH-GCN) of the FC-HGNN.

        # Left and right hemisphere index of fmri data in the ABIDE dataset based on the HO Brain Atlas.
        leftBrain = torch.tensor([  6.,   5.,  55.,   1.,  98.,  71.,  73.,  77.,  63.,  96.,  79.,  15.,
        104.,   4.,  25.,  23.,  41.,  43.,  45.,  17.,  61.,  65.,  59.,  57.,
         86.,  21.,  35.,  37.,  39.,  94., 110.,   3.,  69.,  81.,  84., 100.,
        102., 106.,  47.,  27.,  75.,   2.,  67.,  19.,  49.,  31.,  33., 108.,
         51.,  53.,  88.,  90.,  92.,  29.,   0.])
        rightBrain = torch.tensor([ 13.,  12.,  54.,   8.,  97.,  70.,  72.,  76.,  62.,  95.,  78.,  14.,
        103.,  11.,  24.,  22.,  40.,  42.,  44.,  16.,  60.,  64.,  58.,  56.,
         85.,  20.,  34.,  36.,  38.,  93., 109.,  10.,  68.,  80.,  83.,  99.,
        101., 105.,  46.,  26.,  74.,   9.,  66.,  18.,  48.,  30.,  32., 107.,
         50.,  52.,  87.,  89.,  91.,  28.,   7.])

        # Get a subgraph of the left and right hemispheres of the brain.
        new_left_edges,new_left_edge_attr = subgraph(subset=leftBrain.type(torch.long),edge_index=edges,edge_attr=edge_attr)
        new_right_dges,new_right_edge_attr = subgraph(subset=rightBrain.type(torch.long), edge_index=edges, edge_attr=edge_attr)

        # The intrahemispheric convolution.
        features = F.dropout(features, p=opt.dropout, training=self.training)
        node_features_left = torch.nn.functional.leaky_relu(self.graph_convolution_l_1(features, new_left_edges, new_left_edge_attr))
        node_features_right = torch.nn.functional.leaky_relu(self.graph_convolution_r_1(features, new_right_dges, new_right_edge_attr))
        node_features_1 = torch.zeros(111,64).to(opt.device)
        node_features_1[leftBrain.long(),:] = node_features_left[leftBrain.long(),:]
        node_features_1[rightBrain.long(), :] = node_features_right[rightBrain.long(), :]

        node_features_1 = F.dropout(node_features_1, p=opt.dropout, training=self.training)
        node_features_left = torch.nn.functional.leaky_relu(self.graph_convolution_l_2(node_features_1, new_left_edges, new_left_edge_attr))
        node_features_right = torch.nn.functional.leaky_relu(self.graph_convolution_r_2(node_features_1, new_right_dges, new_right_edge_attr))
        node_features_2 = torch.zeros(111,20).to(opt.device)
        node_features_2[leftBrain.long(),:] = node_features_left[leftBrain.long(),:]
        node_features_2[rightBrain.long(), :] = node_features_right[rightBrain.long(), :]

        # The interhemispheric convolution.
        node_features_2 = torch.nn.functional.leaky_relu(self.graph_convolution_g_1(node_features_2, edges, edge_attr))

        # --The local–global dual-channel pooling (LGP) of the FC-HGNN.
        # The channel 1
        pooling_features, edges, edge_attr,batch, perm, score = self.pooling_1(node_features_2, edges,edge_attr)

        # The channel 2
        ass_matrix=F.softmax(self.socre_gcn(node_features_2,edges),dim=-1)
        H_coarse,assign_matrix, link_loss, ent_loss = self.pooling_2(node_features_2, adj,ass_matrix)

        # The cross-channel convolution.
        inter_channel_adj = features.new_zeros(100,56)
        assign_matrix = torch.squeeze(ass_matrix)
        j = 0
        for i in range(0,110):
            if i in perm:
                inter_channel_adj[j, :]=assign_matrix[i, :]
                j = j+1
        H_coarse = torch.squeeze(H_coarse)
        H1=torch.matmul(inter_channel_adj, H_coarse)
        H2=pooling_features + H1
        graph_embedding = H2.view(1, -1)

        return graph_embedding

class HPG(nn.Module):
    def __init__(self):
        super(HPG, self).__init__()
        self.num_layers = 4
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs1.append(TransformerConv(in_channels=2000,out_channels=20,heads=1))
        self.convs2.append(TransformerConv(in_channels=2000, out_channels=20,heads=1))
        self.bns.append(nn.BatchNorm1d(20))
        self.convs1.append(TransformerConv(in_channels=20, out_channels=20, heads=1))
        self.convs2.append(TransformerConv(in_channels=20, out_channels=20, heads=1))
        self.bns.append(nn.BatchNorm1d(20))
        self.convs1.append(TransformerConv(in_channels=20, out_channels=20, heads=1))
        self.convs2.append(TransformerConv(in_channels=20, out_channels=20, heads=1))
        self.bns.append(nn.BatchNorm1d(20))
        self.convs1.append(TransformerConv(in_channels=20, out_channels= 20, heads=1))
        self.convs2.append(TransformerConv(in_channels=20, out_channels=20, heads=1))
        self.bns.append(nn.BatchNorm1d(20))
        self.out_fc = nn.Linear(80, 2)

        # Set initial weights to speed up the training process.
        self.weights1 = torch.nn.Parameter(torch.empty(4).fill_(0.8))
        self.weights2 = torch.nn.Parameter(torch.empty(4).fill_(0.2))

        self.a = torch.nn.Parameter(torch.Tensor(20, 1))

    def reset_parameters(self):
        for conv in self.convs1:
            conv.reset_parameters()
        for conv in self.convs2:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.out_fc.reset_parameters()
        self.a.reset_parameters()
        torch.nn.init.normal_(self.weights)

    def forward(self, features, same_index,diff_index):
        x = features

        # Graph transformer and information aggregation layers.
        x = F.dropout(x, p=opt.dropout, training=self.training)
        x1 = self.convs1[0](x,same_index)
        x2 = self.convs2[0](x,diff_index)
        weight1 = self.weights1[0] / (self.weights1[0]+ self.weights2[0])
        weight2 = self.weights2[0] / (self.weights1[0] + self.weights2[0])
        x=weight1*x1 + weight2*x2
        x = self.bns[0](x)
        x = F.leaky_relu(x, inplace=True)
        fc = x

        x = F.dropout(x, p=opt.dropout, training=self.training)
        x1 = self.convs1[1](x, same_index)
        x2 = self.convs2[1](x, diff_index)
        weight1 = self.weights1[1] / (self.weights1[1] + self.weights2[1])
        weight2 = self.weights2[1] / (self.weights1[1] + self.weights2[1])
        x = weight1 * x1 + weight2 * x2
        x = self.bns[1](x)
        x = F.leaky_relu(x, inplace=True)
        fc = torch.cat((fc, x), dim=-1)

        x = F.dropout(x, p=opt.dropout, training=self.training)
        x1 = self.convs1[2](x, same_index)
        x2 = self.convs2[2](x, diff_index)
        weight1 = self.weights1[2] / (self.weights1[2] + self.weights2[2])
        weight2 = self.weights2[2] / (self.weights1[2] + self.weights2[2])
        x = weight1 * x1 + weight2 * x2
        x = self.bns[2](x)
        x = F.leaky_relu(x, inplace=True)
        fc = torch.cat((fc, x), dim=-1)

        x = F.dropout(x, p=opt.dropout, training=self.training)
        x1 = self.convs1[3](x, same_index)
        x2 = self.convs2[3](x, diff_index)
        weight1 = self.weights1[3] / (self.weights1[3] + self.weights2[3])
        weight2 = self.weights2[3] / (self.weights1[3] + self.weights2[3])
        x = weight1 * x1 + weight2 * x2
        x = self.bns[3](x)
        x = F.leaky_relu(x, inplace=True)
        fc = torch.cat((fc, x), dim=-1)

        x = self.out_fc(fc)

        return x

class fc_hgnn(torch.nn.Module):

    def __init__(self,nonimg, phonetic_score):
        super(fc_hgnn, self).__init__()
        self.nonimg = nonimg
        self.phonetic_score = phonetic_score
        self._setup()

    def _setup(self):
        self.individual_graph_model = Brain_connectomic_graph()
        self.population_graph_model = HPG()

    def forward(self, graphs):
        dl = dataloader()
        embeddings = []

        # Brain connectomic graph
        for graph in graphs:
            embedding= self.individual_graph_model(graph)
            embeddings.append(embedding)
        embeddings = torch.cat(tuple(embeddings))

        # Heterogeneous population graph (HPG)
        same_index, diff_index = dl.get_inputs(self.nonimg, embeddings, self.phonetic_score)
        same_index = torch.tensor(same_index, dtype=torch.long).to(opt.device)
        diff_index = torch.tensor(diff_index, dtype=torch.long).to(opt.device)

        predictions = self.population_graph_model(embeddings, same_index, diff_index)

        return predictions

class Graph_Transformer(nn.Module):
    def __init__(self, input_dim, output_num,head_num, hidden_dim):
        super(Graph_Transformer, self).__init__()
        #  multi-head self-attention
        self.graph_conv = TransformerConv(input_dim, output_num, head_num)
        self.lin_out = nn.Linear(input_dim, input_dim)

        # feed forward network
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, input_dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        #  multi-head self-attention
        out1 = self.lin_out(self.graph_conv(x, edge_index))

        # feed forward network
        out2 = self.ln1(out1 + x)
        out3 = self.lin2(self.act(self.lin1(out2)))
        out4 = self.ln2(out3 + out2)

        return out4
