import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.aggr import AttentionalAggregation
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
import torch.nn as nn
from torch_geometric.nn import NNConv, global_mean_pool
from src.conv import GINConv


from src.conv import GNN_node, GNN_node_Virtualnode

class GNN(torch.nn.Module):

    def __init__(self, num_class, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = AttentionalAggregation(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)
        return h_graph
    
class MLP(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(torch.nn.ReLU())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) 

class CompleteModel(torch.nn.Module):
    def __init__(self, gnn, classifier):
        super(CompleteModel, self).__init__()
        self.gnn = gnn
        self.classifier = classifier

    def forward(self, x):
        h = self.gnn(x)
        h = F.normalize(h, p=2.0, dim=1)
        return self.classifier(h)

class NoisyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, p_noisy):
        super().__init__()
        self.p = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (1 - torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        return (losses * weights).mean()
  
class VGAEEncoder(nn.Module):
    def __init__(self, in_channels, edge_attr_dim, hidden_dim, latent_dim, drop_ratio):
        super().__init__()
        # Edge network to produce convolution weights from edge attributes        
        self.conv1 = GINConv(hidden_dim)
        self.conv2 = GINConv(hidden_dim)

        self.mu_layer = torch.nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = torch.nn.Linear(hidden_dim, latent_dim)
        self.dropout = torch.nn.Dropout(drop_ratio)

    def forward(self, x, edge_index, edge_attr):
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr), 0.1)
        x = self.dropout(x)
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr), 0.1)
        x = self.dropout(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

class VGAE(nn.Module):
    def __init__(self, in_channels, edge_attr_dim, hidden_dim, latent_dim, num_classes, noise_prob=0.2, drop_ratio=0.2):
        super().__init__()
        self.encoder = VGAEEncoder(in_channels, edge_attr_dim, hidden_dim, latent_dim, drop_ratio)
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.latent_dim = latent_dim

        self.edge_attr_decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, edge_attr_dim)
        )
        self.drop = nn.Dropout(drop_ratio)
        self.criterion = NoisyCrossEntropyLoss(p_noisy=noise_prob)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, edge_index):
        adj_pred = torch.sigmoid(torch.mm(z, z.t()))
        
        edge_feat_input = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        edge_attr_pred = torch.sigmoid(self.edge_attr_decoder(edge_feat_input))
        
        return adj_pred, edge_attr_pred

    def forward(self, data, inference=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = x.reshape((-1, 1))
        mu, logvar = self.encoder(x, edge_index, edge_attr)
        if not inference:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        # Graph-level embedding via mean pooling of latent node embeddings
        graph_emb = global_mean_pool(z, batch)
        graph_emb = self.drop(graph_emb)
        
        class_logits= self.classifier(graph_emb)
        return z, mu, logvar, class_logits
    

    # def gce_loss(self, logits, targets, q=0.7, weights=None):
    #     probs = F.softmax(logits, dim=1)
    #     targets_one_hot = F.one_hot(targets, num_classes=6).float()
    #     pt = (probs * targets_one_hot).sum(dim=1)
    #     loss = (1 - pt ** q) / q

    #     if not weights is None:
    #         weights = weights[targets]
    #         loss = loss * weights
        
    #     return loss.mean()

    def loss(self, z, mu, logvar, class_logits, data, alpha=1, beta=0.02, gamma=0.1, delta=0.1, weights=None):
        classification_loss = self.criterion(class_logits, data.y)

        adj_pred, edge_attr_pred = self.decode(z, data.edge_index)
        adj_true = torch.zeros_like(adj_pred)
        adj_true[data.edge_index[0], data.edge_index[1]]

        adj_loss = F.binary_cross_entropy(adj_pred, adj_true)
        edge_attr_loss = F.mse_loss(edge_attr_pred, data.edge_attr)

        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        loss = (
            alpha * classification_loss +
            beta * adj_loss +
            gamma * edge_attr_loss +
            delta * kl_loss
        )

        return loss


class GNNEncoderDecoder(torch.nn.Module):
    def __init__(self, num_class, num_layer = 5, emb_dim = 300, latent_dim = 16,
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean", noise_prob = 0.2):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNNEncoderDecoder, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")
        
        self.node_emb = torch.nn.Linear(emb_dim, latent_dim)

        self.edge_attr_decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim * 2, latent_dim),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(latent_dim, 7)
        )

        self.classifier = torch.nn.Linear(emb_dim, 6)

        self.criterion = NoisyCrossEntropyLoss(noise_prob)

    def decode(self, z, edge_index):
        adj_pred = torch.sigmoid(torch.mm(z, z.t()))
        
        edge_feat_input = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        edge_attr_pred = torch.sigmoid(self.edge_attr_decoder(edge_feat_input))
        
        return adj_pred, edge_attr_pred

    def forward(self, batched_data):
        h_node = torch.nn.functional.leaky_relu(self.gnn_node(batched_data), 0.1)
        h_graph = self.pool(h_node, batched_data.batch)

        node_emb = self.node_emb(h_node)
        return node_emb, self.classifier(h_graph)
    
    def recon_loss(self, z, data, alpha=0.1, beta=1, gamma=0.1):
        adj_pred, edge_attr_pred = self.decode(z, data.edge_index)
        adj_true = torch.zeros_like(adj_pred)
        adj_true[data.edge_index[0], data.edge_index[1]]

        adj_loss = F.binary_cross_entropy(adj_pred, adj_true)
        edge_attr_loss = F.mse_loss(edge_attr_pred, data.edge_attr)

        # kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return alpha * adj_loss + beta * edge_attr_loss

    def loss(self, node_emb, class_logits, data):
        classification_loss = self.criterion(class_logits, data.y)
        recon_loss = self.recon_loss(node_emb, data)

        loss = 0.1 * recon_loss + classification_loss
        return loss
    
class EnsembleModel(nn.Module):
    def __init__(self, models: list, weights, device):
        super().__init__()
        self.models = nn.ModuleList(models)
        # Trainable weights initialized equally
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.weights = (self.weights / self.weights.sum(dim=0)).to(device)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        dim = len(outputs[0])
        outputs = [el[-1] for el in outputs]
        outputs = torch.stack(outputs)
        weighted_avg = (outputs * self.weights.view(-1, 1, 1)).sum(dim=0)
        if dim == 4:
            return None, None, None, weighted_avg
        return None, weighted_avg
    