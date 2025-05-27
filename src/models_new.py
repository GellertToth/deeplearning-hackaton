import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool
from src.conv import GINConv

class VGAE_MessagePassing(nn.Module):
    def __init__(self, in_channels, edge_attr_dim, hidden_dim, latent_dim):
        super().__init__()
        # Edge network to produce convolution weights from edge attributes        
        self.conv1 = GINConv(hidden_dim)
        self.conv2 = GINConv(hidden_dim)

        self.mu_layer = torch.nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = torch.nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

class VGAE(nn.Module):
    def __init__(self, in_channels, edge_attr_dim, hidden_dim, latent_dim, num_classes):
        super().__init__()
        self.encoder = VGAE_MessagePassing(in_channels, edge_attr_dim, hidden_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.latent_dim = latent_dim

        self.edge_attr_decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_attr_dim)
        )

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
        
        # Graph-level embedding via mean pooling of latent node embeddings
        graph_emb = global_mean_pool(z, batch)
        class_logits= self.classifier(graph_emb)
        return z, mu, logvar, class_logits
    

    def gce_loss(self, logits, targets, q=0.7, weights=None):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=6).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        loss = (1 - pt ** q) / q

        if not weights is None:
            weights = weights[targets]
            loss = loss * weights
        
        return loss.mean()

    def loss(self, z, mu, logvar, class_logits, data, alpha=1, beta=0.1, gamma=0.5, delta=0.3, weights=None):
        classification_loss = F.cross_entropy(class_logits, data.y, weight=weights)

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



