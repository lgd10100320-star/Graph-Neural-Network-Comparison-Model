
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import get_encoder

class GCLModel(nn.Module):

    def __init__(self, encoder_name, input_dim, hidden_dim, projection_dim, num_layers, dropout=0.5, temperature=0.1):
        super(GCLModel, self).__init__()

        self.encoder = get_encoder(encoder_name, input_dim, hidden_dim, num_layers, dropout)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.temperature = temperature

    def forward(self, data1, data2):

        _, z1_pool = self.encoder(data1.x, data1.edge_index, data1.batch)
        _, z2_pool = self.encoder(data2.x, data2.edge_index, data2.batch)

        p1 = self.projection_head(z1_pool)
        p2 = self.projection_head(z2_pool)

        return p1, p2

    def contrastive_loss(self, p1, p2):

        p1 = F.normalize(p1, p=2, dim=1)
        p2 = F.normalize(p2, p=2, dim=1)

        sim_matrix = torch.exp(torch.mm(p1, p2.t()) / self.temperature)

        n_rows, n_cols = sim_matrix.size()
        min_n = min(n_rows, n_cols)

        if min_n == 0:
            return torch.tensor(0.0, device=sim_matrix.device)

        mask = torch.ones_like(sim_matrix)
        idx = torch.arange(min_n, device=sim_matrix.device)
        mask[idx, idx] = 0.0

        pos_sim = sim_matrix[idx, idx]

        neg_sim_all = (sim_matrix * mask).sum(dim=1)
        neg_sim = neg_sim_all[:min_n]

        loss = -torch.log(pos_sim / (pos_sim + neg_sim))
        return loss.mean()

    def get_embedding(self, data):

        self.eval()
        with torch.no_grad():
            _, embedding = self.encoder(data.x, data.edge_index, data.batch)
        return embedding
