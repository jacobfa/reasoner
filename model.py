import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import torch_geometric.utils as gutils

class NCN(nn.Module):
    def __init__(self, num_classes, concept_dim=256, num_concepts=1000, pretrained=True):
        super(NCN, self).__init__()
        self.num_classes = num_classes
        self.concept_dim = concept_dim
        self.num_concepts = num_concepts

        # Transformer Backbone
        self.transformer_backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.transformer_backbone.heads = nn.Identity()  # Remove the classification head

        # Concept Bank (Learnable Embeddings)
        self.concept_bank = nn.Embedding(num_concepts, concept_dim)

        # Attention Mechanism to Map Features to Concepts
        self.attention = nn.Sequential(
            nn.Linear(self.transformer_backbone.hidden_dim, concept_dim),
            nn.Tanh(),
            nn.Linear(concept_dim, num_concepts)
        )

        # Concept Reasoning Module (CRM) using GNN
        self.gnn = GATConv(concept_dim, concept_dim, heads=4, concat=False, dropout=0.6)
        self.gnn_norm = nn.BatchNorm1d(concept_dim)
        self.gnn_activation = nn.ReLU()

        # Aggregation Attention
        self.aggregation_attention = nn.Linear(concept_dim, 1)

        # Classifier Module
        self.classifier = nn.Sequential(
            nn.Linear(concept_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize weights for the attention mechanism and classifier
        for m in self.attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # Initialize weights for the aggregation attention
        nn.init.xavier_uniform_(self.aggregation_attention.weight)
        nn.init.zeros_(self.aggregation_attention.bias)

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        # Transformer Backbone
        features = self.transformer_backbone(x)  # Shape: [batch_size, hidden_dim]

        # Concept Extraction Module
        concept_scores = self.attention(features)  # Shape: [batch_size, num_concepts]

        # Apply softmax to get concept probabilities
        concept_probs = F.softmax(concept_scores, dim=1)  # Shape: [batch_size, num_concepts]

        # Get concept embeddings
        concepts = self.concept_bank.weight  # Shape: [num_concepts, concept_dim]

        # Compute node features by weighting concept embeddings with probabilities
        node_features = concepts.unsqueeze(0) * concept_probs.unsqueeze(2)  # Shape: [batch_size, num_concepts, concept_dim]
        node_features = node_features.view(batch_size * self.num_concepts, self.concept_dim)

        # Create edge_index for a simple ring-shaped graph
        edge_index = self.create_ring_edge_index(self.num_concepts, device)
        edge_index = self.batch_edge_index(edge_index, batch_size, self.num_concepts)

        # GNN Forward Pass
        x_gnn = self.gnn(node_features, edge_index)  # Shape: [batch_size * num_concepts, concept_dim]
        x_gnn = self.gnn_norm(x_gnn)
        x_gnn = self.gnn_activation(x_gnn)

        # Reshape to [batch_size, num_concepts, concept_dim]
        x_gnn = x_gnn.view(batch_size, self.num_concepts, self.concept_dim)

        # Aggregation using attention
        attn_weights = self.aggregation_attention(x_gnn)  # Shape: [batch_size, num_concepts, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        refined_embeddings = torch.sum(attn_weights * x_gnn, dim=1)  # Shape: [batch_size, concept_dim]

        # Classifier Module
        logits = self.classifier(refined_embeddings)  # Shape: [batch_size, num_classes]
        return logits

    @staticmethod
    def create_ring_edge_index(num_nodes, device):
        # Create a ring-shaped edge index
        row = torch.arange(num_nodes, device=device)
        col = torch.roll(row, shifts=-1)
        edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
        return edge_index.long()

    @staticmethod
    def batch_edge_index(edge_index, batch_size, num_nodes):
        # Adjust edge_index for batching
        batch_edge_index = []
        for i in range(batch_size):
            offset = i * num_nodes
            batch_edge_index.append(edge_index + offset)
        batch_edge_index = torch.cat(batch_edge_index, dim=1)
        return batch_edge_index

    def freeze_backbone(self):
        for param in self.transformer_backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.transformer_backbone.parameters():
            param.requires_grad = True
