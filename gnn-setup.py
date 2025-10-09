import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Example of creating a molecular graph from RDKit and passing it to GNN
# Create node features (atoms) and edge index (bonds)
node_features = torch.tensor([[1], [1], [1], [1]], dtype=torch.float)  # Placeholder for atom types
edge_index = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)  # Example bonds (edges)

# Create a Data object to represent the graph
data = Data(x=node_features, edge_index=edge_index)

# Simple GCN model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Initialize and train the GNN
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Example forward pass
out = model(data)
print(out)
