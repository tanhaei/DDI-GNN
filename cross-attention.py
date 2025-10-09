import torch.nn.functional as F

class CrossAttention(torch.nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim):
        super(CrossAttention, self).__init__()
        self.attn1 = torch.nn.Linear(input_dim1, output_dim)
        self.attn2 = torch.nn.Linear(input_dim2, output_dim)
        self.fc = torch.nn.Linear(output_dim, 1)

    def forward(self, mol_embeddings, text_embeddings):
        # Apply attention mechanism
        mol_proj = self.attn1(mol_embeddings)
        text_proj = self.attn2(text_embeddings)
        combined = mol_proj * text_proj  # Element-wise multiplication
        combined = F.relu(combined)
        prediction = self.fc(combined)
        return prediction

# Example of embedding dimensions
mol_embedding_dim = 16  # Example dimension for molecular embeddings
text_embedding_dim = 768  # Example dimension for textual embeddings

# Random embeddings for illustration (replace with actual embeddings)
mol_embeddings = torch.randn(1, mol_embedding_dim)
text_embeddings = torch.randn(1, text_embedding_dim)

# Initialize the cross-attention model
cross_attention_model = CrossAttention(mol_embedding_dim, text_embedding_dim, 32)

# Get the interaction prediction
interaction_prediction = cross_attention_model(mol_embeddings, text_embeddings)
print(interaction_prediction)
