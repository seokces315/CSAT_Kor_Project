import torch.nn as nn
import torch.nn.functional as F


# Class for attention pooling
class AttentionPooling(nn.Module):
    # Generator
    def __init__(self, torch_dtype, hidden_dim):
        super().__init__()
        self.attn_fc = nn.Linear(hidden_dim, 1)
        self.attn_fc.to(dtype=torch_dtype)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)

    # Forward
    def forward(self, last_hidden_state, attention_mask):
        # Dtype conversion
        attention_mask = attention_mask.to(dtype=last_hidden_state.dtype)

        # Get attention scores
        attn_scores = self.attn_fc(last_hidden_state)

        # # Regularization
        # attn_scores = self.ReLU(attn_scores)
        # # attn_scores = self.Tanh(attn_scores)
        # attn_scores = self.dropout(attn_scores)

        # Ignore padding tokens
        attention_mask = attention_mask.unsqueeze(-1)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)

        # Softmax over sequence
        attn_weights = F.softmax(attn_scores, dim=1)

        # Pooling
        attn_embeddings = (last_hidden_state * attn_weights).sum(dim=1)

        return attn_embeddings


# Function for mean pooling
def mean_pooling(last_hidden_state, attention_mask):
    # Dtype conversion
    attention_mask = attention_mask.to(dtype=last_hidden_state.dtype)

    # Prepare broadcasting
    expanded_attention_mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state)

    # Zero-out & Sum with given hidden states
    sum_embeddings = (last_hidden_state * expanded_attention_mask).sum(dim=1)

    # Count of non-masked tokens per sample
    sum_mask = expanded_attention_mask.sum(dim=1).clamp(min=1e-9)

    # Mean pooling
    mean_embeddings = sum_embeddings / sum_mask

    return mean_embeddings
