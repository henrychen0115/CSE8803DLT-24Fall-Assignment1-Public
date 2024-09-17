import torch
import torch.nn as nn

d_model = 64
n_head = 8
dropout = 0.1

batch_size = 2
seq_len = 16


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, d_model=64, n_head=8, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.residual_dropout = nn.Dropout(dropout)

        self.n_head = n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the multi-head masked self-attention layer.
        You should not use network modules other than what defined in the __init__ function.

        Input & output shape: (batch_size, sequence_length, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        head_dim = d_model // self.n_head
        # Reshape for multi-head attention
            # Project input to query, key, and value
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape and transpose for multi-head attention
        q = q.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)

        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_scores.masked_fill_(causal_mask, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn_probs, v)

        # Reshape and concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final projection and dropout
        output = self.o_proj(output)
        output = self.residual_dropout(output)

        return output


# causal_self_attention = CausalSelfAttention(d_model=d_model, n_head=n_head, dropout=dropout)

# Load the model
causal_self_attention = CausalSelfAttention(d_model=d_model, n_head=n_head, dropout=dropout)
causal_self_attention.load_state_dict(torch.load("causal_self_attention.pt"))


# Test the model
# shape: (batch_size, seq_len, d_model)
x = torch.load("x.pt")

y = causal_self_attention(x)
y_expected = torch.load("y.pt")

assert y.shape == y_expected.shape, f"Expected shape: {y_expected.shape}, but got: {y.shape}"
assert torch.sum(torch.abs(y - y_expected) < 1e-5) > 0.78 * batch_size * seq_len * d_model, "The output is incorrect."

print("The output is correct.")
