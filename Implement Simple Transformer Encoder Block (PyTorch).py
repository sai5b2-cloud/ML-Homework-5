import torch
import torch.nn as nn

# -------------------------
# Multi-Head Self-Attention
# -------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=128, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # Linear layers for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Final projection
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # Compute Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads → (batch, heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot Product Attention
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(weights, V)

        # Combine heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, d_model)

        # Final linear layer
        out = self.W_o(attention_output)
        return out

# -------------------------
# Feed Forward Network (FFN)
# -------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model=128, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# -------------------------
# Transformer Encoder Block
# -------------------------
class SimpleTransformerEncoder(nn.Module):
    def __init__(self, d_model=128, num_heads=8):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Multi-Head Attention + Add & Norm
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)  # residual connection

        # Feed Forward + Add & Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)  # residual connection

        return x

# -------------------------
# Testing the Model
# -------------------------
if __name__ == "__main__":
    batch_size = 32
    seq_len = 10
    d_model = 128

    x = torch.randn(batch_size, seq_len, d_model)
    encoder = SimpleTransformerEncoder(d_model=128, num_heads=8)

    output = encoder(x)
    print("Output shape:", output.shape)
