import torch
import torch.nn as nn

class RegressionMLP(nn.Module):
    def __init__(self, in_features=8, hidden1=32, hidden2=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class CompactMNISTMLP(nn.Module):
    def __init__(self, in_features=196, hidden_dim=16, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        if x.dim() > 2:
            x = nn.functional.adaptive_avg_pool2d(x, (14, 14)).view(x.size(0), -1)
        return self.net(x)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class GrokkingTransformer(nn.Module):
    """
    1-Layer Transformer for modular arithmetic reproduction (Nanda et al. 2023 / Power et al. 2022).
    - Sequence length: 3 ([a, b, =])
    - d_model: 128, n_heads: 4 (d_head: 32)
    - d_mlp: 512, ReLU activation
    - No LayerNorm (clean linear representation)
    - Causal attention mask
    - Unembedding: d_model -> p on the final '=' position
    """
    def __init__(self, p=113, d_model=128, n_heads=4, d_mlp=512):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_mlp = d_mlp

        # Embeddings
        self.token_embed = nn.Embedding(p + 1, d_model) # p tokens for numbers, 1 for '='
        self.pos_embed = nn.Parameter(torch.randn(3, d_model) / (d_model ** 0.5))

        # Self-Attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # MLP
        self.mlp_in = nn.Linear(d_model, d_mlp)
        self.relu = nn.ReLU()
        self.mlp_out = nn.Linear(d_mlp, d_model)

        # Unembedding Head
        self.unembed = nn.Linear(d_model, p, bias=False)

        # Register causal mask for seq_len=3
        mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        # x: (batch_size, 3)
        b, seq_len = x.shape
        h = self.token_embed(x) + self.pos_embed[:seq_len] # (b, seq_len, d_model)

        # Multi-Head Attention
        q = self.q_proj(h).view(b, seq_len, self.n_heads, self.d_head).transpose(1, 2) # (b, h, seq, d_h)
        k = self.k_proj(h).view(b, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(h).view(b, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5) # (b, h, seq, seq)
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v) # (b, h, seq, d_h)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)

        # Residual connection (no LayerNorm)
        h = h + attn_out

        # MLP block + Residual connection
        mlp_act = self.relu(self.mlp_in(h))
        mlp_out = self.mlp_out(mlp_act)
        h = h + mlp_out

        # Logits at the final position '=' (index -1)
        final_h = h[:, -1, :] # (b, d_model)
        logits = self.unembed(final_h) # (b, p)
        return logits

def eval_grokking_tensors(model, x_tensor, y_tensor, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(x_tensor)
        loss = criterion(logits, y_tensor).item()
        preds = logits.argmax(dim=-1)
        acc = 100.0 * (preds == y_tensor).float().mean().item()
    return loss, acc

def get_flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])

def set_flat_params(model, flat_params):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset + numel].view_as(p))
        offset += numel

def eval_regression_mse(model, x_tensor, y_tensor, criterion):
    model.eval()
    with torch.no_grad():
        preds = model(x_tensor)
        return criterion(preds, y_tensor).item()

def eval_mnist_tensors(model, x_tensor, y_tensor, criterion):
    model.eval()
    with torch.no_grad():
        out = model(x_tensor)
        loss = criterion(out, y_tensor).item()
        preds = out.argmax(dim=1)
        acc = 100.0 * (preds == y_tensor).sum().item() / y_tensor.size(0)
    return loss, acc
