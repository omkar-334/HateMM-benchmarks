import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[: x.size(0)]


class CrossModalAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        query = self.norm1(query)
        attn_output, _ = self.multihead_attn(query, key, value)
        output = query + self.dropout(attn_output)
        output = self.norm2(output)
        ff_output = self.feed_forward(output)
        output = output + self.dropout(ff_output)
        return output


class CrossModalTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([CrossModalAttention(d_model, nhead, dropout) for _ in range(num_layers)])

    def forward(self, query, key, value):
        for layer in self.layers:
            query = layer(query, key, value)
        return query


class SelfAttentionTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout=0.1):
        super().__init__()
        encoder_layers = TransformerEncoderLayer(d_model, num_heads, dim_feedforward=4 * d_model, dropout=dropout, batch_first=False)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x):
        return self.transformer(x)


class MulT(nn.Module):
    def __init__(self, orig_d, d_model=32, num_heads=8, layers_cross=3, layers_self=3, dropout=0.1, out_dim=1):
        super().__init__()

        # Single projection layer for all modalities
        self.proj = nn.Conv1d(orig_d, d_model, kernel_size=1)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # Single cross-modal transformer for all modality pairs
        self.cross_modal_transformer = CrossModalTransformer(d_model, num_heads, layers_cross, dropout)

        # Single self-attention transformer for all modalities
        self.self_attention_transformer = SelfAttentionTransformer(d_model * 2, num_heads, layers_self, dropout)

        # Output layers combined_dim = d_model * 6  # 2 cross-modal outputs per modality
        self.output_layer = nn.Sequential(
            nn.Linear(d_model * 6, d_model * 3),
            nn.LayerNorm(d_model * 3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, out_dim),
        )

    def forward(self, x_l, x_v, x_a):
        if x_l.dim() == 2:
            x_l = x_l.transpose(1, 0).unsqueeze(0)
        if x_v.dim() == 3:
            x_v = x_v.squeeze(1)  # Remove the extra dimension
        if x_v.dim() == 2:
            x_v = x_v.transpose(1, 0).unsqueeze(0)
        if x_a.dim() == 2:
            x_a = x_a.transpose(1, 0).unsqueeze(0)

        # Project each modality using the same projection layer # (batch_size, d_model, seq_len)
        proj_x_l = self.proj(x_l)
        proj_x_v = self.proj(x_v)
        proj_x_a = self.proj(x_a)

        # Transpose to (seq_len, batch_size, d_model) for transformer input
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)

        # Add positional encoding
        proj_x_l = self.positional_encoding(proj_x_l)
        proj_x_v = self.positional_encoding(proj_x_v)
        proj_x_a = self.positional_encoding(proj_x_a)

        # Cross-modal transformer # (seq_len, batch_size, d_model * 2)
        h_l = torch.cat(
            [
                self.cross_modal_transformer(proj_x_l, proj_x_v, proj_x_v),
                self.cross_modal_transformer(proj_x_l, proj_x_a, proj_x_a),
            ],
            dim=-1,
        )

        h_v = torch.cat(
            [
                self.cross_modal_transformer(proj_x_v, proj_x_l, proj_x_l),
                self.cross_modal_transformer(proj_x_v, proj_x_a, proj_x_a),
            ],
            dim=-1,
        )

        h_a = torch.cat(
            [
                self.cross_modal_transformer(proj_x_a, proj_x_l, proj_x_l),
                self.cross_modal_transformer(proj_x_a, proj_x_v, proj_x_v),
            ],
            dim=-1,
        )
        # Self-attention transformer
        h_l_final = self.self_attention_transformer(h_l)
        h_v_final = self.self_attention_transformer(h_v)
        h_a_final = self.self_attention_transformer(h_a)

        # Concatenate final representations # (batch_size, d_model * 6)
        last_hs = torch.cat([h_l_final.squeeze(1), h_v_final.squeeze(1), h_a_final.squeeze(1)], dim=-1)

        output = self.output_layer(last_hs)
        return output
