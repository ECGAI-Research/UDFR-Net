import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDecoder_9216(nn.Module):
    def __init__(self, out_seq_len=9216, feature_dim=1152, hidden_dims=[768, 512, 256], attn_window=32):
        super().__init__()
        self.seq_len = out_seq_len
        self.feature_dim = feature_dim
        self.attn_window = attn_window

        # ---------------- Projection ----------------
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dims[0]),
            nn.GELU(),
            nn.LayerNorm(hidden_dims[0])
        )

        # ---------------- Hidden blocks ----------------
        self.hidden_blocks = nn.ModuleList()
        in_dim = hidden_dims[0]
        for hdim in hidden_dims[1:]:
            self.hidden_blocks.append(
                nn.Sequential(
                    nn.Linear(in_dim, hdim),
                    nn.GELU(),
                    nn.LayerNorm(hdim)
                )
            )
            in_dim = hdim

        # ---------------- Final output ----------------
        self.output_layer = nn.Linear(in_dim, feature_dim)

        # ---------------- Channel attention ----------------
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feature_dim, feature_dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim // 8, feature_dim, 1),
            nn.Sigmoid()
        )

    def local_attention(self, x):
        """
        Lightweight local attention.
        x: (B, seq_len, feature_dim)
        """
        B, N, C = x.shape
        ws = self.attn_window
        out = torch.zeros_like(x)

        for start in range(0, N, ws):
            end = min(start + ws, N)
            x_chunk = x[:, start:end, :]
            q = x_chunk
            k = x_chunk
            v = x_chunk

            attn_scores = torch.bmm(q, k.transpose(1, 2)) / (C ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)
            out[:, start:end, :] = torch.bmm(attn_weights, v)

        return out

    def forward(self, x):
        """
        x: (B, seq_len, feature_dim)
        returns: (B, seq_len, feature_dim)
        """
        # Projection
        x = self.projection(x)
        # Hidden blocks
        for block in self.hidden_blocks:
            x = block(x)
        # Output
        x = self.output_layer(x)

        # Channel attention
        x_att = x.permute(0, 2, 1)  # (B, feature_dim, seq_len)
        ch_att = self.channel_attention(x_att)
        x = x * ch_att.permute(0, 2, 1)

        # Lightweight local attention
        x = self.local_attention(x)

        # Crop if needed
        if x.shape[1] > self.seq_len:
            x = x[:, :self.seq_len, :]

        return x
