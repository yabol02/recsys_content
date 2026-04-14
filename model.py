import torch
import torch.nn as nn

from config import BIZ_EMB_DIM, N_CLASSES, REVIEW_FEAT_DIM, USER_EMB_DIM, USER_META_DIM


def mlp_block(in_dim, out_dim, dropout=0.0):
    layers = [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class RatingModel(nn.Module):
    def __init__(self, proj_dim=128, dropout_proj=0.2):
        super().__init__()

        # Proyecciones a espacio común
        self.user_meta_proj = mlp_block(USER_META_DIM, 64, dropout_proj)
        self.user_emb_proj = mlp_block(USER_EMB_DIM, proj_dim, dropout_proj)
        self.biz_emb_proj = mlp_block(BIZ_EMB_DIM, proj_dim, dropout_proj)
        self.review_proj = mlp_block(REVIEW_FEAT_DIM, 32)

        # Cabeza de fusión: 64 (user_meta) + 128 (user_emb) + 128 (biz_emb) + 128 (hadamard) + 32 = 480
        fusion_dim = 64 + proj_dim + proj_dim + proj_dim + 32
        self.fusion = nn.Sequential(
            mlp_block(fusion_dim, 256, dropout=0.3),
            mlp_block(256, 128, dropout=0.2),
            nn.Linear(128, N_CLASSES - 1),  # CORN: K-1 = 4 salidas
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_meta, user_emb, biz_emb, review_feats):
        u_m = self.user_meta_proj(user_meta)  # (B, 64)
        u_e = self.user_emb_proj(user_emb)  # (B, 128)
        b_e = self.biz_emb_proj(biz_emb)  # (B, 128)
        r = self.review_proj(review_feats)  # (B, 32)

        interaction = u_e * b_e  # Hadamard (B, 128)

        x = torch.cat([u_m, u_e, b_e, interaction, r], dim=1)  # (B, 480)
        return self.fusion(x)  # (B, 4) - logits para CORN
