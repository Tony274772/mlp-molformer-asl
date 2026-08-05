"""MoLFormer-based API/excipient compatibility model."""

import torch
import torch.nn as nn


def _projection(in_dim, hidden_dim, out_dim, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
    )


class APIExcipientModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.molformer_dim
        proj_dim = config.proj_dim

        self.exc_placeholder = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.01)
        self.exc_global_placeholder = nn.Parameter(torch.randn(1, proj_dim) * 0.01)

        self.api_proj = _projection(dim, 256, proj_dim, config.proj_dropout)
        self.exc_proj = _projection(dim, 256, proj_dim, config.proj_dropout)

        self.attn_exc_to_api = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=config.num_heads,
            dropout=config.attn_dropout,
            batch_first=True,
        )
        self.norm_exc_to_api = nn.LayerNorm(proj_dim)

        self.attn_api_to_exc = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=config.num_heads,
            dropout=config.attn_dropout,
            batch_first=True,
        )
        self.norm_api_to_exc = nn.LayerNorm(proj_dim)

        if config.use_descriptors:
            self.api_desc_proj = _projection(
                config.num_descriptors, 32, config.desc_proj_dim, config.desc_dropout
            )
            self.exc_desc_proj = _projection(
                config.num_descriptors, 32, config.desc_proj_dim, config.desc_dropout
            )

        dim_api = proj_dim
        dim_exc = proj_dim + 1
        if config.use_descriptors:
            dim_api = proj_dim + config.desc_proj_dim
            dim_exc = proj_dim + config.desc_proj_dim + 1
        self.dim_api = dim_api

        pair_dim = dim_api + dim_exc + dim_api + dim_api

        self.classifier = nn.Sequential(
            nn.Linear(pair_dim, config.clf_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.clf_dropout_1),
            nn.Linear(config.clf_hidden_dim, config.clf_hidden_dim_2),
            nn.GELU(),
            nn.Dropout(config.clf_dropout_2),
            nn.Linear(config.clf_hidden_dim_2, 1),
        )

    def _prepend_cls(self, token_sequence, pooler_output, mask, projection):
        token_projection = projection(token_sequence)
        cls_projection = projection(pooler_output).unsqueeze(1)
        sequence = torch.cat([cls_projection, token_projection], dim=1)
        cls_mask = torch.zeros(mask.size(0), 1, dtype=torch.bool, device=mask.device)
        extended_mask = torch.cat([cls_mask, mask], dim=1)
        return sequence, extended_mask

    def forward(self, batch):
        api_seq, api_mask = self._prepend_cls(
            batch["api_tokens"],
            batch["api_global"],
            batch["api_mask"],
            self.api_proj,
        )
        exc_seq, exc_mask = self._prepend_cls(
            batch["exc_tokens"],
            batch["exc_global"],
            batch["exc_mask"],
            self.exc_proj,
        )

        exc_avail = batch["exc_available"].unsqueeze(1)
        batch_size = exc_seq.size(0)
        missing_rows = exc_avail == 0.0

        exc_seq = torch.where(missing_rows.view(batch_size, 1, 1), torch.zeros_like(exc_seq), exc_seq)
        exc_seq[:, :1, :] = torch.where(
            missing_rows.view(batch_size, 1, 1),
            self.exc_global_placeholder.expand(batch_size, -1).unsqueeze(1),
            exc_seq[:, :1, :],
        )
        if exc_seq.size(1) > 1:
            exc_seq[:, 1:2, :] = torch.where(
                missing_rows.view(batch_size, 1, 1),
                self.exc_placeholder.expand(batch_size, -1, -1),
                exc_seq[:, 1:2, :],
            )
            placeholder_mask = torch.ones_like(exc_mask)
            placeholder_mask[:, 0] = False
            placeholder_mask[:, 1] = False
            exc_mask = torch.where(missing_rows.expand_as(exc_mask), placeholder_mask, exc_mask)

        refined_exc, _ = self.attn_exc_to_api(
            query=exc_seq,
            key=api_seq,
            value=api_seq,
            key_padding_mask=api_mask,
        )
        refined_exc = torch.nan_to_num(refined_exc, 0.0)
        exc_out = self.norm_exc_to_api(exc_seq + refined_exc)

        refined_api, _ = self.attn_api_to_exc(
            query=api_seq,
            key=exc_seq,
            value=exc_seq,
            key_padding_mask=exc_mask,
        )
        refined_api = torch.nan_to_num(refined_api, 0.0)
        api_out = self.norm_api_to_exc(api_seq + refined_api)

        h_api_struct = api_out[:, 0, :]
        h_exc_struct = exc_out[:, 0, :]

        if self.config.use_descriptors:
            d_api = self.api_desc_proj(batch["api_desc"])
            d_exc = self.exc_desc_proj(batch["exc_desc"])
            h_api = torch.cat([h_api_struct, d_api], dim=-1)
            h_exc = torch.cat([h_exc_struct, d_exc, exc_avail], dim=-1)
        else:
            h_api = h_api_struct
            h_exc = torch.cat([h_exc_struct, exc_avail], dim=-1)

        h_exc_core = h_exc[:, :self.dim_api]
        interaction = h_api * h_exc_core
        difference = torch.abs(h_api - h_exc_core)

        pair_vec = torch.cat([h_api, h_exc, interaction, difference], dim=-1)
        logits = self.classifier(pair_vec)
        return logits.squeeze(1)
