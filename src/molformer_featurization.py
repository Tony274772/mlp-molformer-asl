"""Frozen MoLFormer featurization for API/excipient SMILES strings."""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class MolFormerFeaturizer(nn.Module):
    """Tokenize SMILES and return frozen MoLFormer sequence and pooler outputs."""

    def __init__(self, model_path: str = "models/molformer-xl-both-10pct"):
        super().__init__()
        self.model_path = model_path
        self.model = AutoModel.from_pretrained(
            model_path,
            deterministic_eval=True,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)
        self.cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def _empty_features(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device
        hidden = torch.zeros(0, self.dim, device=device)
        pooler = torch.zeros(self.dim, device=device)
        mask = torch.ones(0, dtype=torch.bool, device=device)
        num_tokens = torch.tensor(0, dtype=torch.long, device=device)
        return hidden, pooler, mask, num_tokens

    def _run_uncached(self, smiles: List[str]) -> None:
        if not smiles:
            return
        encoded = self.tokenizer(smiles, padding=True, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)

        if not hasattr(outputs, "pooler_output") or outputs.pooler_output is None:
            raise RuntimeError("MoLFormer output did not include pooler_output.")

        attention_mask = encoded["attention_mask"]
        key_padding_mask = attention_mask == 0
        num_tokens = attention_mask.sum(dim=1)

        for idx, smiles_text in enumerate(smiles):
            valid_len = int(num_tokens[idx].item())
            self.cache[smiles_text] = (
                outputs.last_hidden_state[idx, :valid_len].detach(),
                outputs.pooler_output[idx].detach(),
                key_padding_mask[idx, :valid_len].detach(),
                num_tokens[idx].detach(),
            )

    def forward(
        self, smiles_list: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = ["" if smi is None else str(smi).strip() for smi in smiles_list]

        for smi in normalized:
            if smi == "" and smi not in self.cache:
                self.cache[smi] = self._empty_features()

        uncached = []
        seen = set()
        for smi in normalized:
            if smi and smi not in self.cache and smi not in seen:
                uncached.append(smi)
                seen.add(smi)
        self._run_uncached(uncached)

        features = [self.cache[smi] for smi in normalized]
        batch_size = len(features)
        max_len = max((feat[0].size(0) for feat in features), default=0)
        max_len = max(1, max_len)

        device = self.device
        hidden = torch.zeros(batch_size, max_len, self.dim, device=device)
        pooler = torch.zeros(batch_size, self.dim, device=device)
        key_padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)
        num_tokens = torch.zeros(batch_size, dtype=torch.long, device=device)

        for row, (row_hidden, row_pooler, row_mask, row_num_tokens) in enumerate(features):
            length = row_hidden.size(0)
            if length:
                hidden[row, :length] = row_hidden.to(device)
                key_padding_mask[row, :length] = row_mask.to(device)
            pooler[row] = row_pooler.to(device)
            num_tokens[row] = row_num_tokens.to(device)

        return hidden, pooler, key_padding_mask, num_tokens
