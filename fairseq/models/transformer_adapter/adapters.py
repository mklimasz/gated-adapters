import logging
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn

from fairseq import utils
from fairseq.logging import metrics
from fairseq.modules import FairseqDropout, LayerNorm
from fairseq.modules.quant_noise import quant_noise

logger = logging.getLogger(__name__)


class Adapter(nn.Module):
    def __init__(self, cfg, red_fac=2):
        super(Adapter, self).__init__()
        self.cfg = cfg
        self.embed_dim = cfg.encoder_embed_dim
        self.quant_noise = getattr(cfg, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(cfg, "quant_noise_pq_block_size", 8) or 8
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(cfg, "activation_fn", "relu") or "relu"
        )
        self.fc1 = quant_noise(
            nn.Linear(self.embed_dim, int(self.embed_dim // red_fac)),
            p=self.quant_noise,
            block_size=self.quant_noise_block_size,
        )
        self.fc2 = quant_noise(
            nn.Linear(int(self.embed_dim // red_fac), self.embed_dim),
            p=self.quant_noise,
            block_size=self.quant_noise_block_size,
        )
        activation_dropout_p = getattr(cfg, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = getattr(cfg, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        if not hasattr(self.cfg, "adapter_dropout") or self.cfg.adapter_dropout:
            x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x


def adapters_factory(cfg, final_layer_norm: nn.LayerNorm, encoder: bool, gating: nn.Module):
    adapter_type = cfg.adapter_type
    if adapter_type == "base":
        return DiscreteAdapters(cfg, final_layer_norm)
    elif adapter_type == "gated":
        return GatedAdapters(cfg, final_layer_norm, encoder, gating)
    else:
        raise ValueError(f"Unknown adapter type `{adapter_type}`.")


class BaseAdapters(nn.Module):

    def __init__(self, cfg, final_layer_norm: nn.LayerNorm):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.encoder_embed_dim
        export = getattr(cfg, "export", False)
        if cfg.adapter_layer_norm:
            self.adapter_layer_norm = LayerNorm(self.embed_dim, export=export)
        if cfg.adapter_reuse_layer_norm:
            self.final_layer_norm = final_layer_norm
        self.adapter_modules = nn.ModuleDict(dict())
        if hasattr(self.cfg, "bottleneck"):
            bottleneck = self.cfg.bottleneck
        else:
            bottleneck = 2

        self.adapter_idx2name = {}
        for adapter_idx, adapter_name in enumerate(cfg.adapter_names):
            self.adapter_idx2name[adapter_idx] = adapter_name
            if cfg.adapter_layer_norm_per_adapter:
                self.adapter_modules[adapter_name] = nn.Sequential(
                    LayerNorm(self.embed_dim, export=export),
                    Adapter(cfg, red_fac=bottleneck)
                )
            else:
                self.adapter_modules[adapter_name] = Adapter(cfg, red_fac=bottleneck)

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )

    def residual_connection(self, x, residual):
        return residual + x

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError()


class GatedAdapters(BaseAdapters):

    def __init__(self, cfg, final_layer_norm: nn.LayerNorm, encoder: bool, gating: Optional[nn.Module] = None):
        super().__init__(cfg=cfg, final_layer_norm=final_layer_norm)
        self.encoder = encoder
        self.quant_noise = getattr(cfg, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(cfg, "quant_noise_pq_block_size", 8) or 8
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(cfg, "activation_fn", "relu") or "relu"
        )

        if gating is None:
            self.gating = nn.Linear(self.embed_dim, len(self.adapter_modules), bias=False)
        else:
            self.gating = gating

        self.start_temperature = getattr(cfg, "start_adapter_temperature", 1.0)
        self.temperature = getattr(cfg, "const_adapter_temperature", self.start_temperature)
        self.max_temperature = getattr(cfg, "max_adapter_temperature", 1.0)
        self.increasing_steps = getattr(cfg, "adapter_temperature_increasing_steps", 40000)
        self.schedule_type = getattr(cfg, "adapter_schedule_type", "linear")
        if self.schedule_type == "const":
            self.max_temperature = self.temperature

        self.aggregation_func = getattr(cfg, "aggregation_func", "max_pool")

    def set_num_updates(self, num_updates):
        if self.start_temperature != self.max_temperature and self.schedule_type != "const":
            if self.schedule_type == "linear":
                new_temperature = self.start_temperature + (self.max_temperature - self.start_temperature) * \
                                  (num_updates / self.increasing_steps)
            elif self.schedule_type == "quadratic":
                new_temperature = self.start_temperature + ((self.max_temperature - self.start_temperature) *
                                                            (num_updates / self.increasing_steps) ** 2)
            else:
                raise ValueError(f"Unknown scheduler type '{self.schedule_type}'")
            assert new_temperature
            self.temperature = min(self.max_temperature, new_temperature)

        if num_updates % 100 == 0:
            metrics.log_scalar("gate_softmax_temperature", self.temperature)

    @staticmethod
    def masked_max_pool(x: torch.Tensor, mask: Optional[torch.Tensor]):
        if mask is None:
            return x.max(0)[0]
        if mask.dim() == 2:
            assert mask.size(0) == x.size(0) and mask.size(1) == x.size(1)
            mask = ~mask * (-1e30)
            return (x + mask.unsqueeze(-1)).max(dim=0)[0]
        else:
            assert mask.dim() == 3
            seq_len, bsz, hdim = x.size()
            x = x.transpose(1, 0).unsqueeze(1)
            x = x.repeat(1, seq_len, 1, 1)
            mask = ~mask * (-1e30)
            return ((x + mask.unsqueeze(-1)).max(dim=2)[0]).transpose(1, 0)

    @staticmethod
    def masked_avg(x: torch.Tensor, mask: Optional[torch.Tensor]):
        if mask is None:
            return x.mean(0)
        if mask.dim() == 2:
            assert mask.size(0) == x.size(0) and mask.size(1) == x.size(1)
            return (x * mask.unsqueeze(-1)).sum(0) / mask.sum(0).unsqueeze(-1)
        else:
            assert mask.dim() == 3
            seq_len, bsz, hdim = x.size()
            x = x.transpose(1, 0).unsqueeze(1)
            x = x.repeat(1, seq_len, 1, 1)
            x = x * mask.unsqueeze(-1)
            return (x.sum(2) / mask.sum(2).unsqueeze(-1)).transpose(1, 0)

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor, **extras):
        mask = None
        if "padding" in extras and extras["padding"] is not None:
            if extras["padding"].dim() == 2:
                # Padding indicator (1 -> padding) [bsz, seq_len] -> Mask (1 -> not padding) [seq_len, bsz]
                mask = ~extras["padding"].transpose(1, 0)
            else:
                mask = extras["padding"]

        if self.aggregation_func == "avg":
            sentence_x = self.masked_avg(x, mask)
        elif self.aggregation_func == "max_pool":
            sentence_x = self.masked_max_pool(x, mask)
        else:
            raise ValueError(f"Unknown aggregation function '{self.aggregation_func}'")

        gate_logits = self.gating(sentence_x)

        temperature = self.temperature if self.training else self.max_temperature
        gates = torch.softmax(gate_logits/temperature, dim=-1)

        # Normalize weights
        gates_sum = gates.sum(-1) + 1e-9
        gates = gates / gates_sum.unsqueeze(-1)

        if hasattr(self.cfg, "force_using_domains") and getattr(self.cfg, "force_using_domains", False):
            gates = adapter_ids

        assert torch.allclose(gates.sum(-1), torch.ones_like(gates.sum(-1))), \
            f"Gate weights for {self.__class__.__name__} should be equal to one."

        if (not hasattr(self.cfg, "ln_before_adapter") or not self.cfg.ln_before_adapter):
            residual = x
        if self.cfg.adapter_layer_norm:
            x = self.adapter_layer_norm(x)
        elif self.cfg.adapter_reuse_layer_norm:
            x = self.final_layer_norm(x)
        if hasattr(self.cfg, "ln_before_adapter") and self.cfg.ln_before_adapter:
            residual = x

        x_ = []
        for adapter_id in range(len(self.adapter_idx2name)):
            adapter_name = self.adapter_idx2name[adapter_id]
            adapter_module = self.adapter_modules[adapter_name]

            if gates.dim() == 3:
                # Handle decoder auto-regressive training (different gate per token)
                assert not self.encoder
                current_gates = gates[:, :, adapter_id].unsqueeze(-1)
            else:
                current_gates = gates[:, adapter_id].reshape(1, -1, 1)
            x_.append(adapter_module(x) * current_gates)

        # adapters_count x [seq_len, bsz, hdim] -> [seq_len, bsz, adapters_count, hdim]
        x = torch.stack(x_, -2)
        x = x.sum(-2)

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)

        # If gates are provided, do not return them.
        if "gates" in extras and extras["gates"] is not None and not self.gates_everywhere_enabled():
            return x, {}
        return x, {"gates": gate_logits}


class DiscreteAdapters(BaseAdapters):
    """Default, vanilla adapters. (https://aclanthology.org/D19-1165.pdf)
    The code is based on fairseq.models.xmod.transformer_layer_xmod.XMODTransformerEncoderLayerBase
    (especially function lang_adapter) with faster non-homogenous batch handling.
    """

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor, **kwargs):
        if adapter_ids.size(-1) == len(self.adapter_modules):
            with torch.no_grad():
                adapter_ids = adapter_ids.argmax(-1).unsqueeze(-1)
        assert adapter_ids is not None and adapter_ids.size(-1) == 1
        adapter_ids = adapter_ids.flatten().tolist()
        # x: [seq_len, bsz, hidden_dim]
        assert len(adapter_ids) == x.size(1)

        # Group batch indices per adapter.
        indices: List[List[int]] = [[] for _ in range(len(self.adapter_idx2name))]
        for i, adapter_id in enumerate(adapter_ids):
            indices[adapter_id].append(i)

        if (not hasattr(self.cfg, "ln_before_adapter") or not self.cfg.ln_before_adapter):
            residual = x
        if self.cfg.adapter_layer_norm:
            x = self.adapter_layer_norm(x)
        elif self.cfg.adapter_reuse_layer_norm:
            x = self.final_layer_norm(x)
        if hasattr(self.cfg, "ln_before_adapter") and self.cfg.ln_before_adapter:
            residual = x

        new_order = []
        adapters_x = []
        for adapter_id, adapter_name in self.adapter_idx2name.items():
            adapter_indices = indices[adapter_id]
            if not adapter_indices:
                continue
            new_order.extend(adapter_indices)
            adapter_indices = torch.tensor(adapter_indices, dtype=torch.long, device=x.device)
            adapter_x = torch.index_select(x, 1, adapter_indices)
            adapter_x = self.adapter_modules[adapter_name](adapter_x)
            adapters_x.append(adapter_x)

        # Restore initial order.
        arg_sorted_new_order = torch.tensor(new_order, dtype=torch.long, device=x.device).argsort()
        x = torch.cat(adapters_x, 1)[:, arg_sorted_new_order, :]

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)

        return x, {}
