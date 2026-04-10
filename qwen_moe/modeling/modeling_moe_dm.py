# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.distributed as dist
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.utils import logging, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings

from .configuration_moe_dm import MoEConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


@dataclass
class MoECausalLMOutputWithPast(CausalLMOutputWithPast):
    """
    Base class for causal language model outputs with past key values and
    router-side auxiliary losses.

    Parameters:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Cached key/value states for fast autoregressive decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden states from the embedding output and each decoder layer.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Attention weights after the attention softmax.
        router_aux_loss (`torch.FloatTensor`, *optional*):
            Router load-balancing auxiliary loss.
        router_z_loss (`torch.FloatTensor`, *optional*):
            Router z-loss regularizer.
        router_pull_loss (`torch.FloatTensor`, *optional*):
            Prototype pull loss for expert embeddings.
        router_budget_loss (`torch.FloatTensor`, *optional*):
            Soft budget loss encouraging the top-p dispatch to keep each token's
            selected expert count near a target.
    """
    router_aux_loss: Optional[torch.FloatTensor] = None
    router_z_loss: Optional[torch.FloatTensor] = None
    router_pull_loss: Optional[torch.FloatTensor] = None
    router_budget_loss: Optional[torch.FloatTensor] = None


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# def top_p_sampling_batched_all_sequence(logits, top_p=0.9, temperature=1.0):
#     """
#     Apply Top-p sampling to every element in the sequence for each item in the batch.
#     Returns the selected token indices and the corresponding threshold indices.
    
#     :param logits: Logits from a language model with shape (sequence length, batch size, L)
#     :param top_p: Cumulative probability threshold (float)
#     :param temperature: Sampling temperature (float)
#     :return: Tuple of tensors (selected token indices, threshold indices) for each position in each sequence in the batch
#     """
#     # Apply temperature
#     logits = logits / temperature
    
#     # Convert logits to probabilities
#     # probabilities = torch.softmax(logits, dim=-1)
#     # Sort probabilities and their indices in descending order
#     sorted_probs, sorted_indices = torch.sort(logits, descending=True)
#     # print(f"Sorted probabilities: {sorted_probs}")
#     # print(f"top_p: {top_p}")
#     # Compute cumulative probabilities
#     cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
#     mask = cumulative_probs > top_p

#     # Find the threshold indices
#     threshold_indices = mask.long().argmax(dim=-1)
#     threshold_mask = torch.nn.functional.one_hot(threshold_indices, num_classes=sorted_indices.size(-1)).bool()
    
#     mask = mask & ~threshold_mask
#     sorted_indices = torch.where(mask, -1, sorted_indices)
#     sorted_probs = torch.where(mask, 0.0, sorted_probs)   

#     denom = sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
#     sorted_probs = sorted_probs / denom

#     return sorted_probs, sorted_indices

def top_p_sampling_batched_all_sequence(probs, top_p=0.9, temperature=1.0, eps=1e-9):
    """
    probs: (..., num_experts)
    返回:
        selected_probs:  形状同 probs，未选中的位置为 0，选中的位置重新归一化后和为 1
        selected_indices:形状同 probs，未选中的位置为 -1
    """
    # 如果输入已经是概率分布（你当前就是 route_probs），一般不建议再除 temperature。
    # 如果你后面想支持 logits，再单独在外面处理。
    probs = probs.float()
    # probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    # probs = probs.clamp_min(0.0)
    # probs_sum = probs.sum(dim=-1, keepdim=True)
    # uniform_probs = torch.full_like(probs, 1.0 / probs.size(-1))
    # probs = torch.where(probs_sum > eps, probs / probs_sum.clamp_min(eps), uniform_probs)

    # 按概率从大到小排序
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    # print(f"sorted_probs: {sorted_probs}")
    # 计算 cumulative sum，保留使累计概率刚超过 top_p 的那个专家
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs > top_p

    threshold_indices = mask.long().argmax(dim=-1)
    threshold_mask = torch.nn.functional.one_hot(
        threshold_indices, num_classes=sorted_indices.size(-1)
    ).bool()

    # 超过 top_p 的去掉，但第一次超过阈值的那个保留
    mask = mask & ~threshold_mask

    selected_indices = torch.where(mask, -1, sorted_indices)
    selected_probs = torch.where(mask, 0.0, sorted_probs)

    # 关键：对保留下来的专家权重重新归一化到 1
    denom = selected_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    selected_probs = selected_probs / denom

    return selected_probs, selected_indices

def top_k_routing_batched_all_sequence(probs, top_k: int):
    """
    probs: (seq_len, batch_size, num_experts)
    返回:
        topk_probs: (seq_len, batch_size, top_k)
        topk_idx:   (seq_len, batch_size, top_k)
    """
    topk_probs, topk_idx = torch.topk(probs, k=top_k, dim=-1)

    # 重新归一化，只在选中的 top-k 上归一化
    denom = topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    topk_probs = topk_probs / denom
    return topk_probs, topk_idx

def entmax_bisect(
    inputs: torch.Tensor,
    alpha: float = 1.5,
    dim: int = -1,
    n_iter: int = 32,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Differentiable alpha-entmax via bisection.
    alpha in (1, 2], where:
      alpha -> 1 : softmax
      alpha = 2  : sparsemax
    """
    if not (1.0 < alpha <= 2.0):
        raise ValueError(f"alpha must be in (1, 2], got {alpha}")

    alpha_m1 = alpha - 1.0
    inv_alpha_m1 = 1.0 / alpha_m1

    # Shift for numerical stability.
    x = inputs - inputs.max(dim=dim, keepdim=True).values

    # Search tau s.t. sum(((alpha-1)*(x - tau))_+^(1/(alpha-1))) = 1.
    tau_lo = x.min(dim=dim, keepdim=True).values - 1.0
    tau_hi = x.max(dim=dim, keepdim=True).values

    for _ in range(n_iter):
        tau_mid = (tau_lo + tau_hi) * 0.5
        p_mid = torch.clamp(alpha_m1 * (x - tau_mid), min=0.0) ** inv_alpha_m1
        sum_p = p_mid.sum(dim=dim, keepdim=True)
        tau_lo = torch.where(sum_p > 1.0, tau_mid, tau_lo)
        tau_hi = torch.where(sum_p <= 1.0, tau_mid, tau_hi)

    tau_star = (tau_lo + tau_hi) * 0.5
    probs = torch.clamp(alpha_m1 * (x - tau_star), min=0.0) ** inv_alpha_m1
    probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(eps)
    return probs


def _mean_scalar_stat(stat_dicts, key: str):
    values = [float(stats[key]) for stats in stat_dicts if stats and stats.get(key) is not None]
    if not values:
        return None
    return sum(values) / float(len(values))


def _mean_tensor_stat(stat_dicts, key: str):
    values = [stats[key].detach().float() for stats in stat_dicts if stats and torch.is_tensor(stats.get(key))]
    if not values:
        return None
    return torch.stack(values, dim=0).mean(dim=0)

class CrossAttentionRouter(nn.Module):
    """
    Router that scores experts by cross attention from token queries to learnable
    expert embeddings.

    Input:
        hidden_states: (batch, seq, hidden)
    Output:
        route_probs: (batch, seq, num_experts)
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        d_router: Optional[int] = None,
        use_entmax: bool = False,
        alpha: float = 1.5,
        use_softmax_temperature: bool = False,
        softmax_temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.d_router = int(d_router if d_router is not None else hidden_size)
        # print(f"d_router in CrossAttentionRouter: {self.d_router}")
        self.use_entmax = bool(use_entmax)
        self.alpha = alpha
        self.use_softmax_temperature = bool(use_softmax_temperature)
        self.softmax_temperature = float(softmax_temperature)

        self.query = nn.Linear(self.hidden_size, self.d_router, bias=False)
        self.key = nn.Linear(self.d_router, self.d_router, bias=False)

        # Kept for checkpoint compatibility with earlier experiments. The
        # q-k routing space is now reused directly for pull/EMA updates.
        self.token_couple_proj = nn.Linear(self.hidden_size, self.d_router, bias=False)

        self.value = nn.Linear(self.d_router, self.num_experts, bias=False)

        self.expert_embed = nn.Parameter(torch.randn(self.num_experts, self.d_router))
        self._shared_expert_embed_ref = None
        self.last_router_forward_stats = {}

    def set_shared_expert_embed(self, expert_embed: nn.Parameter) -> None:
        self.expert_embed = None
        self._shared_expert_embed_ref = [expert_embed]

    @staticmethod
    def _reshape_legacy_router_weight(
        legacy_router_weight: torch.Tensor,
        target_shape: Tuple[int, int],
    ) -> torch.Tensor:
        source = legacy_router_weight.detach().float()
        candidates = (source, source.transpose(0, 1))

        def _score(candidate: torch.Tensor) -> Tuple[int, int, int]:
            return (
                int(candidate.shape == target_shape),
                int(candidate.shape[0] == target_shape[0]) + int(candidate.shape[1] == target_shape[1]),
                min(candidate.shape[0], target_shape[0]) * min(candidate.shape[1], target_shape[1]),
            )

        best = max(candidates, key=_score)
        reshaped = best.new_zeros(target_shape)
        rows = min(best.shape[0], target_shape[0])
        cols = min(best.shape[1], target_shape[1])
        reshaped[:rows, :cols] = best[:rows, :cols]
        return reshaped

    def initialize_expert_embed_from_legacy_router(self, legacy_router_weight: torch.Tensor) -> torch.Tensor:
        target = self.get_expert_embed()
        reshaped = self._reshape_legacy_router_weight(legacy_router_weight, tuple(target.shape))
        reshaped = reshaped.to(dtype=target.dtype, device=target.device)
        with torch.no_grad():
            target.copy_(reshaped)
        return target

    def get_expert_embed(self) -> torch.Tensor:
        if self._shared_expert_embed_ref is not None:
            return self._shared_expert_embed_ref[0]
        if self.expert_embed is None:
            raise RuntimeError("CrossAttentionRouter expert_embed is not initialized.")
        return self.expert_embed

    def project_routed_expert_repr_to_embed_space(self, routed_expert_repr: torch.Tensor) -> torch.Tensor:
        """
        Map routed expert representations k-space back to the stored
        `expert_embed` parameter space so EMA can still update the parameter.
        """
        key_weight_t = self.key.weight.detach().float().transpose(0, 1)
        key_weight_t_pinv = torch.linalg.pinv(key_weight_t)
        return routed_expert_repr.float() @ key_weight_t_pinv

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_router_repr: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Keep router params in fp32 while allowing upstream hidden states in fp16/bf16.
        router_in = hidden_states.to(self.query.weight.dtype)
        expert_embed = self.get_expert_embed()

        q = self.query(router_in).float()                  # (b, s, d)
        k = self.key(expert_embed.to(self.key.weight.dtype)).float()    # (e, d)
        v = self.value(expert_embed.to(self.value.weight.dtype)).float()  # (e, e)
        attn_scores = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(self.d_router)  # (b, s, e)
        if self.use_entmax:
            attn_weights = entmax_bisect(attn_scores, alpha=self.alpha, dim=-1)
        else:
            softmax_scores = attn_scores
            if self.use_softmax_temperature:
                softmax_scores = softmax_scores / max(self.softmax_temperature, 1e-6)
            attn_weights = F.softmax(softmax_scores, dim=-1, dtype=torch.float32)
        attn_output = torch.matmul(attn_weights, v)        # (b, s, e)

        # Keep routing semantics aligned with the dense router: scores are q-k
        # logits, probabilities are their normalized attention weights.
        route_scores = attn_scores
        route_probs = attn_weights
        attn_scores_f = attn_scores.detach().float()
        route_probs_f = route_probs.detach().float()
        attn_output_f = attn_output.detach().float()
        row_sums = route_probs_f.sum(dim=-1)
        probs_clamped = route_probs_f.clamp_min(1e-9)
        attn_entropy = -(probs_clamped * probs_clamped.log()).sum(dim=-1)
        attn_top1_mass = route_probs_f.max(dim=-1).values
        self.last_router_forward_stats = {
            "attn_scores_mean": float(attn_scores_f.mean().item()),
            "attn_scores_std": float(attn_scores_f.std().item()),
            "attn_scores_min": float(attn_scores_f.min().item()),
            "attn_scores_max": float(attn_scores_f.max().item()),
            "attn_weights_row_sum_mean": float(row_sums.mean().item()),
            "attn_weights_row_sum_abs_err": float((row_sums - 1.0).abs().mean().item()),
            "attn_weights_entropy": float(attn_entropy.mean().item()),
            "attn_weights_top1_mass": float(attn_top1_mass.mean().item()),
            "attn_output_mean": float(attn_output_f.mean().item()),
            "attn_output_std": float(attn_output_f.std().item()),
            "attn_output_min": float(attn_output_f.min().item()),
            "attn_output_max": float(attn_output_f.max().item()),
            "route_prob_min": float(route_probs_f.min().item()),
            "route_prob_has_neg": float(route_probs_f.lt(0).any().item()),
            "route_prob_row_sum_mean": float(row_sums.mean().item()),
            "route_prob_row_sum_abs_err": float((row_sums - 1.0).abs().mean().item()),
        }
        router_repr = (q, k) if return_router_repr else None
        return route_scores, route_probs, router_repr


class SwitchMLP(nn.Module):
    """Routes tokens to N experts with a simplified token-internal router."""

    def __init__(self, config, layer_idx):
        super(SwitchMLP, self).__init__()
        self.layer_num = layer_idx
        self.use_switch = (layer_idx % config.expert_frequency) == 0
        self.last_top_p_active_expert_count = 0
        self.last_router_aux_loss = None
        self.last_router_z_loss = None

        # [新增]
        self.last_router_pull_loss = None
        self.last_router_budget_loss = None
        self.last_router_forward_stats = {}
        self.last_router_dispatch_stats = {}
        self.last_router_ema_stats = {}
        self.router_pull_temperature = float(getattr(config, "router_pull_temperature", 1.0))
        self.router_pull_loss_type = str(getattr(config, "router_pull_loss_type", "soft"))
        self.router_ema_momentum = float(getattr(config, "router_ema_momentum", 0.99))
        self.router_use_ema_update = bool(getattr(config, "router_use_ema_update", False))
        self.router_budget_target_count = float(getattr(config, "router_budget_target_count", 0.0))
        self.router_budget_tau = float(getattr(config, "router_budget_tau", 0.05))
        self._pending_router_ema_token_q = None
        self._pending_router_ema_selected_probs = None

        if self.use_switch:
            self.experts = nn.ModuleList()
            self.num_experts = config.num_experts
            for _ in range(config.num_experts):
                self.experts.append(
                    LlamaMLP(
                        config.hidden_size,
                        config.intermediate_size,
                        config.hidden_act,
                    )
                )

            self.router_top_p = float(getattr(config, "top_p_threshold", 0.7))
            if not (0.0 < self.router_top_p <= 1.0):
                raise ValueError(
                    f"top_p_threshold must be in (0, 1], got {self.router_top_p}"
                )

            self.use_cross_attention_router = getattr(config, "use_cross_attention_router", True)
            self.router_use_entmax = bool(getattr(config, "router_use_entmax", False))
            self.router_entmax_alpha = float(getattr(config, "router_entmax_alpha", 1.5))
            if self.router_use_entmax and not (1.0 < self.router_entmax_alpha <= 2.0):
                raise ValueError(
                    f"`router_entmax_alpha` must be in (1, 2] when `router_use_entmax=True`, "
                    f"got {self.router_entmax_alpha}"
                )

            if self.use_cross_attention_router:
                self.router = CrossAttentionRouter(
                    hidden_size=config.hidden_size,
                    num_experts=config.num_experts,
                    d_router=getattr(config, "router_dim", config.hidden_size),
                    use_entmax=self.router_use_entmax,
                    alpha=self.router_entmax_alpha,
                    use_softmax_temperature=bool(getattr(config, "router_use_softmax_temperature", False)),
                    softmax_temperature=float(getattr(config, "router_softmax_temperature", 1.0)),
                )
            else:
                self.router = nn.Linear(
                    config.hidden_size,
                    config.num_experts,
                    bias=False,
                )

        else:
            self.mlp = LlamaMLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
            )
    def _compute_prototype_pull_loss(
        self,
        token_q: torch.Tensor,          # (b, s, d)
        expert_axis_probs: torch.Tensor,   # (b, s, e), top-p renormalized weights on expert axis
        expert_k: torch.Tensor,         # (e, d)
    ) -> torch.Tensor:
        """
        Pull loss in the actual q-k routing space.
        The top-p-selected probabilities are treated as detached soft labels so
        the prototype alignment follows real dispatch decisions.
        """
        q = F.normalize(token_q.float(), dim=-1)                    # (b, s, d)
        k = F.normalize(expert_k.float(), dim=-1)                   # (e, d)
        temp = max(self.router_pull_temperature, 1e-6)
        sim = torch.matmul(q, k.transpose(0, 1)) / temp             # (b, s, e)

        if self.router_pull_loss_type == "soft":
            assign = expert_axis_probs.detach().float()
            log_probs = F.log_softmax(sim, dim=-1)
            return -(assign * log_probs).sum(dim=-1).mean()

        if self.router_pull_loss_type == "hard_ce":
            target = expert_axis_probs.detach().argmax(dim=-1).long()
            return F.cross_entropy(sim.reshape(-1, sim.size(-1)), target.reshape(-1))

        raise ValueError(f"Unsupported router_pull_loss_type: {self.router_pull_loss_type!r}")

    def _scatter_selected_probs_to_expert_axis(
        self,
        selected_probs: torch.Tensor,   # (b, s, e), weights laid out on sorted top-p slots
        selected_indices: torch.Tensor, # (b, s, e), expert ids or -1
    ) -> torch.Tensor:
        """
        top_p_sampling_batched_all_sequence returns weights aligned to the sorted
        slot order rather than the original expert axis. Scatter them back so
        EMA/pull computations operate on true expert ids.
        """
        expert_axis_probs = selected_probs.new_zeros(selected_probs.shape)
        valid_mask = selected_indices >= 0
        if not torch.any(valid_mask):
            return expert_axis_probs
        scatter_index = selected_indices.clamp_min(0)
        expert_axis_probs.scatter_add_(
            dim=-1,
            index=scatter_index,
            src=selected_probs * valid_mask.to(selected_probs.dtype),
        )
        return expert_axis_probs

    @torch.no_grad()
    def _ema_update_expert_embed(
        self,
        token_q: torch.Tensor,          # (b, s, d)
        expert_axis_probs: torch.Tensor,   # (b, s, e), top-p renormalized weights on expert axis
    ) -> None:
        """
        Compute expert prototypes in q-space with top-p weights, smooth them in
        routed k-space, then map them back to `expert_embed` parameter space.
        """
        if not self.use_cross_attention_router:
            self.last_router_ema_stats = {}
            return

        router = self.router
        expert_embed = router.get_expert_embed()
        device = expert_embed.device
        dtype = expert_embed.dtype

        flat_q = F.normalize(token_q.detach().float(), dim=-1).reshape(-1, token_q.size(-1))     # (b*s, d)
        flat_selected_probs = expert_axis_probs.detach().float().reshape(-1, expert_axis_probs.size(-1))  # (b*s, e)
        proto_sums = flat_selected_probs.transpose(0, 1) @ flat_q                                 # (e, d)
        proto_counts = flat_selected_probs.sum(dim=0)                                             # (e,)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(proto_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(proto_counts, op=dist.ReduceOp.SUM)

        active_mask = proto_counts > 0
        if not torch.any(active_mask):
            self.last_router_ema_stats = {
                "proto_counts": proto_counts.detach().float(),
                "active_expert_count": 0.0,
                "proto_count_mean": float(proto_counts.mean().item()),
                "proto_count_min": float(proto_counts.min().item()),
                "proto_count_max": float(proto_counts.max().item()),
                "proto_update_cosine": None,
                "proto_delta_norm": None,
            }
            return

        proto_means = proto_sums[active_mask] / proto_counts[active_mask].unsqueeze(-1)
        proto_means = F.normalize(proto_means, dim=-1)

        current_routed = router.key(expert_embed.to(router.key.weight.dtype)).detach().float()
        current_routed = F.normalize(current_routed, dim=-1)

        old = current_routed[active_mask]
        new_routed = self.router_ema_momentum * old + (1.0 - self.router_ema_momentum) * proto_means
        new_routed = F.normalize(new_routed, dim=-1)
        proto_update_cosine = F.cosine_similarity(old, new_routed, dim=-1)
        proto_delta_norm = (new_routed - old).norm(dim=-1)

        updated_embed = router.project_routed_expert_repr_to_embed_space(new_routed)
        expert_embed[active_mask].copy_(updated_embed.to(device=device, dtype=dtype))
        self.last_router_ema_stats = {
            "proto_counts": proto_counts.detach().float(),
            "active_expert_count": float(active_mask.sum().item()),
            "proto_count_mean": float(proto_counts.mean().item()),
            "proto_count_min": float(proto_counts.min().item()),
            "proto_count_max": float(proto_counts.max().item()),
            "proto_update_cosine": float(proto_update_cosine.mean().item()),
            "proto_delta_norm": float(proto_delta_norm.mean().item()),
        }

    @torch.no_grad()
    def apply_pending_router_ema_update(self) -> None:
        token_q = self._pending_router_ema_token_q
        expert_axis_probs = self._pending_router_ema_selected_probs
        self._pending_router_ema_token_q = None
        self._pending_router_ema_selected_probs = None
        if token_q is None or expert_axis_probs is None:
            return
        self._ema_update_expert_embed(token_q=token_q, expert_axis_probs=expert_axis_probs)
    
    def forward(self, hidden_states):
        """
        hidden_states: (batch, seq, hidden)
        """
        if not self.use_switch:
            self.last_router_aux_loss = None
            self.last_router_z_loss = None
            self.last_router_pull_loss = None
            self.last_router_budget_loss = None
            self.last_router_forward_stats = {}
            self.last_router_dispatch_stats = {}
            self.last_router_ema_stats = {}
            self._pending_router_ema_token_q = None
            self._pending_router_ema_selected_probs = None
            return self.mlp(hidden_states)

        bsz, seq_len, hidden_dim = hidden_states.size()

        # 1) router output -> route probabilities.
        if self.use_cross_attention_router:
            compute_router_losses = self.training
            router_logits, route_probs, router_repr = self.router(
                hidden_states,
                return_router_repr=compute_router_losses,
            )
            self.last_router_forward_stats = dict(getattr(self.router, "last_router_forward_stats", {}))
            router_logits = router_logits.float()
            route_probs = route_probs.float()
            router_q = None
            expert_k = None
            if router_repr is not None:
                router_q, expert_k = router_repr
                router_q = router_q.float()
                expert_k = expert_k.float()

            if route_probs.shape != router_logits.shape:
                raise RuntimeError(
                    "CrossAttentionRouter must return per-expert probabilities with shape "
                    f"{tuple(router_logits.shape)}, got {tuple(route_probs.shape)}"
                )
            if compute_router_losses:
                self.last_router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
            else:
                self.last_router_z_loss = None
        else:
            router_logits = self.router(hidden_states)
            route_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
            if self.router_use_entmax:
                route_probs = entmax_bisect(router_logits.float(), alpha=self.router_entmax_alpha, dim=-1)
            self.last_router_z_loss = None
            self.last_router_forward_stats = {}
            router_q = None
            expert_k = None

        route_probs_for_stats = route_probs.float()
        soft_load = route_probs_for_stats.mean(dim=(0, 1))
        top1_experts = route_probs_for_stats.argmax(dim=-1)
        hard_load = F.one_hot(top1_experts, num_classes=self.num_experts).to(route_probs_for_stats.dtype).mean(dim=(0, 1))
        if self.training:
            self.last_router_aux_loss = self.num_experts * torch.sum(hard_load * soft_load)
        else:
            self.last_router_aux_loss = None

        # 2) top-p routing.
        topk_weights, topk_ind = top_p_sampling_batched_all_sequence(
            route_probs,
            self.router_top_p,
        )  # both: (b, s, num_experts), masked indices are -1
        expert_axis_topk_weights = self._scatter_selected_probs_to_expert_axis(topk_weights, topk_ind)
        selected_expert_count_per_token = (topk_ind >= 0).sum(dim=-1)  # (b, s)
        self.last_top_p_avg_expert_count = selected_expert_count_per_token.float().mean().item()
        top2_k = min(2, route_probs_for_stats.size(-1))
        top2_values = torch.topk(route_probs_for_stats, k=top2_k, dim=-1).values
        top1_top2_margin = top2_values[..., 0] - top2_values[..., 1] if top2_k == 2 else top2_values[..., 0]
        selected_mask = topk_ind >= 0
        gathered_route_probs = route_probs_for_stats.gather(-1, topk_ind.clamp_min(0))
        top_p_pre_mass = (gathered_route_probs * selected_mask.to(gathered_route_probs.dtype)).sum(dim=-1)
        top_p_post_sum = topk_weights.sum(dim=-1)

        sorted_route_probs = torch.sort(route_probs, dim=-1, descending=True).values
        if sorted_route_probs.size(-1) > 1:
            prev_cum = torch.cumsum(sorted_route_probs, dim=-1)[..., :-1]
            soft_extra_selected = torch.sigmoid(
                (self.router_top_p - prev_cum) / max(self.router_budget_tau, 1e-6)
            )
            soft_selected_expert_count = 1.0 + soft_extra_selected.sum(dim=-1)
        else:
            soft_selected_expert_count = torch.ones_like(sorted_route_probs[..., 0])

        self.last_router_dispatch_stats = {
            "avg_selected_expert_count": float(selected_expert_count_per_token.float().mean().item()),
            "soft_selected_expert_count": float(soft_selected_expert_count.detach().float().mean().item()),
            "soft_load": soft_load.detach().float(),
            "hard_load": hard_load.detach().float(),
            "dead_expert_ratio": float((hard_load <= 0).float().mean().item()),
            "top1_top2_margin": float(top1_top2_margin.mean().item()),
            "top_p_pre_mass_mean": float(top_p_pre_mass.mean().item()),
            "top_p_post_sum_mean": float(top_p_post_sum.mean().item()),
            "top_p_post_sum_abs_err": float((top_p_post_sum - 1.0).abs().mean().item()),
            "router_top_p": float(self.router_top_p),
        }

        if self.training and self.router_budget_target_count > 0.0:
            self.last_router_budget_loss = (
                soft_selected_expert_count.mean() - self.router_budget_target_count
            ).pow(2)
        else:
            self.last_router_budget_loss = None

        # [新增] Prototype Pull Loss
        if self.training and self.use_cross_attention_router:
            if router_q is None or expert_k is None:
                raise RuntimeError("CrossAttentionRouter did not return q-k router representations during training.")
            self.last_router_pull_loss = self._compute_prototype_pull_loss(
                token_q=router_q,
                expert_axis_probs=expert_axis_topk_weights,
                expert_k=expert_k,
            )
        else:
            self.last_router_pull_loss = None

        # [新增] EMA 更新 expert_embed
        if self.training and self.use_cross_attention_router and self.router_use_ema_update:
            self._pending_router_ema_token_q = router_q.detach()
            self._pending_router_ema_selected_probs = expert_axis_topk_weights.detach()
        else:
            self._pending_router_ema_token_q = None
            self._pending_router_ema_selected_probs = None
            if not self.router_use_ema_update:
                self.last_router_ema_stats = {}

        # 3) flatten tokens then sparse expert dispatch.
        flat_hidden = hidden_states.reshape(-1, hidden_dim)                  # (b*s, h)
        flat_topk_weights = topk_weights.reshape(-1, topk_weights.size(-1))  # (b*s, num_experts)
        flat_topk_ind = topk_ind.reshape(-1, topk_ind.size(-1))              # (b*s, num_experts)

        output_total = torch.zeros_like(flat_hidden)
        for expert_num, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(flat_topk_ind == expert_num)
            if token_idx.numel() == 0:
                continue

            selected_hidden = flat_hidden[token_idx]  # (n, h)
            expert_output = expert(selected_hidden)   # (n, h)

            expert_weight = flat_topk_weights[token_idx, slot_idx].unsqueeze(-1)
            expert_weight = expert_weight.to(expert_output.dtype)
            output_total[token_idx] += expert_output * expert_weight

        output_total = output_total.view(bsz, seq_len, hidden_dim)
        # print(f"switch output finite: {torch.isfinite(output_total).all().item()}")
        return output_total

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: MoEConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        # self.mlp = LlamaMLP(
        #     hidden_size=self.hidden_size,
        #     intermediate_size=config.intermediate_size,
        #     hidden_act=config.hidden_act,
        # )
        self.mlp = SwitchMLP(config, layer_idx)
        # self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        # print(f"decoder hidden_states in finite      : {torch.isfinite(hidden_states).all().item()}")
        residual = hidden_states

        # hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.input_norm(hidden_states)
        # print(f"after input_norm finite             : {torch.isfinite(hidden_states).all().item()}")


        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # print(f"self_attn output finite             : {torch.isfinite(hidden_states).all().item()}")
        hidden_states = residual + hidden_states
        # print(f"after attn residual add finite      : {torch.isfinite(hidden_states).all().item()}")

        # Fully Connected
        residual = hidden_states
        # hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.post_attention_norm(hidden_states)
        # print(f"after post_attention_norm finite    : {torch.isfinite(hidden_states).all().item()}")
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class MoEPreTrainedModel(PreTrainedModel):
    config_class = MoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            if not module.weight.dtype.is_floating_point:
                raise RuntimeError(
                    "Tried to initialize a non-floating Linear weight: "
                    f"module_cls={module.__class__.__name__}, "
                    f"dtype={module.weight.dtype}, "
                    f"shape={tuple(module.weight.shape)}"
                )
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, MoEModel):
            module.gradient_checkpointing = value

LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class MoEModel(MoEPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: MoEConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.shared_expert_embed = None
        self.last_router_pull_loss = None
        self.last_router_budget_loss = None
        self.last_router_forward_stats = {}
        self.last_router_dispatch_stats = {}
        self.last_router_ema_stats = {}
        # print(f"share_router_expert_embedding: {getattr(config, 'share_router_expert_embedding', False)}, \
        #       use_cross_attention_router: {getattr(config, 'use_cross_attention_router', True)}, num_experts: {getattr(config, 'num_experts', 0)},\
        #       num_hidden_layers: {getattr(config, 'num_hidden_layers', 0)}")
        # print(f"share_router_expert_embedding: {getattr(config, 'share_router_expert_embedding', False)}")
        if (
            bool(getattr(config, "share_router_expert_embedding", False))
            and bool(getattr(config, "use_cross_attention_router", True))
            and int(getattr(config, "num_experts", 0)) > 0
        ):
            d_router = int(getattr(config, "router_dim", config.hidden_size))
            # print(f"d_router: {d_router}")
            self.shared_expert_embed = nn.Parameter(torch.randn(config.num_experts, d_router))

        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        if self.shared_expert_embed is not None:
            for layer in self.layers:
                mlp = getattr(layer, "mlp", None)
                if not getattr(mlp, "use_switch", False):
                    continue
                if not getattr(mlp, "use_cross_attention_router", False):
                    continue
                mlp.router.set_shared_expert_embed(self.shared_expert_embed)

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.last_router_aux_loss = None
        self.last_router_z_loss = None
        self.last_router_budget_loss = None

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # print(f"inputs_embeds finite                 : {torch.isfinite(inputs_embeds).all().item()}")
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        # print(f"hidden_states before layers finite   : {torch.isfinite(hidden_states).all().item()}")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        self.last_router_aux_loss = None
        self.last_router_z_loss = None
        self.last_router_pull_loss = None
        self.last_router_budget_loss = None
        self.last_router_forward_stats = {}
        self.last_router_dispatch_stats = {}
        self.last_router_ema_stats = {}

        for idx, decoder_layer in enumerate(self.layers):
            # print(f"layer {idx} input finite             : {torch.isfinite(hidden_states).all().item()}")
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            # print(f"layer {idx} output finite            : {torch.isfinite(hidden_states).all().item()}")

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if self.training:
            aux_terms = []
            z_terms = []
            pull_terms = []
            budget_terms = []
            forward_stat_terms = []
            dispatch_stat_terms = []
            ema_stat_terms = []

            for layer in self.layers:
                mlp = getattr(layer, "mlp", None)
                aux = getattr(mlp, "last_router_aux_loss", None)
                if aux is not None:
                    aux_terms.append(aux)

                z_loss = getattr(mlp, "last_router_z_loss", None)
                if z_loss is not None:
                    z_terms.append(z_loss)

                pull_loss = getattr(mlp, "last_router_pull_loss", None)
                if pull_loss is not None:
                    pull_terms.append(pull_loss)
                budget_loss = getattr(mlp, "last_router_budget_loss", None)
                if budget_loss is not None:
                    budget_terms.append(budget_loss)
                forward_stats = getattr(mlp, "last_router_forward_stats", None)
                if forward_stats:
                    forward_stat_terms.append(forward_stats)
                dispatch_stats = getattr(mlp, "last_router_dispatch_stats", None)
                if dispatch_stats:
                    dispatch_stat_terms.append(dispatch_stats)
                ema_stats = getattr(mlp, "last_router_ema_stats", None)
                if ema_stats:
                    ema_stat_terms.append(ema_stats)

            if aux_terms:
                self.last_router_aux_loss = torch.stack(aux_terms).mean()
            else:
                self.last_router_aux_loss = hidden_states.new_zeros(())

            if z_terms:
                self.last_router_z_loss = torch.stack(z_terms).mean()
            else:
                self.last_router_z_loss = hidden_states.new_zeros(())

            if pull_terms:
                self.last_router_pull_loss = torch.stack(pull_terms).mean()
            else:
                self.last_router_pull_loss = hidden_states.new_zeros(())

            if budget_terms:
                self.last_router_budget_loss = torch.stack(budget_terms).mean()
            else:
                self.last_router_budget_loss = hidden_states.new_zeros(())

            forward_scalar_keys = (
                "attn_scores_mean",
                "attn_scores_std",
                "attn_scores_min",
                "attn_scores_max",
                "attn_weights_row_sum_mean",
                "attn_weights_row_sum_abs_err",
                "attn_weights_entropy",
                "attn_weights_top1_mass",
                "attn_output_mean",
                "attn_output_std",
                "attn_output_min",
                "attn_output_max",
                "route_prob_min",
                "route_prob_has_neg",
                "route_prob_row_sum_mean",
                "route_prob_row_sum_abs_err",
            )
            self.last_router_forward_stats = (
                {key: _mean_scalar_stat(forward_stat_terms, key) for key in forward_scalar_keys}
                if forward_stat_terms else {}
            )

            dispatch_scalar_keys = (
                "avg_selected_expert_count",
                "soft_selected_expert_count",
                "dead_expert_ratio",
                "top1_top2_margin",
                "top_p_pre_mass_mean",
                "top_p_post_sum_mean",
                "top_p_post_sum_abs_err",
                "router_top_p",
            )
            self.last_router_dispatch_stats = (
                {key: _mean_scalar_stat(dispatch_stat_terms, key) for key in dispatch_scalar_keys}
                if dispatch_stat_terms else {}
            )
            if dispatch_stat_terms:
                self.last_router_dispatch_stats["soft_load"] = _mean_tensor_stat(dispatch_stat_terms, "soft_load")
                self.last_router_dispatch_stats["hard_load"] = _mean_tensor_stat(dispatch_stat_terms, "hard_load")

            ema_scalar_keys = (
                "active_expert_count",
                "proto_count_mean",
                "proto_count_min",
                "proto_count_max",
                "proto_update_cosine",
                "proto_delta_norm",
            )
            self.last_router_ema_stats = (
                {key: _mean_scalar_stat(ema_stat_terms, key) for key in ema_scalar_keys}
                if ema_stat_terms else {}
            )
            if ema_stat_terms:
                self.last_router_ema_stats["proto_counts"] = _mean_tensor_stat(ema_stat_terms, "proto_counts")

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class MoEForCausalLM(MoEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = MoEModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoECausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoECausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            # Keep `loss` as pure language-modeling CE. Router regularizers are
            # exposed separately through `self.model.last_router_*_loss`.

        router_aux_loss = getattr(self.model, "last_router_aux_loss", None)
        router_z_loss = getattr(self.model, "last_router_z_loss", None)
        router_pull_loss = getattr(self.model, "last_router_pull_loss", None)
        router_budget_loss = getattr(self.model, "last_router_budget_loss", None)

        if not return_dict:
            output = (logits,) + outputs[1:]
            router_terms = (router_aux_loss, router_z_loss, router_pull_loss, router_budget_loss)
            return ((loss,) + output + router_terms) if loss is not None else (output + router_terms)

        return MoECausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_aux_loss=router_aux_loss,
            router_z_loss=router_z_loss,
            router_pull_loss=router_pull_loss,
            router_budget_loss=router_budget_loss,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(MoEPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
