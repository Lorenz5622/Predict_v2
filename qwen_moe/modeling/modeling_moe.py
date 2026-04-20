# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Qwen2MoE model."""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask, is_flash_attn_available
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import auto_docstring, can_return_tuple, is_torch_flex_attn_available, logging
from transformers.utils.deprecation import deprecate_kwarg
from .configuration_moe import Qwen2MoeConfig


if is_flash_attn_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask

logger = logging.get_logger(__name__)


@dataclass
class Qwen2MoeCausalLMOutputWithPast(MoeCausalLMOutputWithPast):
    router_aux_loss: Optional[torch.FloatTensor] = None
    router_z_loss: Optional[torch.FloatTensor] = None
    router_pull_loss: Optional[torch.FloatTensor] = None
    router_budget_loss: Optional[torch.FloatTensor] = None


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, routing_weights.shape[1]))
            .reshape(-1, routing_weights.shape[1])
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    device_index = routing_weights.device.index if routing_weights.device.index is not None else 0
    rank = routing_weights.shape[1] * int(device_index)
    overall_loss = torch.sum(
        tokens_per_expert[:, rank : rank + routing_weights.shape[1]] * router_prob_per_expert.unsqueeze(0)
    )
    return overall_loss * num_experts


def top_k_routing_batched_all_sequence(probs: torch.Tensor, top_k: int):
    topk_probs, topk_idx = torch.topk(probs, k=top_k, dim=-1)
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
    if not (1.0 < alpha <= 2.0):
        raise ValueError(f"alpha must be in (1, 2], got {alpha}")

    alpha_m1 = alpha - 1.0
    inv_alpha_m1 = 1.0 / alpha_m1
    x = inputs - inputs.max(dim=dim, keepdim=True).values
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


def _pairwise_cosine_stats(vectors: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
    vectors = vectors.detach().float()
    num_vectors = int(vectors.size(0))
    if num_vectors < 2:
        return None, None
    vectors = F.normalize(vectors, dim=-1)
    cosine = torch.matmul(vectors, vectors.transpose(0, 1))
    mask = ~torch.eye(num_vectors, dtype=torch.bool, device=cosine.device)
    pairwise = cosine[mask]
    if pairwise.numel() == 0:
        return None, None
    return float(pairwise.mean().item()), float(pairwise.max().item())


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2Moe
class Qwen2MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2MoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Qwen2Moe
class Qwen2MoeRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen2MoeConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Modified from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2Moe
class Qwen2MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LowRankRouter(nn.Module):
    """Low-rank router: hidden -> rank -> experts, with optional sharp routing."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        rank: int,
        router_temperature_init: float = 10.0,
        router_eps: float = 1e-6,
        normalize_q: bool = True,
        normalize_k: bool = True,
    ):
        super().__init__()
        self.router_eps = router_eps
        self.normalize_q = normalize_q
        self.normalize_k = normalize_k

        self.router_down = nn.Linear(hidden_size, rank, bias=False)
        self.router_up = nn.Linear(rank, num_experts, bias=False)
        self.log_router_temperature = nn.Parameter(torch.log(torch.tensor(float(router_temperature_init))))

    @property
    def router_temperature(self):
        return torch.exp(self.log_router_temperature)

    def _normalize_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(self.router_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.router_down(hidden_states)
        if self.normalize_q:
            q = self._normalize_last_dim(q)

        k = self.router_up.weight
        if self.normalize_k:
            k = k / k.norm(dim=-1, keepdim=True).clamp_min(self.router_eps)

        router_logits = torch.matmul(q, k.transpose(0, 1))
        return self.router_temperature * router_logits


class CrossAttentionRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        d_router: Optional[int] = None,
        use_entmax: bool = False,
        alpha: float = 1.5,
        use_softmax_temperature: bool = True,
        softmax_temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.d_router = int(d_router if d_router is not None else hidden_size)
        self.use_entmax = bool(use_entmax)
        self.alpha = float(alpha)
        self.use_softmax_temperature = bool(use_softmax_temperature)
        self.softmax_temperature = float(softmax_temperature)

        self.query = nn.Linear(self.hidden_size, self.d_router, bias=False)
        self.key = nn.Linear(self.d_router, self.d_router, bias=False)
        self.token_couple_proj = nn.Linear(self.hidden_size, self.d_router, bias=False)
        self.value = nn.Linear(self.d_router, self.d_router, bias=False)
        self.router_context_proj = nn.Linear(self.d_router, self.hidden_size, bias=False)

        init_expert_state = torch.randn(self.num_experts, self.d_router)
        self.expert_embed = nn.Parameter(init_expert_state.clone())
        self.expert_key = nn.Parameter(init_expert_state.clone())
        self.expert_value = nn.Parameter(init_expert_state.clone())
        self._shared_expert_key_ref = None
        self.last_router_forward_stats = {}
        nn.init.zeros_(self.router_context_proj.weight)

    def set_shared_expert_key(self, expert_key: nn.Parameter) -> None:
        self.expert_key = None
        self._shared_expert_key_ref = [expert_key]

    def set_shared_expert_embed(self, expert_embed: nn.Parameter) -> None:
        self.set_shared_expert_key(expert_embed)

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

    def initialize_expert_key_from_legacy_router(self, legacy_router_weight: torch.Tensor) -> torch.Tensor:
        target = self.get_expert_key()
        reshaped = self._reshape_legacy_router_weight(legacy_router_weight, tuple(target.shape))
        reshaped = reshaped.to(dtype=target.dtype, device=target.device)
        with torch.no_grad():
            target.copy_(reshaped)
        return target

    def get_expert_embed(self) -> torch.Tensor:
        return self.expert_embed

    def get_expert_key(self) -> torch.Tensor:
        if self._shared_expert_key_ref is not None:
            return self._shared_expert_key_ref[0]
        if self.expert_key is None:
            raise RuntimeError("CrossAttentionRouter expert_key is not initialized.")
        return self.expert_key

    def get_expert_value(self) -> torch.Tensor:
        return self.expert_value

    def project_routed_expert_repr_to_key_space(self, routed_expert_repr: torch.Tensor) -> torch.Tensor:
        key_weight_t = self.key.weight.detach().float().transpose(0, 1)
        key_weight_t_pinv = torch.linalg.pinv(key_weight_t)
        return routed_expert_repr.float() @ key_weight_t_pinv

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_router_repr: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        router_in = hidden_states.to(self.query.weight.dtype)
        expert_key = self.get_expert_key()
        expert_value = self.get_expert_value()

        q = self.query(router_in).float()
        k = self.key(expert_key.to(self.key.weight.dtype)).float()
        v = self.value(expert_value.to(self.value.weight.dtype)).float()
        attn_scores = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(self.d_router)
        if self.use_entmax:
            attn_weights = entmax_bisect(attn_scores, alpha=self.alpha, dim=-1)
        else:
            softmax_scores = attn_scores
            if self.use_softmax_temperature:
                softmax_scores = softmax_scores / max(self.softmax_temperature, 1e-6)
            attn_weights = F.softmax(softmax_scores, dim=-1, dtype=torch.float32)
        router_context = torch.matmul(attn_weights, v)
        projected_context = self.router_context_proj(
            router_context.to(self.router_context_proj.weight.dtype)
        ).float()

        attn_scores_f = attn_scores.detach().float()
        route_probs_f = attn_weights.detach().float()
        router_context_f = router_context.detach().float()
        projected_context_f = projected_context.detach().float()
        projected_value_f = v.detach().float()
        projected_value_norms = projected_value_f.norm(dim=-1)
        router_context_norms = router_context_f.norm(dim=-1)
        projected_context_norms = projected_context_f.norm(dim=-1)
        hidden_norms = hidden_states.detach().float().norm(dim=-1).clamp_min(1e-12)
        router_context_delta_ratio = projected_context_norms / hidden_norms
        token_q_norms = q.detach().float().norm(dim=-1)
        expert_key_pairwise_cos_mean, expert_key_pairwise_cos_max = _pairwise_cosine_stats(expert_key)
        expert_value_pairwise_cos_mean, expert_value_pairwise_cos_max = _pairwise_cosine_stats(expert_value)
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
            "route_prob_min": float(route_probs_f.min().item()),
            "route_prob_has_neg": float(route_probs_f.lt(0).any().item()),
            "route_prob_row_sum_mean": float(row_sums.mean().item()),
            "route_prob_row_sum_abs_err": float((row_sums - 1.0).abs().mean().item()),
            "projected_value_std": float(projected_value_f.std(unbiased=False).item()),
            "projected_value_norm_mean": float(projected_value_norms.mean().item()),
            "router_context_norm_mean": float(router_context_norms.mean().item()),
            "router_context_norm_std": float(router_context_norms.std(unbiased=False).item()),
            "router_context_proj_out_mean": float(projected_context_f.mean().item()),
            "router_context_proj_out_std": float(projected_context_f.std(unbiased=False).item()),
            "router_context_delta_ratio": float(router_context_delta_ratio.mean().item()),
            "expert_key_pairwise_cos_mean": expert_key_pairwise_cos_mean,
            "expert_key_pairwise_cos_max": expert_key_pairwise_cos_max,
            "expert_value_pairwise_cos_mean": expert_value_pairwise_cos_mean,
            "expert_value_pairwise_cos_max": expert_value_pairwise_cos_max,
            "token_q_norm_mean": float(token_q_norms.mean().item()),
            "token_q_norm_std": float(token_q_norms.std(unbiased=False).item()),
        }
        router_repr = (q, k) if return_router_repr else None
        return attn_scores, attn_weights, projected_context, router_repr


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# copied from transformers.models.qwen2.modeling_qwen2.Qwen2Attention with Qwen2->Qwen2Moe
# no longer copied after attention refactors
class Qwen2MoeAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2MoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.config.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.config.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.config.qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2MoeRotaryEmbedding(config=self.config)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


# NO LONGER EXIST Copied from transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2 with Qwen2->Qwen2Moe
# TODO cyril: modular
class Qwen2MoeFlashAttention2(Qwen2MoeAttention):
    """
    Qwen2Moe flash attention module, following Qwen2Moe attention module. This module inherits from `Qwen2MoeAttention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = flash_attn_supports_top_left_mask()

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        device_type = query_states.device.type if query_states.device.type != "mps" else "cpu"
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = (
                    torch.get_autocast_dtype(device_type)
                    if hasattr(torch, "get_autocast_dtype")
                    else torch.get_autocast_gpu_dtype()
                )
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


# NO LONGER EXIST Copied from transformers.models.qwen2.modeling_qwen2.Qwen2SdpaAttention with Qwen2->Qwen2Moe
# TODO cyril: modular
class Qwen2MoeSdpaAttention(Qwen2MoeAttention):
    """
    Qwen2Moe attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2MoeAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2MoeAttention.forward
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2MoeModel is using Qwen2MoeSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = bool(causal_mask is None and q_len > 1)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None


QWEN2MOE_ATTENTION_CLASSES = {
    "eager": Qwen2MoeAttention,
    "flash_attention_2": Qwen2MoeFlashAttention2,
    "sdpa": Qwen2MoeSdpaAttention,
}


class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.use_switch = True

        self.use_low_rank_router = getattr(config, "use_low_rank_router", False)
        self.use_sharp_router = getattr(config, "use_sharp_router", True)
        self.use_cross_attention_router = getattr(config, "use_cross_attention_router", False)
        self.router_top_k = int(getattr(config, "router_top_k", self.top_k))
        self.router_use_entmax = bool(getattr(config, "router_use_entmax", False))
        self.router_entmax_alpha = float(getattr(config, "router_entmax_alpha", 1.5))
        self.router_pull_temperature = float(getattr(config, "router_pull_temperature", 1.0))
        self.router_pull_loss_type = str(getattr(config, "router_pull_loss_type", "soft"))
        self.router_ema_momentum = float(getattr(config, "router_ema_momentum", 0.99))
        self.router_use_ema_update = bool(getattr(config, "router_use_ema_update", False))
        self.last_router_aux_loss = None
        self.last_router_z_loss = None
        self.last_router_pull_loss = None
        self.last_router_budget_loss = None
        self.last_router_forward_stats = {}
        self.last_router_dispatch_stats = {}
        self.last_router_ema_stats = {}
        self._pending_router_ema_token_q = None
        self._pending_router_ema_selected_probs = None

        # gating
        if self.use_cross_attention_router:
            self.router = CrossAttentionRouter(
                hidden_size=config.hidden_size,
                num_experts=config.num_experts,
                d_router=getattr(config, "router_dim", config.hidden_size),
                use_entmax=self.router_use_entmax,
                alpha=self.router_entmax_alpha,
                use_softmax_temperature=bool(getattr(config, "router_use_softmax_temperature", True)),
                softmax_temperature=float(getattr(config, "router_softmax_temperature", 1.0)),
            )
            self.gate = None
        elif self.use_low_rank_router and self.use_sharp_router:
            self.gate = LowRankRouter(
                hidden_size=config.hidden_size,
                num_experts=config.num_experts,
                rank=config.router_rank,
                router_temperature_init=getattr(config, "router_temperature_init", 10.0),
                router_eps=getattr(config, "router_eps", 1e-6),
                normalize_q=getattr(config, "router_normalize_q", True),
                normalize_k=getattr(config, "router_normalize_k", True),
            )
        elif self.use_low_rank_router:
            self.gate = LowRankRouter(
                hidden_size=config.hidden_size,
                num_experts=config.num_experts,
                rank=config.router_rank,
                router_temperature_init=1.0,
                router_eps=getattr(config, "router_eps", 1e-6),
                normalize_q=False,
                normalize_k=False,
            )
        else:
            self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen2MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

        self.shared_expert = Qwen2MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.use_cross_attention_router:
            self.last_router_aux_loss = None
            self.last_router_z_loss = None
            self.last_router_pull_loss = None
            self.last_router_budget_loss = None
            self.last_router_forward_stats = {}
            self.last_router_dispatch_stats = {}
            self.last_router_ema_stats = {}
            self._pending_router_ema_token_q = None
            self._pending_router_ema_selected_probs = None

        if self.use_cross_attention_router:
            return self._forward_cross_attention_router(hidden_states)
        return self._forward_legacy_router(hidden_states)

    def _forward_legacy_router(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    def _compute_prototype_pull_loss(
        self,
        token_q: torch.Tensor,
        expert_axis_probs: torch.Tensor,
        expert_k: torch.Tensor,
    ) -> torch.Tensor:
        q = F.normalize(token_q.float(), dim=-1)
        k = F.normalize(expert_k.float(), dim=-1)
        temp = max(self.router_pull_temperature, 1e-6)
        sim = torch.matmul(q, k.transpose(0, 1)) / temp

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
        selected_probs: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> torch.Tensor:
        expert_axis_probs = selected_probs.new_zeros(
            selected_probs.shape[:-1] + (self.num_experts,)
        )
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
    def _ema_update_expert_key(
        self,
        token_q: torch.Tensor,
        expert_axis_probs: torch.Tensor,
    ) -> None:
        router = self.router
        expert_key_param = router.get_expert_key()
        device = expert_key_param.device
        dtype = expert_key_param.dtype

        flat_q = F.normalize(token_q.detach().float(), dim=-1).reshape(-1, token_q.size(-1))
        flat_selected_probs = expert_axis_probs.detach().float().reshape(-1, expert_axis_probs.size(-1))
        proto_sums = flat_selected_probs.transpose(0, 1) @ flat_q
        proto_counts = flat_selected_probs.sum(dim=0)
        token_counts = (flat_selected_probs > 0).sum(dim=0).to(torch.float32)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(proto_sums, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(proto_counts, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(token_counts, op=torch.distributed.ReduceOp.SUM)

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
                "ema_update_ratio": None,
                "expert_token_count_cv": float(
                    (token_counts.std(unbiased=False) / token_counts.mean().clamp_min(1e-12)).item()
                ) if token_counts.numel() > 0 else None,
            }
            return

        proto_means = proto_sums[active_mask] / proto_counts[active_mask].unsqueeze(-1)
        proto_means = F.normalize(proto_means, dim=-1)
        current_routed = router.key(expert_key_param.to(router.key.weight.dtype)).detach().float()
        current_routed = F.normalize(current_routed, dim=-1)
        old = current_routed[active_mask]
        new_routed = self.router_ema_momentum * old + (1.0 - self.router_ema_momentum) * proto_means
        new_routed = F.normalize(new_routed, dim=-1)
        proto_update_cosine = F.cosine_similarity(old, new_routed, dim=-1)
        proto_delta_norm = (new_routed - old).norm(dim=-1)
        old_key_param = expert_key_param[active_mask].detach().float()
        updated_key = router.project_routed_expert_repr_to_key_space(new_routed)
        ema_update_ratio = (
            (updated_key - old_key_param).norm(dim=-1)
            / old_key_param.norm(dim=-1).clamp_min(1e-12)
        )

        expert_key_param[active_mask].copy_(updated_key.to(device=device, dtype=dtype))
        self.last_router_ema_stats = {
            "proto_counts": proto_counts.detach().float(),
            "active_expert_count": float(active_mask.sum().item()),
            "proto_count_mean": float(proto_counts.mean().item()),
            "proto_count_min": float(proto_counts.min().item()),
            "proto_count_max": float(proto_counts.max().item()),
            "proto_update_cosine": float(proto_update_cosine.mean().item()),
            "proto_delta_norm": float(proto_delta_norm.mean().item()),
            "ema_update_ratio": float(ema_update_ratio.mean().item()),
            "expert_token_count_cv": float(
                (token_counts.std(unbiased=False) / token_counts.mean().clamp_min(1e-12)).item()
            ) if token_counts.numel() > 0 else None,
        }

    @torch.no_grad()
    def apply_pending_router_ema_update(self) -> None:
        token_q = self._pending_router_ema_token_q
        expert_axis_probs = self._pending_router_ema_selected_probs
        self._pending_router_ema_token_q = None
        self._pending_router_ema_selected_probs = None
        if token_q is None or expert_axis_probs is None:
            return
        self._ema_update_expert_key(token_q=token_q, expert_axis_probs=expert_axis_probs)

    def _forward_cross_attention_router(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        router_logits, route_probs, projected_context, router_repr = self.router(
            hidden_states,
            return_router_repr=self.training,
        )
        self.last_router_forward_stats = dict(getattr(self.router, "last_router_forward_stats", {}))
        router_logits = router_logits.float()
        route_probs = route_probs.float()
        conditioned_hidden_states = hidden_states + projected_context.to(dtype=hidden_states.dtype)

        router_q = None
        expert_k = None
        if router_repr is not None:
            router_q, expert_k = router_repr
            router_q = router_q.float()
            expert_k = expert_k.float()

        if self.training:
            self.last_router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        else:
            self.last_router_z_loss = None

        route_probs_for_stats = route_probs.float()
        soft_load = route_probs_for_stats.mean(dim=(0, 1))
        top1_experts = route_probs_for_stats.argmax(dim=-1)
        hard_load = F.one_hot(top1_experts, num_classes=self.num_experts).to(route_probs_for_stats.dtype).mean(dim=(0, 1))
        self.last_router_aux_loss = self.num_experts * torch.sum(hard_load * soft_load) if self.training else None

        topk_weights, topk_ind = top_k_routing_batched_all_sequence(
            route_probs,
            min(self.router_top_k, route_probs.size(-1)),
        )
        expert_axis_topk_weights = self._scatter_selected_probs_to_expert_axis(topk_weights, topk_ind)
        selected_expert_count_per_token = (topk_ind >= 0).sum(dim=-1)
        top2_k = min(2, route_probs_for_stats.size(-1))
        top2_values = torch.topk(route_probs_for_stats, k=top2_k, dim=-1).values
        top1_top2_margin = top2_values[..., 0] - top2_values[..., 1] if top2_k == 2 else top2_values[..., 0]
        selected_mask = topk_ind >= 0
        gathered_route_probs = route_probs_for_stats.gather(-1, topk_ind.clamp_min(0))
        topk_pre_mass = (gathered_route_probs * selected_mask.to(gathered_route_probs.dtype)).sum(dim=-1)
        topk_post_sum = topk_weights.sum(dim=-1)
        hard_counts = torch.bincount(
            topk_ind.reshape(-1)[topk_ind.reshape(-1) >= 0],
            minlength=self.num_experts,
        ).to(torch.float32)
        self.last_router_dispatch_stats = {
            "avg_selected_expert_count": float(selected_expert_count_per_token.float().mean().item()),
            "soft_selected_expert_count": float(selected_expert_count_per_token.float().mean().item()),
            "soft_load": soft_load.detach().float(),
            "hard_load": hard_load.detach().float(),
            "dead_expert_ratio": float((hard_load <= 0).float().mean().item()),
            "top1_top2_margin": float(top1_top2_margin.mean().item()),
            "topk_pre_mass_mean": float(topk_pre_mass.mean().item()),
            "topk_post_sum_mean": float(topk_post_sum.mean().item()),
            "topk_post_sum_abs_err": float((topk_post_sum - 1.0).abs().mean().item()),
            "router_top_k": float(self.router_top_k),
            "expert_token_count_cv": float(
                (hard_counts.std(unbiased=False) / hard_counts.mean().clamp_min(1e-12)).item()
            ),
        }
        self.last_router_budget_loss = hidden_states.new_zeros(()) if self.training else None

        if self.training and router_q is not None and expert_k is not None:
            self.last_router_pull_loss = self._compute_prototype_pull_loss(
                token_q=router_q,
                expert_axis_probs=expert_axis_topk_weights,
                expert_k=expert_k,
            )
        else:
            self.last_router_pull_loss = None

        if self.training and self.router_use_ema_update and router_q is not None:
            self._pending_router_ema_token_q = router_q.detach()
            self._pending_router_ema_selected_probs = expert_axis_topk_weights.detach()
        else:
            self._pending_router_ema_token_q = None
            self._pending_router_ema_selected_probs = None
            if not self.router_use_ema_update:
                self.last_router_ema_stats = {}

        flat_hidden = conditioned_hidden_states.reshape(-1, hidden_dim)
        flat_topk_weights = topk_weights.reshape(-1, topk_weights.size(-1))
        flat_topk_ind = topk_ind.reshape(-1, topk_ind.size(-1))
        output_total = torch.zeros_like(flat_hidden)
        for expert_num, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(flat_topk_ind == expert_num)
            if token_idx.numel() == 0:
                continue
            selected_hidden = flat_hidden[token_idx]
            expert_output = expert(selected_hidden)
            expert_weight = flat_topk_weights[token_idx, slot_idx].unsqueeze(-1).to(expert_output.dtype)
            output_total[token_idx] += expert_output * expert_weight

        shared_hidden = hidden_states.reshape(-1, hidden_dim)
        shared_expert_output = self.shared_expert(shared_hidden)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(shared_hidden)) * shared_expert_output
        output_total = output_total + shared_expert_output
        output_total = output_total.view(batch_size, sequence_length, hidden_dim)
        return output_total, router_logits.view(batch_size * sequence_length, self.num_experts)


class Qwen2MoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = QWEN2MOE_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen2MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen2MoeMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Cache`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


@auto_docstring
class Qwen2MoePreTrainedModel(PreTrainedModel):
    config: Qwen2MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2MoeDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen2MoeRMSNorm):
            module.weight.data.fill_(1.0)


@auto_docstring
class Qwen2MoeModel(Qwen2MoePreTrainedModel):
    def __init__(self, config: Qwen2MoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.shared_expert_key = None
        self.last_router_aux_loss = None
        self.last_router_z_loss = None
        self.last_router_pull_loss = None
        self.last_router_budget_loss = None
        self.last_router_forward_stats = {}
        self.last_router_dispatch_stats = {}
        self.last_router_ema_stats = {}
        if (
            bool(getattr(config, "share_router_expert_embedding", False))
            and bool(getattr(config, "use_cross_attention_router", False))
            and int(getattr(config, "num_experts", 0)) > 0
        ):
            d_router = int(getattr(config, "router_dim", config.hidden_size))
            self.shared_expert_key = nn.Parameter(torch.randn(config.num_experts, d_router))
            for layer in self.layers:
                mlp = getattr(layer, "mlp", None)
                if not getattr(mlp, "use_cross_attention_router", False):
                    continue
                mlp.router.set_shared_expert_key(self.shared_expert_key)
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2MoeRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> MoeModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        self.last_router_aux_loss = None
        self.last_router_z_loss = None
        self.last_router_pull_loss = None
        self.last_router_budget_loss = None
        self.last_router_forward_stats = {}
        self.last_router_dispatch_stats = {}
        self.last_router_ema_stats = {}

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

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
            if z_terms:
                self.last_router_z_loss = torch.stack(z_terms).mean()
            if pull_terms:
                self.last_router_pull_loss = torch.stack(pull_terms).mean()
            if budget_terms:
                self.last_router_budget_loss = torch.stack(budget_terms).mean()

            forward_scalar_keys = (
                "attn_scores_mean",
                "attn_scores_std",
                "attn_scores_min",
                "attn_scores_max",
                "attn_weights_row_sum_mean",
                "attn_weights_row_sum_abs_err",
                "attn_weights_entropy",
                "attn_weights_top1_mass",
                "route_prob_min",
                "route_prob_has_neg",
                "route_prob_row_sum_mean",
                "route_prob_row_sum_abs_err",
                "projected_value_std",
                "projected_value_norm_mean",
                "router_context_norm_mean",
                "router_context_norm_std",
                "router_context_proj_out_mean",
                "router_context_proj_out_std",
                "router_context_delta_ratio",
                "expert_key_pairwise_cos_mean",
                "expert_key_pairwise_cos_max",
                "expert_value_pairwise_cos_mean",
                "expert_value_pairwise_cos_max",
                "token_q_norm_mean",
                "token_q_norm_std",
            )
            if forward_stat_terms:
                self.last_router_forward_stats = {key: _mean_scalar_stat(forward_stat_terms, key) for key in forward_scalar_keys}

            dispatch_scalar_keys = (
                "avg_selected_expert_count",
                "soft_selected_expert_count",
                "dead_expert_ratio",
                "top1_top2_margin",
                "topk_pre_mass_mean",
                "topk_post_sum_mean",
                "topk_post_sum_abs_err",
                "router_top_k",
                "expert_token_count_cv",
            )
            if dispatch_stat_terms:
                self.last_router_dispatch_stats = {key: _mean_scalar_stat(dispatch_stat_terms, key) for key in dispatch_scalar_keys}
                self.last_router_dispatch_stats["soft_load"] = _mean_tensor_stat(dispatch_stat_terms, "soft_load")
                self.last_router_dispatch_stats["hard_load"] = _mean_tensor_stat(dispatch_stat_terms, "hard_load")

            ema_scalar_keys = (
                "active_expert_count",
                "proto_count_mean",
                "proto_count_min",
                "proto_count_max",
                "proto_update_cosine",
                "proto_delta_norm",
                "ema_update_ratio",
                "expert_token_count_cv",
            )
            if ema_stat_terms:
                self.last_router_ema_stats = {key: _mean_scalar_stat(ema_stat_terms, key) for key in ema_scalar_keys}
                self.last_router_ema_stats["proto_counts"] = _mean_tensor_stat(ema_stat_terms, "proto_counts")

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )

    # Copied from transformers.models.phimoe.modeling_phimoe.PhimoeModel._update_causal_mask with Phimoe->Qwen2Moe
    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2Moe. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # StaticCache
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.phimoe.modeling_phimoe.PhimoeModel._prepare_4d_causal_attention_mask_with_cache_position with Phimoe->Qwen2Moe
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2MoeConfig,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2MoeConfig`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            diagonal_attend_mask = torch.arange(target_length, device=cache_position.device) > cache_position.reshape(
                -1, 1
            )
            text_config = config.get_text_config()
            if getattr(text_config, "use_sliding_window", True) and text_config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                is_static_sliding_cache = isinstance(past_key_values, StaticCache) and all(past_key_values.is_sliding)
                if not is_static_sliding_cache or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=cache_position.device) <= (
                        cache_position.reshape(-1, 1) - text_config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class Qwen2MoeForCausalLM(Qwen2MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Qwen2MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2MoeForCausalLM

        >>> model = Qwen2MoeForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
        router_aux_loss = getattr(self.model, "last_router_aux_loss", None)
        router_z_loss = getattr(self.model, "last_router_z_loss", None)
        router_pull_loss = getattr(self.model, "last_router_pull_loss", None)
        router_budget_loss = getattr(self.model, "last_router_budget_loss", None)
        if router_aux_loss is None:
            router_aux_loss = aux_loss

        return Qwen2MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            router_aux_loss=router_aux_loss,
            router_z_loss=router_z_loss,
            router_pull_loss=router_pull_loss,
            router_budget_loss=router_budget_loss,
        )


class Qwen2MoeForSequenceClassification(GenericForSequenceClassification, Qwen2MoePreTrainedModel): ...


class Qwen2MoeForTokenClassification(GenericForTokenClassification, Qwen2MoePreTrainedModel): ...


class Qwen2MoeForQuestionAnswering(GenericForQuestionAnswering, Qwen2MoePreTrainedModel): ...


__all__ = [
    "Qwen2MoeForCausalLM",
    "Qwen2MoeForQuestionAnswering",
    "Qwen2MoeModel",
    "Qwen2MoePreTrainedModel",
    "Qwen2MoeForSequenceClassification",
    "Qwen2MoeForTokenClassification",
]
