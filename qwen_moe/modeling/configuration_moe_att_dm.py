# # coding=utf-8
# # Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# #
# # This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# # and OPT implementations in this library. It has been modified from its
# # original forms to accommodate minor architectural differences compared
# # to GPT-NeoX and OPT used by the Meta AI team that trained the model.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """ LLaMA model configuration"""

# from transformers.configuration_utils import PretrainedConfig
# from transformers.utils import logging
# logger = logging.get_logger(__name__)

# LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

# class MoEConfig(PretrainedConfig):
#     r"""
#     This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
#     model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
#     defaults will yield a similar configuration to that of the LLaMA-7B.

#     Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
#     documentation from [`PretrainedConfig`] for more information.

#     Args:
#         vocab_size (`int`, *optional*, defaults to 32000):
#             Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
#             `inputs_ids` passed when calling [`LlamaModel`]
#         hidden_size (`int`, *optional*, defaults to 4096):
#             Dimension of the hidden representations.
#         intermediate_size (`int`, *optional*, defaults to 11008):
#             Dimension of the MLP representations.
#         num_hidden_layers (`int`, *optional*, defaults to 32):
#             Number of hidden layers in the Transformer encoder.
#         num_attention_heads (`int`, *optional*, defaults to 32):
#             Number of attention heads for each attention layer in the Transformer encoder.
#         num_key_value_heads (`int`, *optional*):
#             This is the number of key_value heads that should be used to implement Grouped Query Attention. If
#             `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
#             `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
#             converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
#             by meanpooling all the original heads within that group. For more details checkout [this
#             paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
#             `num_attention_heads`.
#         pretraining_tp (`int`, *optional*, defaults to `1`):
#             Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
#             document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
#             necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
#             issue](https://github.com/pytorch/pytorch/issues/76232).
#         hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
#             The non-linear activation function (function or string) in the decoder.
#         max_position_embeddings (`int`, *optional*, defaults to 2048):
#             The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
#             Llama 2 up to 4096, CodeLlama up to 16384.
#         initializer_range (`float`, *optional*, defaults to 0.02):
#             The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
#         rms_norm_eps (`float`, *optional*, defaults to 1e-12):
#             The epsilon used by the rms normalization layers.
#         use_cache (`bool`, *optional*, defaults to `True`):
#             Whether or not the model should return the last key/values attentions (not used by all models). Only
#             relevant if `config.is_decoder=True`.
#         tie_word_embeddings(`bool`, *optional*, defaults to `False`):
#             Whether to tie weight embeddings
#         rope_theta (`float`, *optional*, defaults to 10000.0):
#             The base period of the RoPE embeddings.
#         rope_scaling (`Dict`, *optional*):
#             Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
#             strategies: linear and dynamic. Their scaling factor must be an float greater than 1. The expected format
#             is `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
#             `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
#             these scaling strategies behave:
#             https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
#             experimental feature, subject to breaking API changes in future versions.
#     """
#     model_type = "llama"
#     keys_to_ignore_at_inference = ["past_key_values"]

#     def __init__(
#         self,
#         vocab_size=32000,
#         hidden_size=4096,
#         intermediate_size=11008,
#         num_hidden_layers=32,
#         num_attention_heads=32,
#         num_key_value_heads=None,
#         hidden_act="silu",
#         max_position_embeddings=2048,
#         initializer_range=0.02,
#         rms_norm_eps=1e-6,
#         use_cache=True,
#         pad_token_id=None,
#         bos_token_id=1,
#         eos_token_id=2,
#         pretraining_tp=1,
#         tie_word_embeddings=False,
#         rope_theta=10000.0,
#         rope_scaling=None,
#         # -------- MoE 相关配置 --------
#         num_experts=-1,
#         num_null_experts=0,
#         experts_topk=4,
#         expert_frequency=2,
#         top_p_threshold=0.4,
#         # -------- AdaMOE k predictor 相关配置 --------
#         use_k_predictor=True,
#         max_k=8,
#         k_gumbel_tau=1.0,
#         allow_zero_k=False,
#         # Keep disabled by default for backward compatibility with checkpoints
#         # that only contain dense router weights (gate.weight).
#         use_low_rank_router: bool = False,
#         router_rank: int = 64,
#         use_sharp_router: bool = True,
#         router_top_k: int = 2,
#         router_temperature_init: float = 10.0,
#         router_normalize_q: bool = True,
#         router_normalize_k: bool = True,
#         router_eps: float = 1e-6,
#         **kwargs,
#     ):
#         self.vocab_size = vocab_size
#         self.max_position_embeddings = max_position_embeddings
#         self.hidden_size = hidden_size
#         self.intermediate_size = intermediate_size
#         self.num_hidden_layers = num_hidden_layers
#         self.num_attention_heads = num_attention_heads
#         # MoE / AdaMOE
#         self.num_experts = num_experts
#         self.num_null_experts = num_null_experts
#         self.expert_frequency = expert_frequency
#         self.experts_topk = experts_topk 
#         self.top_p_threshold = top_p_threshold
#         self.use_low_rank_router = use_low_rank_router
#         self.router_rank = router_rank

#         self.use_sharp_router = use_sharp_router
#         self.router_top_k = router_top_k
#         self.router_temperature_init = router_temperature_init
#         self.router_normalize_q = router_normalize_q
#         self.router_normalize_k = router_normalize_k
#         self.router_eps = router_eps

#         # k predictor：是否启用、最大 k、Gumbel-Softmax 温度
#         self.use_k_predictor = bool(use_k_predictor)
#         # 如果 max_k 未显式指定，则默认允许到所有专家（真+null）
#         total_experts = (self.num_experts if self.num_experts is not None else -1)
#         if total_experts < 0:
#             # 兼容性考虑：如果 num_experts 尚未配置好，则先退化为 0，由上层脚本覆盖
#             self.max_k = int(max_k)
#         else:
#             default_max_k = total_experts + int(self.num_null_experts)
#             self.max_k = int(max_k) if int(max_k) > 0 else int(default_max_k)
#         self.k_gumbel_tau = float(k_gumbel_tau)
#         self.allow_zero_k = bool(allow_zero_k)
        
#         # for backward compatibility
#         if num_key_value_heads is None:
#             num_key_value_heads = num_attention_heads

#         self.num_key_value_heads = num_key_value_heads
#         self.hidden_act = hidden_act
#         self.initializer_range = initializer_range
#         self.rms_norm_eps = rms_norm_eps
#         self.pretraining_tp = pretraining_tp
#         self.use_cache = use_cache
#         self.rope_theta = rope_theta
#         self.rope_scaling = rope_scaling
#         self._rope_scaling_validation()

#         super().__init__(
#             pad_token_id=pad_token_id,
#             bos_token_id=bos_token_id,
#             eos_token_id=eos_token_id,
#             tie_word_embeddings=tie_word_embeddings,
#             **kwargs,
#         )

#     def _rope_scaling_validation(self):
#         """
#         Validate the `rope_scaling` configuration.
#         """
#         if self.rope_scaling is None:
#             return

#         if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
#             raise ValueError(
#                 "`rope_scaling` must be a dictionary with with two fields, `name` and `factor`, "
#                 f"got {self.rope_scaling}"
#             )
#         rope_scaling_type = self.rope_scaling.get("type", None)
#         rope_scaling_factor = self.rope_scaling.get("factor", None)
#         if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
#             raise ValueError(
#                 f"`rope_scaling`'s name field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
#             )
#         if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
#             raise ValueError(f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}")
        

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
""" LLaMA model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
logger = logging.get_logger(__name__)

LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

class MoEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        pretraining_tp (`int`, *optional*, defaults to `1`):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
            Llama 2 up to 4096, CodeLlama up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be an float greater than 1. The expected format
            is `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
    """
    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        # -------- MoE / router (active in modeling_moe_dm.py) --------
        num_experts=-1,
        expert_frequency=2,
        router_top_k: int = 2,
        use_cross_attention_router: bool = True,
        router_dim=None,
        share_router_expert_embedding: bool = False,
        router_use_entmax: bool = False,
        router_entmax_alpha: float = 1.7,
        router_use_softmax_temperature: bool = False,
        router_softmax_temperature: float = 1.0,
        use_router_context: bool = True,
        router_context_scale: float = 1.0,
        router_anchor_momentum: float = 0.99,
        router_pull_loss_type: str = "soft",
        router_top_p: float = 0.4,
        router_budget_target_count: float = 0.0,
        router_budget_tau: float = 0.05,
        moe_att_use_detach: bool = True,
        moe_att_expert_loss_coef: float = 1.0,
        moe_att_kl_loss_coef: float = 1.0,
        moe_att_eps: float = 1e-9,
        # -------- legacy router params (unused by current simplified router) --------
        # These are intentionally kept as comments for backward compatibility context.
        # experts_topk=2,
        # top_p_threshold=0.4,
        # use_low_rank_router: bool = False,
        # router_rank: int = 64,
        # use_sharp_router: bool = True,
        # router_dim: int = None,
        # router_value_dim: int = None,
        # router_temperature_init: float = 1.0,
        # router_normalize_q: bool = True,
        # router_normalize_k: bool = True,
        # router_use_scale: bool = True,
        # router_dropout: float = 0.0,
        # router_eps: float = 1e-6,
        # router_use_entmax: bool = True,
        # router_entmax_alpha: float = 1.7,
        # router_min_prob: float = 0.0,
        # use_router_context: bool = True,
        # router_context_mode: str = "add",
        # detach_router_context_probs: bool = False,
        # use_value_aware_routing: bool = False,
        # value_norm_type: str = "l1",
        # value_norm_detach: bool = False,
        # value_norm_scale: float = 1.0,

        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        # -------- MoE / router (active in modeling_moe_dm.py) --------
        self.num_experts = int(num_experts)
        self.expert_frequency = int(expert_frequency)
        self.router_top_k = int(router_top_k)
        self.router_dim = int(hidden_size if router_dim is None else router_dim)
        self.share_router_expert_embedding = bool(share_router_expert_embedding)
        self.router_use_entmax = bool(router_use_entmax)
        self.router_entmax_alpha = float(router_entmax_alpha)
        self.router_use_softmax_temperature = bool(router_use_softmax_temperature)
        self.router_softmax_temperature = float(router_softmax_temperature)
        self.use_router_context = bool(use_router_context)
        self.router_context_scale = float(router_context_scale)
        self.router_anchor_momentum = float(router_anchor_momentum)
        self.router_pull_loss_type = str(router_pull_loss_type)
        self.router_budget_target_count = float(router_budget_target_count)
        self.router_budget_tau = float(router_budget_tau)
        self.moe_att_use_detach = bool(moe_att_use_detach)
        self.moe_att_expert_loss_coef = float(moe_att_expert_loss_coef)
        self.moe_att_kl_loss_coef = float(moe_att_kl_loss_coef)
        self.moe_att_eps = float(moe_att_eps)

        # Legacy compatibility: old checkpoints may only have `use_low_rank_router`.
        legacy_use_low_rank_router = bool(kwargs.get("use_low_rank_router", False))
        self.use_low_rank_router = legacy_use_low_rank_router
        self.use_cross_attention_router = bool(use_cross_attention_router or legacy_use_low_rank_router)

        # -------- legacy router params (kept for checkpoint/config compatibility) --------
        self.experts_topk = int(kwargs.get("experts_topk", 2))
        self.top_p_threshold = float(kwargs.get("top_p_threshold", router_top_p))

        self.ensure_model_attributes()

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def ensure_model_attributes(self):
        """Backfill and validate model fields consumed by modeling_moe_dm.py."""
        self.num_experts = int(getattr(self, "num_experts", -1))
        self.expert_frequency = int(getattr(self, "expert_frequency", 2))
        self.router_top_k = int(getattr(self, "router_top_k", 2))
        self.router_dim = int(getattr(self, "router_dim", self.hidden_size))
        self.share_router_expert_embedding = bool(getattr(self, "share_router_expert_embedding", False))
        self.router_use_entmax = bool(getattr(self, "router_use_entmax", False))
        self.router_entmax_alpha = float(getattr(self, "router_entmax_alpha", 1.5))
        self.router_use_softmax_temperature = bool(getattr(self, "router_use_softmax_temperature", False))
        self.router_softmax_temperature = float(getattr(self, "router_softmax_temperature", 1.0))
        self.use_router_context = bool(getattr(self, "use_router_context", True))
        self.router_context_scale = float(getattr(self, "router_context_scale", 1.0))
        self.router_anchor_momentum = float(getattr(self, "router_anchor_momentum", 0.99))
        self.router_pull_loss_type = str(getattr(self, "router_pull_loss_type", "soft"))
        self.router_budget_target_count = float(getattr(self, "router_budget_target_count", 0.0))
        self.router_budget_tau = float(getattr(self, "router_budget_tau", 0.05))
        self.moe_att_use_detach = bool(getattr(self, "moe_att_use_detach", True))
        self.moe_att_expert_loss_coef = float(getattr(self, "moe_att_expert_loss_coef", 1.0))
        self.moe_att_kl_loss_coef = float(getattr(self, "moe_att_kl_loss_coef", 1.0))
        self.moe_att_eps = float(getattr(self, "moe_att_eps", 1e-9))
        # `top_p_threshold` is kept only so older configs/checkpoints still load.
        self.top_p_threshold = float(getattr(self, "top_p_threshold", 0.4))

        legacy_use_low_rank_router = bool(getattr(self, "use_low_rank_router", False))
        self.use_cross_attention_router = bool(
            getattr(self, "use_cross_attention_router", True) or legacy_use_low_rank_router
        )

        if self.expert_frequency <= 0:
            raise ValueError(f"`expert_frequency` must be >= 1, got {self.expert_frequency}")
        if self.router_top_k <= 0:
            raise ValueError(f"`router_top_k` must be >= 1, got {self.router_top_k}")
        if self.router_dim <= 0:
            raise ValueError(f"`router_dim` must be >= 1, got {self.router_dim}")
        if self.num_experts > 0 and self.router_top_k > self.num_experts:
            raise ValueError(
                f"`router_top_k` ({self.router_top_k}) cannot exceed `num_experts` ({self.num_experts})"
            )
        if self.router_use_entmax and not (1.0 < self.router_entmax_alpha <= 2.0):
            raise ValueError(
                f"`router_entmax_alpha` must be in (1, 2] when `router_use_entmax=True`, "
                f"got {self.router_entmax_alpha}"
            )
        if self.router_softmax_temperature <= 0.0:
            raise ValueError(
                f"`router_softmax_temperature` must be > 0, got {self.router_softmax_temperature}"
            )
        if self.router_context_scale < 0.0:
            raise ValueError(f"`router_context_scale` must be >= 0, got {self.router_context_scale}")
        if not (0.0 <= self.router_anchor_momentum < 1.0):
            raise ValueError(
                f"`router_anchor_momentum` must be in [0, 1), got {self.router_anchor_momentum}"
            )
        if self.router_pull_loss_type not in {"soft", "hard_ce"}:
            raise ValueError(
                f"`router_pull_loss_type` must be one of {{'soft', 'hard_ce'}}, "
                f"got {self.router_pull_loss_type!r}"
            )
        if self.router_budget_target_count < 0.0:
            raise ValueError(
                f"`router_budget_target_count` must be >= 0, got {self.router_budget_target_count}"
            )
        if self.router_budget_tau <= 0.0:
            raise ValueError(f"`router_budget_tau` must be > 0, got {self.router_budget_tau}")
        if self.moe_att_expert_loss_coef < 0.0:
            raise ValueError(
                f"`moe_att_expert_loss_coef` must be >= 0, got {self.moe_att_expert_loss_coef}"
            )
        if self.moe_att_kl_loss_coef < 0.0:
            raise ValueError(
                f"`moe_att_kl_loss_coef` must be >= 0, got {self.moe_att_kl_loss_coef}"
            )
        if self.moe_att_eps <= 0.0:
            raise ValueError(f"`moe_att_eps` must be > 0, got {self.moe_att_eps}")

        # Legacy params kept for checkpoint compatibility (unused by current simplified router):
        # router_rank, router_dim, router_value_dim, router_temperature_init, router_normalize_q,
        # router_normalize_k, router_use_scale, router_dropout, router_eps, router_use_entmax,
        # router_entmax_alpha, router_min_prob, use_router_context, router_context_mode,
        # detach_router_context_probs, use_value_aware_routing, value_norm_type, value_norm_detach,
        # value_norm_scale, use_sharp_router, experts_topk, top_p_threshold.
        return self

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `name` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s name field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}")
