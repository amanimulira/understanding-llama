import math
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama4 import Llama4VisionConfig

from activations import ACT2FN
from cache_utils import Cache, DynamicCache
from generation import GenerationMixin
from integrations.hub_kernels import use_kernel_forward_from_hub
from masking_utils import create_causal_mask, create_chunked_causal_mask
from modeling_flash_attention_utils import FlashAttentionKwargs
from modeling_layers import GradientCheckpointingLayer
from modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, CausalLMoutputWithPast, ModelOutput
from modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from processing_utils import Unpack
from utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from configuration_llama4 import Llama4Config, Llama4TextConfig

logger = logging.get_logger(__name__)

"""
mixture of experts layer

instead of having a single feed-forward network, there are multiple "expert" networks.

a "gate" mechanism then determines which expert(s) should process each input.

improving model capacity and efficiency.

"""
class Llama4TextExperts(nn.Module):
	def __init__(self, config: Llama4TextConfig):
		super().__init__()
		self.num_experts = config.num_local_experts
		self.intermediate_size = config.intermediate_size
		self.hidden_size = config.hidden_size
		self.expert_dim = self.intermediate_size
		self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
		self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
		self.act_fn = ACT2FN[config.hidden_act]
        
	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		# (batch_size * sequence_length, hidden_size) -> sequence dims are flattened.
		# (num_tokens, hidden_size), where num_tokens is the total number of tokens (batch_size * sequence_length)
		
		# reshapes input, which distributes the tokens among the experts.
		# first dim explicitly represents different experts.
		# -1 pytorch automatically calculates this dim, will be num_tokens / num_experts.
		# each num_tokens_per_expert chunk of the orignial hidden_states is conceptually assigned to a specific expert.
		# hidden_size - feature dim remains the same.

		# This effectively "assigns" groups of input tokens to different experts.
		# Performs a simple block wise assingment.

		hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)

		# hidden_states -> (self.num_experts, total_tokens_per_expert, self.hidden_size)
		# self.gate_proj -> (self.num_experts, self.hidden_size, 2 * self.expert_dim) 

		# So then torch.bmm performs batch matrix multiplication, iterating thorugh the first dim.
		# Performs matrix multipliatoin for each corresponding pair of "batches"
		# gate_up -> (total_tokens_per_expert, 2 * expert_dim)

		# This step applies the initial linear projection for both the "gate" and the "up projection" 
		# components independently for each expert. We are going from hidden_size to 2 * expert_dim.
		gate_up = torch.bmm(hidden_states, self.gate_up_proj)

		# Spliting the gate_up tensor into two equal-sized chunks along the last dim.

		# gate shape: (self.num_experts, total_tokens_per_expert, self.expert_dim)
		# up shape: (self.num_experts, total_tokens_per_expert, self.expert_dim)

		# Seperates the comnined projection into two distinct tensors

		# gate, will control activation
		# up projection, main transformed features
		gate, up = gate_up.chunk(2, dim=-1) # not supported for DTensors

		# self.act_fn(gate): applies non-linear activation funciotn element wise to the gate tensor. (shape preserved)
		# up * self.act_fn(gate): element-wise multiplication (Hadamard product) between up and activated gate tensor.
		# CORE OF GATING MECHANISM.

		# self.down_proj Shape: (self.num_experts, self.expert_dim, self.hidden_size)

		# torch.bmm, for each expert i:
		# intermediate result: (up * self.act_fn(gate))[i]
		# the shape -> (total_tokens_per_expert, expert_dim)

		# self.down_proj[i] (shape expert_dim, hidden_size)

		# Completes individual expert compution: the modulated features are transformed back to the model's hidden_size
		# by the expert's specific down_proj weights.
		next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)

		# (self.num_experts, total_tokens_per_expert, self.hidden_size) -> (total_tokens, self.hidden_size)

		# Restores the output to the original "flattened tokens" shape. 

		# Effectively combines the outputs from all experts back into a single sequence of hidden states.
		next_states = next_states.view(-1, self.hidden_size)

		return next_states
	
class Llama4TextMLP(nn.Module):
	def __init__(self, config, intermediate_size=None):
		super().__init__()

		if intermediate_size is None:
			intermediate_size = config.intermediate_size

		self.config = config
		self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
		self.activation_fn = ACT2FN[config.hidden_act]

	def forward(self, x):
		down_proj = self.activation_fn(self.gate_proj(x)) * self.up_proj(x)
		return self.down_proj(down_proj)
	
class Llama4TextL2Norm(torch.nn.Module):
	def __init__(self, eps: float = 1e-6):
		super().__init__()
		self.eps = eps
	
	def _norm(self, x):
		return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
	
	def forward(self, x):
		return self._norm(x.float()).type_as(x)
	
	def extra_repr(self):
		return f"eps={self.eps}"
	
class Llama4TextRMSNorm(nn.Module):
	def __init__(self, hidden_size, eps=1e-5):
		super().__init__()

		self.eps = eps
		self.weight = nn.Parameter(torch.one(hidden_size))

	def _norm(self, x):
		return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
	
	def forward(self, x):
		output = self._norm(x.float()).type_as(x)
		return output * self.weight
	
	def extra_pepr(self):
		return f"{tuple(self.weight.shape)}, eps={self.eps}"

@use_kernel_forward_from_hub("Llama4TextMoe")
class Llama4TextMoe(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.top_k = config.num_experts_per_tok
		self.hidden_dim = config.hidden_size
		self.num_experts = config.num_local_experts
		self.experts = Llama4TextExperts(config)
		self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
		self.shared_expert = Llama4TextMLP(config)

	def forward(self, hidden_states):
		hidden_states = hidden_states.reshape(-1, self.hidden_dim)
		router_logits = self.router(hidden_states)

		router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)

		router_scores = (
			torch.full_like(router_logits, float("-inf")).scatter_(1, router_indices, router_top_value).transpose(0, 1)
		)
		router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

		routed_in = hidden_states.repeat(self.num_experts, 1)
		routed_in = routed_in * router_scores.reshape(-1, 1)
		routed_out = self.experts(routed_in)

		out = self.shared_expert(hidden_states)
		out.add(routed_out.reshape(self.num_experts, -1, self.hidden_dim).sum(dim=0))

		return out, router_scores

class Llama4TextRotaryEmbedding(nn.Module):
	def __init__(self, config: Llama4TextConfig, device=None):
		super().__init__()
		self.rope_type = "llama3" if config.rope_scaling is not None else "default"

		self.max_seq_len_cached = config.max_position_embeddings
		self.original_max_seq_len = config.max_position_embeddings

		self.config = config
		self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

		inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
		self.register_buffer("inv_freq", inv_freq, persistent=False)
		self.original_inv_freq = self.inv_freq

	@torch.no_grad()
	@dynamic_rope_update
	def forward(self, x, position_ids):
		inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
		position_ids_expanded = position_ids[:, None, :].float()

		device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
		with torch.autocast(device_type=device_type, enabled=False):
			freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
			freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
			freqs_cis = freqs_cis * self.attention_scaling

		return freqs_cis
	
def apply_rotary_emb(
		xq: torch.Tensor, 
		xk: torch.Tensor, 
		freq_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
	xk_ = torch.view_as_complex(xk.float().reshape(*xq.shape[:-1], -1, 2))

	xq_out = torch.view_as_real(xq_ * freq_cis[:, :, None, :]).flatten(3)
	xk_out = torch.view_as_real(xk_ * freq_cis[:, :, None, :]).flatten(3)

	return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
	"""
	like torch.repeat_interleave. Hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
	"""
	batch, num_key_value_heads, slen, head_dim = hidden_states.shape
	if n_rep == 1:
		return hidden_states
	hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
	return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
		module: nn.Module,
		query: torch.Tensor,
		key: torch.Tensor, 
		value: torch.Tensor, 
		attention_mask: Optional[torch.Tensor],
		scaling: float,
		dropout: float = 0.0,
		**kwargs,
):
	
	key_states = repeat_kv(key, module.num_key_value_groups)
	value_states = repeat_kv(value, module.num_key_value_groups)

	attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
	if attention_mask is not None:
		causal_mask = attention_mask[:, :, :, :, key_states.shape[-2]]
		attn_weights = attn_weights + causal_mask

	attn_weights = nn.functional.softmax(attn_weights, dim=-1)
	attn_weights = nn.funcitonal.dropout(attn_weights, p=dropout, training=module.training)
	attn_output = torch.matmul(attn_weights, value_states)
	attn_output = attn_output.transpose(1, 2).contiguous()

	return attn_output, attn_weights

def vision_eager_attention_forward(
		module: nn.Module, 
		query: torch.Tensor, 
		key: torch.Tensor, 
		value: torch.Tensor, 
		attention_mask: Optional[torch.Tensor],
		scaling: float, 
		dropout: float = 0.0,
		**kwargs,
):
	key_states = repeat_kv(key, module.num_key_value_groups)
	value_states = repeat_kv(value, module.num_key_value_groups)

	attn_weights = torch.matmul(key_states, query)
	if attention_mask is not None:
		causal_mask = attention_mask[:, :, :, :, key_states.shape[-2]]
		attn_weights = attn_weights + causal_mask
	
	attn_weights = nn.funcitonal.softmax(attn_weights, dim=-1)
	attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

	attn_output = torch.matmul(attn_weights, value_states)
	attn_output = attn_output.transpose(1, 2).contiguous()

	return attn_output, attn_weights

class Llama4TextAttention(nn.Module):

	def __init__(self, config: Llama4TextConfig, layer_idx):
		super().__init__()
		self.config = config
		self.layer_idx = layer_idx
		self.head_dim = getattr(config, "haed_dim", config.hidden_size // config.num_attention_heads)
		self.num_attention_heads = config.num_attention_heads
		self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
		self.num_key_vlaue_heads = config.num_key_value_heads
		self.scaling = self.head_dim**-0.5
		self.attn_scale = config.attn_scale 
		self.floor_scale = config.floor_scale
		self.attn_temperature_tuning = config.attn_temperature_tuning
		self.attention_dropout = config.attention_dropout 
		self.is_causal = True
		self.use_rope = config.no_rope_layers[layer_idx]
		self.q_proj = nn.Linear(
			config.hidden_size, config.num_attention_heads * config.head_dim, bias=config.attention_bias
		)
		self.k_proj = nn.Linear(
			config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
		)
		self.v_proj = nn.Linear(
			config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
		)
		self.o_proj = nn.Linear(
			config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
		)

		if self.config.use_qk_norm and self.use_rope:
			self.qk_norm = Llama4TextL2Norm(config.rms_norm_eps)
		
	def forward(
			self, 
			hidden_states: torch.Tensor,
			position_embeddings: tuple[torch.Tensor, torch.Tensor],
			attention_mask: Optional[torch.Tensor],
			past_key_value: Optional[Cache] = None,
			cache_position: Optional[torch.LongTensor] = None, 
			**kwargs: Unpack[FlashAttentionKwargs]
	) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
		input_shape = hidden_states.shape[:-1]
		hidden_shape = (*input_shape, -1, self.head_dim)

		query_states = self.q_proj(hidden_states).view(hidden_shape)
		key_states = self.k_proj(hidden_states).view(*input_shape, -1, self.head_dim)
		value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

		if self.use_rope:
			query_states, key_states = apply_rotary_emb(
				query_states, key_states, position_embeddings.to(query_states.device)
			)

		if hasattr(self, "qk_norm"):
			query_states = self.qk_norm(query_states)
			key_states = self.qk_norm(key_states)

		# Temperature tuning for the NoROPE layers
		if self.attn_temperature_tuning and not self.use_rope:
			attn_scales = (
				torch.log(torch.floor((cache_position.float() + 1.0) / self.floor_scale) + 1.0) * self.attn_scale + 1.0
			)
			attn_scales = attn_scales.view((1, input_shape[-1], 1, 1)).expand((*input_shape, 1, 1)) # batch size > 1
			query_states = (query_states * attn_scales).to(query_states.dtype)

		query_states = query_states.transpose(1, 2)
		key_states = key_states.transpose(1, 2)

		if past_key_value is not None:
			# sin and cos are spcific to RoPE models; cache_position needed for the static cache
			cache_kwargs = {"cache_position": cache_position}
			key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

		attnetion_interface: Callable = eager_attention_forward
		if self.config._attn_implementation != "eager":
			attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
		attn_output, attn_weights = attention_interface(
			self, 
			query_states, 
			key_states,
			value_states,
			attention_mask,
			dropout=0.0 if not self.training else self.attention_dropout,
			scaling=self.scaling,
			**kwargs
		)

		attn_output = attn_output.reshape(*input_shape, -1).contiguous()
		attn_output = attn_output.o_proj(attn_output)

		return attn_output, attn_weights








