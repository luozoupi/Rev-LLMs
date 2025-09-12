"""
Reversible Qwen3 with Modern PyTorch AMP and Optimized Training - IMPROVED VERSION
===================================================================================

This module integrates reversible mechanisms from the Reformer with Qwen3 
candidate selection attention using modern PyTorch APIs for better convergence and efficiency.

Key Improvements:
- Modern torch.amp API (torch.amp.GradScaler instead of torch.cuda.amp.GradScaler)
- Better numerical stability with proper dtype handling
- Optimized gradient accumulation and scaling
- Improved memory efficiency for reversible blocks
- Enhanced initialization strategies
- Better learning rate scheduling support
"""
import math
import warnings
try:
    from native_sparse_attention_pytorch import SparseAttention
    NATIVE_SPARSE_AVAILABLE = True
except Exception:
    NATIVE_SPARSE_AVAILABLE = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states
from typing import Optional, Tuple, List, Dict, Any
from torch import Tensor
import candi_sel_new_v2 as candi_sel_new

# Import from existing modules
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config, 
    Qwen3RMSNorm, 
    repeat_kv, 
    apply_rotary_pos_emb,
    Qwen3MLP
)
from transformers.cache_utils import Cache

# Modern rotary embedding import with fallback
try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    QWEN3_ROTARY_AVAILABLE = True
except Exception:
    Qwen3RotaryEmbedding = None
    QWEN3_ROTARY_AVAILABLE = False

class ModernNativeSparseAttentionWrapper(nn.Module):
    """
    Modern wrapper for native sparse attention with improved memory management
    and proper AMP support
    """
    def __init__(self, config, layer_idx: int, inner_dim: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.inner_dim = inner_dim
        
        if NATIVE_SPARSE_AVAILABLE:
            try:
                self.sparse_attention = SparseAttention(
                    dim=inner_dim,
                    dim_head=config.hidden_size // config.num_attention_heads,
                    heads=config.num_attention_heads,
                    kv_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
                    causal=True,
                    sliding_window_size=getattr(config, 'sliding_window_size', 64),
                    compress_block_size=getattr(config, 'compress_block_size', 16),
                    compress_block_sliding_stride=getattr(config, 'compress_block_sliding_stride', 8),
                    selection_block_size=getattr(config, 'selection_block_size', 16),
                    num_selected_blocks=getattr(config, 'num_selected_blocks', 4),
                    use_diff_topk=getattr(config, 'use_diff_topk', True),
                    query_heads_share_selected_kv=getattr(config, 'query_heads_share_selected_kv', True),
                    use_triton_kernel=getattr(config, 'use_triton_kernel', True),
                )
                self.native_sparse_available = True
            except Exception as e:
                print(f"Warning: Native sparse attention initialization failed: {e}")
                self.native_sparse_available = False
        else:
            self.native_sparse_available = False
    
    def forward(self, x):
        if not self.native_sparse_available:
            return x  # Pass through if not available
        
        try:
            # Use modern autocast API
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                attended_output = self.sparse_attention(x)
            return attended_output
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Fallback: use float16 precision temporarily
                print(f"OOM in native sparse attention, trying float16: {e}")
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available(), dtype=torch.float16):
                    attended_output = self.sparse_attention(x.half()).float()
                return attended_output
            else:
                raise e

class ImprovedRotaryEmbedding(nn.Module):
    """Improved rotary embedding with better numerical stability"""
    
    def __init__(self, dim, max_position_embeddings=8192, base=10000, device=None, dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequency components with better numerical stability
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq.to(dtype=dtype))
        
        # Cache for position embeddings
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
            
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = max(seq_len, self.max_position_embeddings)
            
            t = torch.arange(self._seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            
            # Use more stable computation
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self._cos_cached = emb.cos().to(x.dtype)
            self._sin_cached = emb.sin().to(x.dtype)
        
        return (
            self._cos_cached[:seq_len].to(x.device),
            self._sin_cached[:seq_len].to(x.device)
        )

class DeterministicFunction(nn.Module):
    """Improved deterministic wrapper for better gradient flow"""
    
    def __init__(self, fn, args_map=None):
        super().__init__()
        self.fn = fn
        self.args_map = args_map or {}
        
    def forward(self, *args, **kwargs):
        # Use mixed precision if available
        if torch.is_autocast_enabled():
            return self.fn(*args, **kwargs)
        else:
            # Ensure consistent dtype
            device = args[0].device if args else next(iter(kwargs.values())).device
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                return self.fn(*args, **kwargs)

class ImprovedReversibleQwen3CandidateAttention(nn.Module):
    """
    Improved Qwen3 Multi-type Attention with modern PyTorch APIs and better convergence
    """
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Enhanced head configuration with validation
        self.num_attention_heads = max(1, getattr(config, 'num_attention_heads', 8))
        
        if hasattr(config, 'num_key_value_heads') and config.num_key_value_heads is not None and config.num_key_value_heads > 0:
            self.num_key_value_heads = max(1, config.num_key_value_heads)
        else:
            self.num_key_value_heads = self.num_attention_heads
        
        # Ensure valid head grouping
        if self.num_attention_heads % self.num_key_value_heads != 0:
            print(f"Warning: Adjusting head configuration for layer {layer_idx}")
            # Find largest valid divisor
            for divisor in [self.num_key_value_heads, self.num_key_value_heads // 2, 1]:
                if divisor > 0 and self.num_attention_heads % divisor == 0:
                    self.num_key_value_heads = divisor
                    break
        
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_attention_heads)
        self.attention_dropout = getattr(config, 'attention_dropout', 0.0)
        self.scaling = self.head_dim ** -0.5
        
        # Attention type configuration
        self.attention_type = getattr(config, 'attention_type', 'standard')
        
        # Input dimension for reversible networks (full hidden_size after doubling)
        input_dim = self.hidden_size
        
        # Initialize attention components based on type
        if self.attention_type == "native_sparse" and NATIVE_SPARSE_AVAILABLE:
            try:
                self.sparse_attention_wrapper = ModernNativeSparseAttentionWrapper(
                    config, layer_idx, input_dim
                )
                self.use_native_sparse = True
            except Exception as e:
                print(f"Failed to initialize native sparse attention: {e}")
                self.use_native_sparse = False
                self.attention_type = "standard"
        else:
            self.use_native_sparse = False
        
        # Standard/candidate selection attention components
        if not self.use_native_sparse:
            # Projection layers with improved initialization
            q_proj_dim = self.num_attention_heads * self.head_dim
            kv_proj_dim = self.num_key_value_heads * self.head_dim
            
            self.q_proj = nn.Linear(input_dim, q_proj_dim, bias=getattr(config, 'attention_bias', False))
            self.k_proj = nn.Linear(input_dim, kv_proj_dim, bias=getattr(config, 'attention_bias', False))
            self.v_proj = nn.Linear(input_dim, kv_proj_dim, bias=getattr(config, 'attention_bias', False))
            self.o_proj = nn.Linear(q_proj_dim, input_dim, bias=getattr(config, 'attention_bias', False))
            
            # Improved initialization for better convergence
            self._init_weights()
            
            # RMS normalization layers
            rms_eps = getattr(config, 'rms_norm_eps', 1e-6)
            self.q_norm = Qwen3RMSNorm(self.head_dim, eps=rms_eps)
            self.k_norm = Qwen3RMSNorm(self.head_dim, eps=rms_eps)
            
            # Candidate selection parameters
            if self.attention_type == "candidate_selection":
                self.pr_ratio = getattr(config, 'candidate_pr_ratio', 0.5)
                self.top_k = getattr(config, 'candidate_top_k', 40)
        
        # Sliding window support
        self.sliding_window = None
        if hasattr(config, 'layer_types') and layer_idx < len(config.layer_types):
            if config.layer_types[layer_idx] == "sliding_attention":
                self.sliding_window = getattr(config, 'sliding_window', None)
    
    def _init_weights(self):
        """Improved weight initialization for better convergence"""
        
        # Xavier/Glorot initialization for projections
        for module in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        # Special initialization for output projection (slightly smaller scale)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=0.5)
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)
    
    def forward(
        self, 
        x, 
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """
        Enhanced forward pass with modern mixed precision support
        """
        # Validate input dimensions
        batch_size, seq_len, input_dim = x.shape
        
        if self.use_native_sparse:
            return self._forward_native_sparse(x, position_embeddings, attention_mask, past_key_values, cache_position)
        else:
            return self._forward_manual_attention(x, position_embeddings, attention_mask, past_key_values, cache_position)
    
    def _forward_native_sparse(
        self, 
        x, 
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        """Enhanced native sparse attention forward pass"""
        
        if past_key_values is not None:
            warnings.warn("KV cache not fully supported with native sparse attention yet")
        
        if position_embeddings is not None:
            warnings.warn("Position embeddings integration with native sparse attention needs refinement")
        
        # Use modern sparse attention wrapper
        return self.sparse_attention_wrapper(x)
    
    def _forward_manual_attention(
        self, 
        x, 
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        """
        Enhanced manual attention computation with improved numerical stability
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Project to Q, K, V with proper dtype handling
        query_proj = self.q_proj(x)
        key_proj = self.k_proj(x)
        value_proj = self.v_proj(x)
        
        # Reshape with validation
        expected_q_dim = self.num_attention_heads * self.head_dim
        expected_kv_dim = self.num_key_value_heads * self.head_dim
        
        if query_proj.shape[-1] != expected_q_dim or key_proj.shape[-1] != expected_kv_dim:
            raise ValueError(f"Projection dimension mismatch: Q={query_proj.shape[-1]} (expected {expected_q_dim}), "
                           f"K={key_proj.shape[-1]} (expected {expected_kv_dim})")
        
        query_states = query_proj.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        key_states = key_proj.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_proj.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        
        # Apply RMS normalization with proper dtype preservation
        original_dtype = query_states.dtype
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        
        # Ensure dtype consistency
        query_states = query_states.to(original_dtype)
        key_states = key_states.to(original_dtype)
        
        # Transpose to [batch, num_heads, seq_len, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # Apply rotary position embeddings with improved handling
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # Ensure position embeddings are on the correct device and dtype
            cos = cos.to(device=query_states.device, dtype=query_states.dtype)
            sin = sin.to(device=query_states.device, dtype=query_states.dtype)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Handle KV cache with improved integration
        if past_key_values is not None:
            cache_kwargs = {}
            if position_embeddings is not None:
                cos, sin = position_embeddings
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        # Repeat KV heads for grouped query attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention scores with improved numerical stability
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        
        # Apply scaling with better precision handling
        if attn_weights.dtype == torch.float16:
            # Use float32 for scaling to avoid overflow
            attn_weights = (attn_weights.float() * self.scaling).half()
        else:
            attn_weights = attn_weights * self.scaling
        
        # Apply causal mask with proper broadcasting
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
            elif attention_mask.dim() == 2:
                # Convert 2D mask to 4D
                causal_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                causal_mask = causal_mask.expand(batch_size, 1, seq_len, key_states.shape[-2])
                attn_weights = attn_weights + causal_mask
        
        # Apply sparsity for candidate selection
        if self.attention_type == "candidate_selection":
            extended_attention_mask = self._prepare_attention_mask(attention_mask, batch_size, seq_len, key_states.shape[-2])
            
            try:
                # Apply candidate selection with error handling
                S_prune_mask = candi_sel_new.QK_prune_binary_quant(
                    query_states, key_states, extended_attention_mask,
                    pr_ratio=self.pr_ratio, top_k=self.top_k
                )
                attn_weights = attn_weights + S_prune_mask
            except Exception as e:
                warnings.warn(f"Candidate selection failed: {e}, using standard attention")
        
        # Softmax with improved numerical stability
        if attn_weights.dtype == torch.float16:
            # Compute in float32 for stability
            attn_weights = F.softmax(attn_weights.float(), dim=-1).to(attn_weights.dtype)
        else:
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply dropout
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # Compute final output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output
    
    def _prepare_attention_mask(self, attention_mask, batch_size, seq_len, key_len):
        """Prepare attention mask for candidate selection"""
        if attention_mask is None:
            return None
        
        if attention_mask.dim() == 4:
            return attention_mask
        elif attention_mask.dim() == 2:
            # Convert 2D mask to 4D format expected by candidate selection
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = extended_mask.expand(batch_size, 1, seq_len, key_len)
            # Convert to additive mask format (0 for valid, -inf for masked)
            extended_mask = (1.0 - extended_mask) * -10000.0
            return extended_mask
        else:
            return attention_mask

class ModernReversibleFunction(Function):
    """
    Improved reversible function with modern PyTorch APIs and better memory management
    """
    
    @staticmethod
    def forward(ctx, x, f, g, f_args=(), g_args=(), f_kwargs=None, g_kwargs=None):
        """
        Enhanced forward pass with better error handling and memory efficiency
        """
        f_kwargs = f_kwargs or {}
        g_kwargs = g_kwargs or {}
        
        ctx.f = f
        ctx.g = g
        ctx.f_args = f_args
        ctx.g_args = g_args
        ctx.f_kwargs = f_kwargs
        ctx.g_kwargs = g_kwargs
        
        # Split input more safely
        x1, x2 = torch.chunk(x, 2, dim=-1)
        
        # Store for backward pass with better memory management
        ctx.save_for_backward(x1.detach(), x2.detach())
        
        # Forward computation with mixed precision support
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            try:
                y1 = x1 + f(x2, *f_args, **f_kwargs)
                y2 = x2 + g(y1, *g_args, **g_kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Fallback to CPU computation if OOM
                    print(f"OOM in reversible forward, falling back to CPU: {e}")
                    x1_cpu = x1.cpu()
                    x2_cpu = x2.cpu()
                    y1 = x1_cpu + f(x2_cpu, *f_args, **f_kwargs).cpu()
                    y2 = x2_cpu + g(y1, *g_args, **g_kwargs).cpu()
                    y1 = y1.to(x.device)
                    y2 = y2.to(x.device)
                else:
                    raise e
        
        return torch.cat([y1, y2], dim=-1)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Enhanced backward pass with improved gradient computation
        """
        f, g = ctx.f, ctx.g
        f_args, g_args = ctx.f_args, ctx.g_args
        f_kwargs, g_kwargs = ctx.f_kwargs, ctx.g_kwargs
        x1, x2 = ctx.saved_tensors
        
        # Split gradient
        grad_y1, grad_y2 = torch.chunk(grad_output, 2, dim=-1)
        
        # Reconstruct forward states for gradient computation
        with torch.enable_grad():
            x1.requires_grad_(True)
            x2.requires_grad_(True)
            
            # Forward pass reconstruction with autocast
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                y1 = x1 + f(x2, *f_args, **f_kwargs)
                y2 = x2 + g(y1, *g_args, **g_kwargs)
            
            # Backward reconstruction with improved stability
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                # Compute gradients for g function
                if y1.requires_grad:
                    grad_g_y1 = torch.autograd.grad(
                        outputs=y2, inputs=y1, grad_outputs=grad_y2,
                        retain_graph=True, only_inputs=True
                    )[0]
                else:
                    grad_g_y1 = torch.zeros_like(y1)
                
                # Update gradient for y1
                grad_y1_updated = grad_y1 + grad_g_y1
                
                # Compute gradients for f function
                if x2.requires_grad:
                    grad_f_x2 = torch.autograd.grad(
                        outputs=y1, inputs=x2, grad_outputs=grad_y1_updated,
                        retain_graph=True, only_inputs=True
                    )[0]
                else:
                    grad_f_x2 = torch.zeros_like(x2)
                
                # Final gradients
                grad_x1 = grad_y1_updated
                grad_x2 = grad_y2 + grad_f_x2
        
        return torch.cat([grad_x1, grad_x2], dim=-1), None, None, None, None, None, None

class ModernReversibleQwen3Block(nn.Module):
    """
    Improved reversible Qwen3 block with modern PyTorch APIs and better convergence
    """
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # For reversible blocks, we need to work with doubled hidden size
        self.inner_dim = self.hidden_size
        
        # Attention layer
        self.self_attn = ImprovedReversibleQwen3CandidateAttention(config, layer_idx)
        
        # MLP layers
        self.mlp = Qwen3MLP(config)
        
        # Layer normalization
        rms_eps = getattr(config, 'rms_norm_eps', 1e-6)
        self.input_layernorm = Qwen3RMSNorm(self.inner_dim, eps=rms_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(self.inner_dim, eps=rms_eps)
        
        # Deterministic wrappers for reversible computation
        self.f_block = DeterministicFunction(self._attention_block)
        self.g_block = DeterministicFunction(self._mlp_block)
        
        # Position embedding support
        if QWEN3_ROTARY_AVAILABLE:
            try:
                self.rotary_emb = Qwen3RotaryEmbedding(
                    config.hidden_size // config.num_attention_heads,
                    max_position_embeddings=getattr(config, 'max_position_embeddings', 8192),
                )
            except Exception:
                self.rotary_emb = ImprovedRotaryEmbedding(
                    config.hidden_size // config.num_attention_heads,
                    max_position_embeddings=getattr(config, 'max_position_embeddings', 8192),
                )
        else:
            self.rotary_emb = ImprovedRotaryEmbedding(
                config.hidden_size // config.num_attention_heads,
                max_position_embeddings=getattr(config, 'max_position_embeddings', 8192),
            )
    
    def _attention_block(self, x, attention_mask=None, past_key_values=None, cache_position=None):
        """Attention computation for reversible block"""
        
        # Layer norm
        normed_x = self.input_layernorm(x)
        
        # Get position embeddings
        cos, sin = self.rotary_emb(normed_x)
        position_embeddings = (cos, sin)
        
        # Self attention
        attn_output = self.self_attn(
            normed_x,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        
        return attn_output
    
    def _mlp_block(self, x):
        """MLP computation for reversible block"""
        
        # Layer norm
        normed_x = self.post_attention_layernorm(x)
        
        # MLP
        mlp_output = self.mlp(normed_x)
        
        return mlp_output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """Enhanced forward pass with modern reversible computation"""
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # For reversible computation, we need doubled hidden dimension
        if hidden_dim == self.hidden_size:
            # Double the hidden states by concatenating with itself
            hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        elif hidden_dim != 2 * self.hidden_size:
            raise ValueError(f"Expected hidden dimension {self.hidden_size} or {2 * self.hidden_size}, got {hidden_dim}")
        
        # Apply reversible function with modern API
        output = ModernReversibleFunction.apply(
            hidden_states,
            self.f_block.fn,  # attention block
            self.g_block.fn,  # mlp block
            (),  # f_args
            (),  # g_args
            {"attention_mask": attention_mask, "past_key_values": past_key_values, "cache_position": cache_position},  # f_kwargs
            {},  # g_kwargs
        )
        
        # Split back to original dimension
        output = output[:, :, :self.hidden_size]
        
        return (output,)

class ModernReversibleQwen3Model(nn.Module):
    """
    Improved Reversible Qwen3 Model with modern PyTorch APIs and enhanced training support
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        
        # Embedding layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Reversible layers
        self.layers = nn.ModuleList([
            ModernReversibleQwen3Block(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Improved initialization
        self._init_weights()
        
        # Gradient checkpointing support
        self.gradient_checkpointing = False
        
    def _init_weights(self):
        """Improved weight initialization for better convergence"""
        
        # Embedding initialization
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        
        # LM head initialization (tied with embeddings is common)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        
        # Initialize reversible blocks
        for layer in self.layers:
            # Apply layer-specific initialization if needed
            pass
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Enhanced forward pass with modern mixed precision support"""
        
        # Handle inputs
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            hidden_states = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
            hidden_states = inputs_embeds
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=hidden_states.device, dtype=torch.bool)
        
        # Convert attention mask to causal mask
        if attention_mask.dim() == 2:
            causal_mask = self._prepare_causal_attention_mask(attention_mask, seq_len, hidden_states.dtype, hidden_states.device)
        else:
            causal_mask = attention_mask
        
        # Apply layers with modern mixed precision
        all_hidden_states = [] if output_hidden_states else None
        
        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            if self.gradient_checkpointing and self.training:
                # Use modern checkpoint API
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    cache_position,
                    use_reentrant=False  # Modern checkpointing
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                )
            
            hidden_states = layer_outputs[0]
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Return in expected format
        if return_dict:
            from transformers.modeling_outputs import BaseModelOutputWithPast
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
                hidden_states=all_hidden_states,
                attentions=None,  # Not implemented for reversible
            )
        else:
            return {
                'logits': logits,
                'last_hidden_state': hidden_states,
                'past_key_values': past_key_values,
            }
    
    def _prepare_causal_attention_mask(self, attention_mask, seq_len, dtype, device):
        """Prepare causal attention mask with improved efficiency"""
        
        # Create causal mask
        causal_mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        
        # Apply attention mask
        if attention_mask is not None:
            # Expand dimensions for broadcasting
            expanded_mask = attention_mask[:, None, None, :].expand(-1, 1, seq_len, -1)
            # Apply mask
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) + (1.0 - expanded_mask) * float('-inf')
        else:
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        return causal_mask

def create_modern_reversible_qwen3_model(
    vocab_size: int = 32000,
    hidden_size: int = 2048,
    num_hidden_layers: int = 24,
    num_attention_heads: int = 16,
    num_key_value_heads: Optional[int] = None,
    intermediate_size: Optional[int] = None,
    max_position_embeddings: int = 8192,
    rms_norm_eps: float = 1e-6,
    attention_type: str = "standard",
    use_reversible: bool = True,
    reverse_thres: int = 512,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    **kwargs
) -> ModernReversibleQwen3Model:
    """
    Create an improved reversible Qwen3 model with modern PyTorch APIs
    
    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension size
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of key-value heads (for GQA)
        intermediate_size: MLP intermediate size
        max_position_embeddings: Maximum sequence length
        rms_norm_eps: RMS norm epsilon
        attention_type: Type of attention ('standard', 'candidate_selection', 'native_sparse')
        use_reversible: Whether to use reversible layers
        reverse_thres: Threshold for reversible computation
        device: Device to place model on
        dtype: Model dtype
        **kwargs: Additional configuration parameters
    
    Returns:
        ModernReversibleQwen3Model instance
    """
    
    # Set defaults
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads
    
    if intermediate_size is None:
        intermediate_size = hidden_size * 4
    
    # Create configuration
    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        attention_dropout=0.1,
        **kwargs
    )
    
    # Add custom attributes
    config.attention_type = attention_type
    config.use_reversible = use_reversible
    config.reverse_thres = reverse_thres
    
    # Add attention-specific parameters
    if attention_type == "candidate_selection":
        config.candidate_pr_ratio = kwargs.get('candidate_pr_ratio', 0.5)
        config.candidate_top_k = kwargs.get('candidate_top_k', 40)
    elif attention_type == "native_sparse":
        config.sliding_window_size = kwargs.get('sliding_window_size', 64)
        config.compress_block_size = kwargs.get('compress_block_size', 16)
        config.compress_block_sliding_stride = kwargs.get('compress_block_sliding_stride', 8)
        config.selection_block_size = kwargs.get('selection_block_size', 16)
        config.num_selected_blocks = kwargs.get('num_selected_blocks', 4)
        config.use_diff_topk = kwargs.get('use_diff_topk', True)
        config.query_heads_share_selected_kv = kwargs.get('query_heads_share_selected_kv', True)
        config.use_triton_kernel = kwargs.get('use_triton_kernel', True)
    
    # Create model
    model = ModernReversibleQwen3Model(config)
    
    # Move to device and set dtype
    model = model.to(device=device, dtype=dtype)
    
    print(f"Created Modern Reversible Qwen3 Model:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Layers: {num_hidden_layers}")
    print(f"  Attention heads: {num_attention_heads}")
    print(f"  KV heads: {num_key_value_heads}")
    print(f"  Attention type: {attention_type}")
    print(f"  Reversible: {use_reversible}")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")
    
    return model

def modern_train_with_mixed_precision(model, train_loader, val_loader, config):
    """
    Modern mixed precision training with improved convergence for reversible models
    """
    
    # Use modern AMP API
    scaler = torch.amp.GradScaler('cuda')
    
    # Enhanced optimizer configuration
    optimizer_config = config.get('optimizer_kwargs', {})
    optimizer_config.setdefault('lr', 3e-4)
    optimizer_config.setdefault('betas', (0.9, 0.95))
    optimizer_config.setdefault('eps', 1e-8)
    optimizer_config.setdefault('weight_decay', 0.01)
    
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
    
    # Enhanced learning rate scheduling
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = config.get('warmup_steps', total_steps // 10)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop with improved monitoring
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(model.device if hasattr(model, 'device') else 'cuda')
            targets = targets.to(model.device if hasattr(model, 'device') else 'cuda')
            
            # Modern autocast API
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Compute loss with proper shape handling
                if logits.dim() == 3:  # [batch, seq, vocab]
                    logits = logits.view(-1, logits.size(-1))
                    targets = targets.view(-1)
                
                loss = F.cross_entropy(logits, targets, ignore_index=-100)
                loss = loss / config.get('gradient_accumulation_steps', 1)
            
            # Scale loss and backward
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % config.get('gradient_accumulation_steps', 1) == 0:
                # Gradient clipping with scaler
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('max_grad_norm', 1.0))
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Progress logging
            if batch_idx % config.get('log_interval', 100) == 0:
                avg_loss = total_loss / num_batches
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
        
        # Validation with modern autocast
        if val_loader:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(model.device if hasattr(model, 'device') else 'cuda')
                    targets = targets.to(model.device if hasattr(model, 'device') else 'cuda')
                    
                    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                        outputs = model(inputs)
                        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                        
                        if logits.dim() == 3:
                            logits = logits.view(-1, logits.size(-1))
                            targets = targets.view(-1)
                        
                        loss = F.cross_entropy(logits, targets, ignore_index=-100)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
            print(f"Epoch {epoch}, Train Loss: {total_loss / num_batches:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return model

# Test function for the modern implementation
def test_modern_reversible_qwen3():
    """Test the modern reversible Qwen3 implementation"""
    
    print("Testing Modern Reversible Qwen3 Implementation...")
    
    # Create test model
    model = create_modern_reversible_qwen3_model(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        attention_type="standard",
        use_reversible=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    
    # Test with modern autocast
    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        outputs = model(input_ids)
    
    print(f"Test passed! Output shape: {outputs['logits'].shape}")
    print(f"Memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB" if torch.cuda.is_available() else "CPU mode")
    
    return model

if __name__ == "__main__":
    # Run test
    test_model = test_modern_reversible_qwen3()
    print("Modern Reversible Qwen3 implementation ready!")
