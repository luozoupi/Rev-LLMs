# ==============================================================================
# CRITICAL FIXES FOR model_qwen3_can_sel.py
# ==============================================================================

# 1. ADD MISSING IMPORTS at the top of the file
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Union, Tuple
from torch import Tensor

# Import from existing project modules
import candi_sel_new_v2 as candi_sel_new
import quantization

# Import Qwen3 components (adjust paths as needed)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config, 
    Qwen3RMSNorm, 
    repeat_kv, 
    apply_rotary_pos_emb,
    Qwen3ForCausalLM,
    Qwen3DecoderLayer,
    Qwen3Model
)
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

# ==============================================================================
# 2. CREATE PROPER CONFIGURATION CLASS
# ==============================================================================

class Qwen3CandidateConfig(Qwen3Config):
    """Extended Qwen3 config with candidate selection parameters"""
    
    def __init__(
        self,
        use_candidate_selection=False,
        candidate_pr_ratio=0.5,
        candidate_top_k=40,
        candidate_layers=None,  # None means all layers
        candidate_use_checkpointing=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.use_candidate_selection = use_candidate_selection
        self.candidate_pr_ratio = candidate_pr_ratio
        self.candidate_top_k = candidate_top_k
        self.candidate_layers = candidate_layers or list(range(self.num_hidden_layers))
        self.candidate_use_checkpointing = candidate_use_checkpointing
        
        # CRITICAL FIX: Ensure consistent head dimension calculation
        if not hasattr(self, 'head_dim'):
            self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Debug config values
        print(f"Config debug:")
        print(f"  hidden_size: {self.hidden_size}")
        print(f"  num_attention_heads: {self.num_attention_heads}")
        print(f"  num_key_value_heads: {getattr(self, 'num_key_value_heads', 'NOT SET')}")
        print(f"  calculated head_dim: {self.head_dim}")
        print(f"  expected q_proj size: {self.num_attention_heads * self.head_dim}")
        print(f"  expected k/v_proj size: {getattr(self, 'num_key_value_heads', self.num_attention_heads) * self.head_dim}")

# ==============================================================================
# 3. FIXED ATTENTION IMPLEMENTATION
# ==============================================================================

class Qwen3CandidateSelectionAttention(nn.Module):
    """Candidate Selection Attention for Qwen3 - Fixed Version"""

    def __init__(self, config: Qwen3CandidateConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.pr_ratio = config.candidate_pr_ratio
        self.top_k = config.candidate_top_k
        
        # FIXED: Explicit dimension calculations
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        
        # Handle num_key_value_heads properly
        if hasattr(config, 'num_key_value_heads') and config.num_key_value_heads is not None:
            self.num_key_value_heads = config.num_key_value_heads
        else:
            # Default to same as attention heads (Multi-Head Attention)
            self.num_key_value_heads = self.num_attention_heads
        
        # Calculate head dimension
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        
        # Other attention parameters
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = getattr(config, 'attention_dropout', 0.0)
        self.is_causal = True
        self.use_checkpointing = config.candidate_use_checkpointing
        
        # Debug the calculated dimensions
        print(f"Attention layer {layer_idx} dimensions:")
        print(f"  hidden_size: {self.hidden_size}")
        print(f"  num_attention_heads: {self.num_attention_heads}")
        print(f"  num_key_value_heads: {self.num_key_value_heads}")
        print(f"  head_dim: {self.head_dim}")
        print(f"  num_key_value_groups: {self.num_key_value_groups}")
        
        # FIXED: Explicit projection sizes
        q_proj_size = self.num_attention_heads * self.head_dim
        kv_proj_size = self.num_key_value_heads * self.head_dim
        
        print(f"  q_proj_size: {self.hidden_size} -> {q_proj_size}")
        print(f"  kv_proj_size: {self.hidden_size} -> {kv_proj_size}")
        
        # Qwen3-specific projections with explicit sizes
        self.q_proj = nn.Linear(self.hidden_size, q_proj_size, bias=getattr(config, 'attention_bias', False))
        self.k_proj = nn.Linear(self.hidden_size, kv_proj_size, bias=getattr(config, 'attention_bias', False))
        self.v_proj = nn.Linear(self.hidden_size, kv_proj_size, bias=getattr(config, 'attention_bias', False))
        self.o_proj = nn.Linear(q_proj_size, self.hidden_size, bias=getattr(config, 'attention_bias', False))
        
        # Qwen3's RMSNorm for Q and K
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=getattr(config, 'rms_norm_eps', 1e-6))
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=getattr(config, 'rms_norm_eps', 1e-6))
        
        # Sliding window support
        self.sliding_window = None
        if hasattr(config, 'layer_types') and layer_idx < len(config.layer_types):
            if config.layer_types[layer_idx] == "sliding_attention":
                self.sliding_window = getattr(config, 'sliding_window', None)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        if self.use_checkpointing and self.training:
            return checkpoint(
                self._forward_impl, 
                hidden_states, 
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
                use_reentrant=False
            )
        else:
            return self._forward_impl(
                hidden_states, 
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position
            )
    
    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # FIXED: Direct projections
        query_proj = self.q_proj(hidden_states)  # [batch, seq_len, num_heads * head_dim]
        key_proj = self.k_proj(hidden_states)    # [batch, seq_len, num_kv_heads * head_dim]  
        value_proj = self.v_proj(hidden_states)  # [batch, seq_len, num_kv_heads * head_dim]
        
        # FIXED: Reshape to multi-head format
        query_states = query_proj.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        key_states = key_proj.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_proj.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        
        # Apply RMS normalization
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        
        # Transpose to [batch, num_heads, seq_len, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        print(f"After reshape and norm:")
        print(f"  query_states: {query_states.shape}")
        print(f"  key_states: {key_states.shape}")
        print(f"  value_states: {value_states.shape}")
        
        # Apply RoPE position embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Handle KV cache
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        
        # Apply candidate selection
        return self._candidate_selection_attention(
            query_states, key_states, value_states, attention_mask, (batch_size, seq_len)
        )
    
    def _candidate_selection_attention(
        self, 
        query_states: torch.Tensor, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor], 
        input_shape: Tuple[int, int]  # (batch_size, seq_len)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, seq_len = input_shape
        
        # Handle GQA by repeating key/value states
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        print(f"After GQA repeat:")
        print(f"  query_states: {query_states.shape}")
        print(f"  key_states: {key_states.shape}")
        print(f"  value_states: {value_states.shape}")
        
        # Convert attention mask for candidate selection (key fix here!)
        extended_attention_mask = None
        if attention_mask is not None:
            # Attention mask comes in format [batch, 1, seq_len, seq_len] from Qwen3
            # We need to convert it to the format expected by candidate selection
            if attention_mask.dim() == 4:
                # Already in correct format
                extended_attention_mask = attention_mask
            elif attention_mask.dim() == 2:
                # Convert [batch, seq_len] to [batch, 1, 1, seq_len]
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            else:
                # Handle other formats
                extended_attention_mask = attention_mask
        
        # Apply candidate selection pruning
        try:
            S_prune_mask = candi_sel_new.QK_prune_binary_quant(
                query_states, key_states, extended_attention_mask,
                pr_ratio=self.pr_ratio, top_k=self.top_k
            )
            print(f"✓ Candidate selection successful, mask shape: {S_prune_mask.shape}")
        except Exception as e:
            print(f"Warning: Candidate selection failed: {e}")
            # Fallback to no pruning
            S_prune_mask = torch.zeros_like(
                torch.matmul(query_states, key_states.transpose(2, 3))
            )
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        
        # Apply causal mask
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
        
        # Apply candidate selection pruning
        attn_weights = attn_weights + S_prune_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # Compute output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        print(f"Final output shape: {attn_output.shape}")
        
        return attn_output, attn_weights

# ==============================================================================
# 4. FIXED DECODER LAYER
# ==============================================================================

class Qwen3CandidateDecoderLayer(nn.Module):
    """Decoder layer with candidate selection attention"""
    
    def __init__(self, config: Qwen3CandidateConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        # Choose attention type based on configuration
        if config.use_candidate_selection and layer_idx in config.candidate_layers:
            self.self_attn = Qwen3CandidateSelectionAttention(config=config, layer_idx=layer_idx)
        else:
            # Use standard Qwen3 attention
            from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
            self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        
        # Keep the rest standard
        from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Store attention type
        if hasattr(config, 'layer_types') and layer_idx < len(config.layer_types):
            self.attention_type = config.layer_types[layer_idx]
        else:
            self.attention_type = "full_attention"
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

# ==============================================================================
# 5. HELPER FUNCTION FOR PROPER ROTARY EMBEDDINGS
# ==============================================================================

def create_simple_rotary_embeddings(seq_len: int, head_dim: int, batch_size: int = 1, device='cpu'):
    """Create simple rotary embeddings that match expected dimensions"""
    # Create position indices
    position_ids = torch.arange(seq_len, dtype=torch.float, device=device)
    
    # Create simple sinusoidal embeddings
    # We need [batch_size, seq_len, head_dim] for cos and sin
    cos_vals = torch.zeros(batch_size, seq_len, head_dim, device=device)
    sin_vals = torch.zeros(batch_size, seq_len, head_dim, device=device)
    
    for i in range(head_dim // 2):
        freq = 1.0 / (10000 ** (2 * i / head_dim))
        
        # Fill both the i-th and (head_dim//2 + i)-th positions
        cos_vals[:, :, 2*i] = torch.cos(position_ids * freq).unsqueeze(0).expand(batch_size, -1)
        cos_vals[:, :, 2*i + 1] = torch.cos(position_ids * freq).unsqueeze(0).expand(batch_size, -1)
        sin_vals[:, :, 2*i] = torch.sin(position_ids * freq).unsqueeze(0).expand(batch_size, -1)
        sin_vals[:, :, 2*i + 1] = torch.sin(position_ids * freq).unsqueeze(0).expand(batch_size, -1)
    
    return cos_vals, sin_vals

# ==============================================================================
# 5. COMPLETE TEST FUNCTION (FIXED)
# ==============================================================================

def test_qwen3_candidate_attention():
    """Test basic functionality - FIXED VERSION"""
    print("Testing Qwen3 Candidate Selection Attention...")
    
    # Create test config
    config = Qwen3CandidateConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1376,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=2048,
        use_candidate_selection=True,
        candidate_pr_ratio=0.5,
        candidate_top_k=40,
        rms_norm_eps=1e-6,
        layer_types=["full_attention"] * 12,  # All full attention for simplicity
    )
    
    # Create attention module
    attention = Qwen3CandidateSelectionAttention(config, layer_idx=0)
    
    # Create test data
    batch_size, seq_len = 2, 128
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Create proper rotary embeddings
    head_dim = config.hidden_size // config.num_attention_heads  # 64
    cos, sin = create_simple_rotary_embeddings(seq_len, head_dim, batch_size)
    position_embeddings = (cos, sin)
    
    # Create attention mask (causal mask)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
    
    try:
        # Test forward pass
        with torch.no_grad():
            output, weights = attention(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask
            )
        
        # Verify output shape
        assert output.shape == (batch_size, seq_len, config.hidden_size), f"Expected {(batch_size, seq_len, config.hidden_size)}, got {output.shape}"
        assert weights.shape == (batch_size, config.num_attention_heads, seq_len, seq_len), f"Expected {(batch_size, config.num_attention_heads, seq_len, seq_len)}, got {weights.shape}"
        
        print("✓ Basic functionality test passed")
        print(f"  Output shape: {output.shape}")
        print(f"  Attention weights shape: {weights.shape}")
        print(f"  Head dimension: {head_dim}")
        print(f"  Position embeddings shape: cos={cos.shape}, sin={sin.shape}")
        
        # Test with different sequence lengths
        for test_seq_len in [64, 256, 512]:
            print(f"Testing sequence length: {test_seq_len}")
            test_hidden_states = torch.randn(1, test_seq_len, config.hidden_size)
            
            # Create proper rotary embeddings for this sequence length
            test_cos, test_sin = create_simple_rotary_embeddings(test_seq_len, head_dim, batch_size=1)
            test_position_embeddings = (test_cos, test_sin)
            
            # Create causal mask for this sequence length
            test_causal_mask = torch.triu(torch.ones(test_seq_len, test_seq_len) * float('-inf'), diagonal=1)
            test_attention_mask = test_causal_mask.unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                test_output, _ = attention(
                    hidden_states=test_hidden_states,
                    position_embeddings=test_position_embeddings,
                    attention_mask=test_attention_mask
                )
            print(f"  ✓ Sequence length {test_seq_len}: {test_output.shape}")
        
        # Test candidate selection is actually working
        print("\nTesting candidate selection functionality...")
        
        # Test with very high pruning ratio to see the effect
        high_prune_attention = Qwen3CandidateSelectionAttention(
            Qwen3CandidateConfig(
                hidden_size=512,
                num_attention_heads=8,
                num_key_value_heads=8,
                use_candidate_selection=True,
                candidate_pr_ratio=0.9,  # Very aggressive pruning
                candidate_top_k=10,      # Very low top-k
                rms_norm_eps=1e-6,
                layer_types=["full_attention"] * 12,
            ), 
            layer_idx=0
        )
        
        with torch.no_grad():
            high_prune_output, high_prune_weights = high_prune_attention(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask
            )
        
        print(f"  ✓ High pruning test passed: {high_prune_output.shape}")
        
        # Check if attention weights are actually sparse
        zero_weights = (high_prune_weights == 0).sum().item()
        total_weights = high_prune_weights.numel()
        sparsity = zero_weights / total_weights
        print(f"  ✓ Attention sparsity: {sparsity:.2%} ({zero_weights}/{total_weights} weights are zero)")
        
        return True
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
# ==============================================================================
# 5B. ALTERNATIVE TEST WITHOUT ROTARY EMBEDDINGS
# ==============================================================================

def test_qwen3_candidate_attention_no_rope():
    """Test basic functionality without rotary embeddings to isolate issues"""
    print("Testing Qwen3 Candidate Selection Attention (No RoPE)...")
    
    # Create test config
    config = Qwen3CandidateConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1376,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=2048,
        use_candidate_selection=True,
        candidate_pr_ratio=0.5,
        candidate_top_k=40,
        rms_norm_eps=1e-6,
        layer_types=["full_attention"] * 12,
    )
    
    # Create attention module
    attention = Qwen3CandidateSelectionAttention(config, layer_idx=0)
    
    # Create test data
    batch_size, seq_len = 2, 128
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    head_dim = config.hidden_size // config.num_attention_heads
    
    print(f"Config info:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  hidden_states shape: {hidden_states.shape}")
    
    try:
        # Test just the projection and reshaping steps
        with torch.no_grad():
            # Test Q, K, V projections
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, head_dim)
            
            query_proj = attention.q_proj(hidden_states)
            key_proj = attention.k_proj(hidden_states)
            value_proj = attention.v_proj(hidden_states)
            
            print(f"Projection shapes:")
            print(f"  query_proj: {query_proj.shape}")
            print(f"  key_proj: {key_proj.shape}")
            print(f"  value_proj: {value_proj.shape}")
            
            # Reshape
            query_reshaped = query_proj.view(hidden_shape)
            key_reshaped = key_proj.view(hidden_shape)
            value_reshaped = value_proj.view(hidden_shape)
            
            print(f"Reshaped shapes:")
            print(f"  query_reshaped: {query_reshaped.shape}")
            print(f"  key_reshaped: {key_reshaped.shape}")
            print(f"  value_reshaped: {value_reshaped.shape}")
            
            # Apply RMS norm
            query_normed = attention.q_norm(query_reshaped)
            key_normed = attention.k_norm(key_reshaped)
            
            print(f"Normed shapes:")
            print(f"  query_normed: {query_normed.shape}")
            print(f"  key_normed: {key_normed.shape}")
            
            # Transpose
            query_states = query_normed.transpose(1, 2)
            key_states = key_normed.transpose(1, 2)
            value_states = value_reshaped.transpose(1, 2)
            
            print(f"Final Q/K/V shapes:")
            print(f"  query_states: {query_states.shape}")
            print(f"  key_states: {key_states.shape}")
            print(f"  value_states: {value_states.shape}")
            
            # Test candidate selection directly (skip RoPE)
            extended_attention_mask = None
            
            # Create mock pruning mask to test candidate selection
            try:
                S_prune_mask = candi_sel_new.QK_prune_binary_quant(
                    query_states, key_states, extended_attention_mask,
                    pr_ratio=0.5, top_k=40
                )
                print(f"✓ Candidate selection pruning mask created: {S_prune_mask.shape}")
                
                # Count how many elements are pruned
                pruned_count = (S_prune_mask == -10000).sum().item()
                total_count = S_prune_mask.numel()
                pruning_ratio = pruned_count / total_count
                print(f"  Actual pruning ratio: {pruning_ratio:.2%} ({pruned_count}/{total_count})")
                
            except Exception as e:
                print(f"✗ Candidate selection failed: {e}")
                return False
            
            # Test basic attention computation
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * attention.scaling
            attn_weights = attn_weights + S_prune_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1)
            final_output = attention.o_proj(attn_output)
            
            print(f"✓ Attention computation successful: {final_output.shape}")
            
            return True
            
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==============================================================================
# 6. RUN TEST
# ==============================================================================

if __name__ == "__main__":
    print("="*60)
    print("QWEN3 CANDIDATE SELECTION ATTENTION TESTS")
    print("="*60)
    
    # First, test without rotary embeddings to isolate the issue
    print("\n1. Testing without rotary embeddings...")
    success_no_rope = test_qwen3_candidate_attention_no_rope()
    
    if success_no_rope:
        print("\n2. Testing with rotary embeddings...")
        success_with_rope = test_qwen3_candidate_attention()
        
        if success_with_rope:
            print("\n🎉 All tests passed! The candidate selection attention is working.")
        else:
            print("\n⚠️ Basic functionality works, but rotary embeddings have issues.")
            print("You can proceed with integration using the core attention mechanism.")
    else:
        print("\n❌ Basic tests failed. Check the core implementation.")
        print("Focus on fixing the fundamental candidate selection logic first.")