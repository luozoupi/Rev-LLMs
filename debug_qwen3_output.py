#!/usr/bin/env python3
"""
Debug script to check Qwen3 model output format
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from qwen3_reversible_02_2 import create_reversible_qwen3_model

def debug_qwen3_output():
    print("=== DEBUGGING QWEN3 MODEL OUTPUT ===\n")
    
    # Create a small model for testing
    model = create_reversible_qwen3_model(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_type="standard",
        use_reversible=False,
        reverse_thres=999999,
        candidate_pr_ratio=0.7,
        candidate_top_k=32
    )
    
    print(f"Model type: {type(model)}")
    print(f"Model class: {model.__class__.__name__}")
    
    # Test input
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    
    print(f"Input shape: {input_ids.shape}")
    
    # Test model output
    model.eval()
    with torch.no_grad():
        output = model(input_ids)
        
    print(f"Output type: {type(output)}")
    
    if isinstance(output, dict):
        print(f"Output keys: {list(output.keys())}")
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: type={type(value)}")
    elif isinstance(output, torch.Tensor):
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
    else:
        print(f"Unexpected output type: {type(output)}")

if __name__ == "__main__":
    debug_qwen3_output()