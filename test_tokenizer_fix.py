#!/usr/bin/env python3
"""
Quick test to verify tokenizer padding token configuration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from glue_plus_benchmark import GLUEPlusBenchmark
from comprehensive_qwen3_dizo_benchmark_2 import DiZODatasetCompatibilityLayer

def test_tokenizer_configuration():
    print("Testing tokenizer configurations...")
    
    # Test 1: GLUEPlusBenchmark with default tokenizer
    print("\n1. Testing GLUEPlusBenchmark with default tokenizer...")
    try:
        benchmark = GLUEPlusBenchmark()
        print(f"   ✓ pad_token: {benchmark.tokenizer.pad_token}")
        print(f"   ✓ pad_token_id: {benchmark.tokenizer.pad_token_id}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: GLUEPlusBenchmark with Qwen tokenizer
    print("\n2. Testing GLUEPlusBenchmark with Qwen tokenizer...")
    try:
        benchmark = GLUEPlusBenchmark(tokenizer_name="qwen/Qwen-1_8B")
        print(f"   ✓ pad_token: {benchmark.tokenizer.pad_token}")
        print(f"   ✓ pad_token_id: {benchmark.tokenizer.pad_token_id}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: DiZODatasetCompatibilityLayer
    print("\n3. Testing DiZODatasetCompatibilityLayer...")
    try:
        data_loader = DiZODatasetCompatibilityLayer()
        if data_loader.tokenizer:
            print(f"   ✓ pad_token: {data_loader.tokenizer.pad_token}")
            print(f"   ✓ pad_token_id: {data_loader.tokenizer.pad_token_id}")
        else:
            print("   ! No tokenizer available")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 4: Simple tokenization test
    print("\n4. Testing actual tokenization with padding...")
    try:
        data_loader = DiZODatasetCompatibilityLayer()
        if data_loader.tokenizer:
            test_text = "This is a test sentence."
            tokens = data_loader.tokenizer(
                test_text,
                max_length=50,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            print(f"   ✓ Tokenization successful: input_ids shape = {tokens['input_ids'].shape}")
        else:
            print("   ! Skipping - no tokenizer available")
    except Exception as e:
        print(f"   ✗ Tokenization failed: {e}")

if __name__ == "__main__":
    test_tokenizer_configuration()