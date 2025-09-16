#!/usr/bin/env python3
"""
Debug Qwen tokenizer properties
"""

from transformers import AutoTokenizer

def debug_qwen_tokenizer():
    print("Debugging Qwen tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen-1_8B", trust_remote_code=True)
        
        print(f"Tokenizer class: {type(tokenizer)}")
        print(f"pad_token: {repr(tokenizer.pad_token)}")
        print(f"pad_token_id: {repr(tokenizer.pad_token_id)}")
        print(f"eos_token: {repr(tokenizer.eos_token)}")
        print(f"eos_token_id: {repr(tokenizer.eos_token_id)}")
        print(f"unk_token: {repr(tokenizer.unk_token)}")
        print(f"unk_token_id: {repr(tokenizer.unk_token_id)}")
        print(f"bos_token: {repr(tokenizer.bos_token)}")
        print(f"bos_token_id: {repr(tokenizer.bos_token_id)}")
        
        # Check vocab size and some tokens
        print(f"Vocab size: {len(tokenizer.get_vocab())}")
        vocab_items = list(tokenizer.get_vocab().items())[:10]
        print(f"First 10 vocab items: {vocab_items}")
        
        # Check special tokens
        print(f"Special tokens: {tokenizer.special_tokens_map}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_qwen_tokenizer()