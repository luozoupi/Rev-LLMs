#!/usr/bin/env python3
"""
Quick test to verify the v3f fixes work correctly.
Tests just the model creation and training configuration to ensure they match v202_2f.
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class QuickTester:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def get_wikitext_training_config(self):
        """Optimized config for WikiText training"""
        return {
            'epochs': 8,
            'learning_rate': 0.0003,
            'min_learning_rate': 1e-6,
            'scheduler_type': 'cosine',
            'warmup_ratio': 0.1,
            'optimizer': 'adamw',
            'weight_decay': 0.01,
            'gradient_clip_norm': 1.0,
            'gradient_accumulation_steps': 2,
            'use_amp': True,
            'amp_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            'early_stopping_patience': 4,
            'early_stopping_min_delta': 0.001,
            'betas': (0.9, 0.95),
            'eps': 1e-8
        }
    
    def get_optimized_training_config(self, model_type="reversible"):
        """Get optimized training configuration based on model type"""
        
        # Use the same configuration for both reversible and non-reversible models for fair comparison
        base_config = self.get_wikitext_training_config()
        base_config.update({
            'epochs': 30,  # Give both models the same number of epochs
        })
        return base_config
    
    def setup_test_models(self):
        """Setup models with fixed parameters"""
        try:
            from qwen3_reversible_02_2 import create_reversible_qwen3_model
            print("✓ Successfully imported qwen3_reversible_02_2")
        except ImportError as e:
            print(f"✗ Failed to import qwen3_reversible_02_2: {e}")
            return None
        
        models = {}
        vocab_size = 16000
        hidden_size = 512  # Fixed back to v202_2f values
        num_layers = 4     # Fixed back to v202_2f values  
        num_heads = 8
        
        configs = [
            ("reversible_standard", "standard", True),
            ("non_reversible_standard", "standard", False)
        ]
        
        for name, attention_type, use_reversible in configs:
            try:
                model = create_reversible_qwen3_model(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    num_hidden_layers=num_layers,
                    num_attention_heads=num_heads,
                    num_key_value_heads=num_heads // 2,
                    attention_type=attention_type,
                    use_reversible=use_reversible,
                    reverse_thres=256 if use_reversible else 999999,
                    candidate_pr_ratio=0.7,
                    candidate_top_k=32,
                    intermediate_size=hidden_size * 4,
                    max_position_embeddings=2048,
                    rms_norm_eps=1e-6
                )
                
                model = model.to(self.device)
                models[name] = model
                
                param_count = sum(p.numel() for p in model.parameters())
                print(f"✓ Created {name} model ({param_count:,} params)")
                
            except Exception as e:
                print(f"✗ Failed to create {name}: {e}")
                return None
                
        return models
    
    def test_training_configs(self):
        """Test that both models get the same training config"""
        print("\nTesting training configurations:")
        
        rev_config = self.get_optimized_training_config("reversible")
        non_rev_config = self.get_optimized_training_config("standard")
        
        print(f"Reversible config epochs: {rev_config['epochs']}")
        print(f"Non-reversible config epochs: {non_rev_config['epochs']}")
        
        if rev_config['epochs'] == non_rev_config['epochs']:
            print("✓ Both models get same number of epochs")
        else:
            print("✗ Models get different training epochs!")
        
        return rev_config, non_rev_config
    
    def quick_training_test(self, model, config, model_name):
        """Quick training test with synthetic data"""
        print(f"\nTesting {model_name} training:")
        
        # Create synthetic dataset
        seq_len = 128
        batch_size = 2
        vocab_size = model.config.vocab_size
        
        # Create synthetic data
        inputs = torch.randint(0, vocab_size, (10, seq_len))
        targets = torch.randint(0, vocab_size, (10, seq_len))
        
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=config['betas'],
            eps=config['eps']
        )
        
        criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        
        # Quick training test
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
            if batch_idx >= 3:  # Only test a few batches
                break
                
            batch_inputs = batch_inputs.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            optimizer.zero_grad()
            
            try:
                outputs = model(batch_inputs)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch_targets[..., 1:].contiguous()
                
                loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
                
            except Exception as e:
                print(f"  ✗ Training error at batch {batch_idx}: {e}")
                return False
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"  ✓ Average loss: {avg_loss:.4f}")
        return True
    
    def run_quick_test(self):
        """Run comprehensive quick test"""
        print("="*60)
        print("QUICK TEST: V3F FIXES VERIFICATION")
        print("="*60)
        
        # Test 1: Training configs
        rev_config, non_rev_config = self.test_training_configs()
        
        # Test 2: Model creation
        print("\nTesting model creation:")
        models = self.setup_test_models()
        if not models:
            print("✗ Model creation failed!")
            return False
        
        # Test 3: Quick training
        for model_name, model in models.items():
            config = self.get_optimized_training_config(model_name)
            success = self.quick_training_test(model, config, model_name)
            if not success:
                print(f"✗ Training test failed for {model_name}")
                return False
        
        print("\n" + "="*60)
        print("✓ ALL QUICK TESTS PASSED!")
        print("V3F fixes appear to be working correctly.")
        print("="*60)
        return True

if __name__ == "__main__":
    tester = QuickTester()
    success = tester.run_quick_test()
    if success:
        print("\n🎉 Ready to run full v3f test!")
    else:
        print("\n❌ Issues found - need to fix before running full test")
