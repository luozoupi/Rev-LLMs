#!/usr/bin/env python3
"""
Minimal test of the fixed training pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from comprehensive_qwen3_dizo_benchmark_2 import (
    BenchmarkConfig, ModelTrainer, DiZODatasetCompatibilityLayer, 
    GLUE_PLUS_TASKS, MODEL_CREATION_AVAILABLE
)

def test_minimal_training():
    print("=== MINIMAL TRAINING TEST ===\n")
    
    if not MODEL_CREATION_AVAILABLE:
        print("❌ Model creation not available!")
        return
    
    # Very small config for fast testing
    config = BenchmarkConfig(scale="small")
    config.num_epochs = 1
    config.batch_size = 2
    config.eval_size = 4
    config.train_size = 8
    
    trainer = ModelTrainer(config, device='cpu')
    
    # Test data loading
    print("1. Testing data loading...")
    train_data = trainer.data_loader.load_glue_dataset('sst2', 'train', config.train_size)
    val_data = trainer.data_loader.load_glue_dataset('sst2', 'validation', config.eval_size)
    
    if not train_data or not val_data:
        print("❌ Failed to load data")
        return
    
    print(f"   ✓ Loaded {len(train_data)} train, {len(val_data)} val examples")
    
    # Test model creation
    print("\n2. Testing model creation...")
    models = trainer.create_models()
    if not models:
        print("❌ Failed to create models")
        return
    
    print(f"   ✓ Created {len(models)} models")
    
    # Test minimal training
    print("\n3. Testing minimal training...")
    model_name = list(models.keys())[0]
    model = list(models.values())[0]
    
    try:
        results = trainer.train_model_on_task(model, 'sst2', model_name)
        print(f"   ✓ Training completed!")
        print(f"   Results: {results}")
        
        if 'best_metric' in results and results['best_metric'] > 0:
            print(f"   ✓ Got valid metric: {results['best_metric']:.3f}")
        else:
            print(f"   ⚠ Metric might be low: {results.get('best_metric', 'N/A')}")
            
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal_training()