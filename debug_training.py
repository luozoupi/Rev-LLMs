#!/usr/bin/env python3
"""
Debug script to identify training failure issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from comprehensive_qwen3_dizo_benchmark_2 import (
    BenchmarkConfig, ModelTrainer, DiZODatasetCompatibilityLayer, 
    GLUE_PLUS_TASKS, MODEL_CREATION_AVAILABLE
)

def debug_training_pipeline():
    print("=== DEBUGGING TRAINING PIPELINE ===\n")
    
    # 1. Test data loading
    print("1. Testing data loading...")
    data_loader = DiZODatasetCompatibilityLayer()
    
    task_name = 'sst2'
    train_data = data_loader.load_glue_dataset(task_name, 'train', 10)  # Just 10 examples
    val_data = data_loader.load_glue_dataset(task_name, 'validation', 10)
    
    print(f"   Train data loaded: {len(train_data) if train_data else 'None'}")
    print(f"   Val data loaded: {len(val_data) if val_data else 'None'}")
    
    if train_data and len(train_data) > 0:
        print(f"   Sample train example: {train_data[0]}")
        print(f"   Sample keys: {list(train_data[0].keys())}")
        
        # Check tensor shapes and types
        sample = train_data[0]
        if 'input_ids' in sample:
            print(f"   input_ids shape: {sample['input_ids'].shape}")
            print(f"   input_ids dtype: {sample['input_ids'].dtype}")
            print(f"   label: {sample['label']}, type: {type(sample['label'])}")
    
    # 2. Test model creation
    print("\n2. Testing model creation...")
    if not MODEL_CREATION_AVAILABLE:
        print("   ❌ Model creation not available!")
        return
    
    config = BenchmarkConfig(scale="small")
    trainer = ModelTrainer(config, device='cpu')  # Use CPU for debugging
    
    try:
        models = trainer.create_models()
        print(f"   ✓ Created {len(models)} models")
        for name, model in models.items():
            print(f"     {name}: {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return
    
    # 3. Test model forward pass
    print("\n3. Testing model forward pass...")
    if train_data and models:
        model = list(models.values())[0]  # Get first model
        task_config = GLUE_PLUS_TASKS[task_name]
        classifier = trainer._create_classification_model(model, task_config.num_labels)
        
        # Test with single example
        sample = train_data[0]
        try:
            if 'input_ids' in sample:
                input_ids = sample['input_ids'].unsqueeze(0)  # Add batch dimension
                attention_mask = sample['attention_mask'].unsqueeze(0)  # Add batch dimension
                print(f"   Input shape: {input_ids.shape}")
                print(f"   Attention mask shape: {attention_mask.shape}")
                
                # Forward pass
                outputs = classifier(input_ids, attention_mask=attention_mask)
                print(f"   Output shape: {outputs.shape}")
                print(f"   Output sample: {outputs[0]}")
                
                # Test loss calculation
                label = sample['label'].unsqueeze(0)  # Add batch dimension
                print(f"   Label: {label}")
                
                loss = nn.CrossEntropyLoss()(outputs, label)
                print(f"   ✓ Loss calculated: {loss.item()}")
            else:
                print("   ⚠ No input_ids found, skipping forward pass test")
                
        except Exception as e:
            print(f"   ❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. Test dataloader creation
    print("\n4. Testing dataloader creation...")
    if train_data:
        try:
            train_loader = trainer._create_dataloader(train_data, batch_size=2, shuffle=False)
            print(f"   ✓ DataLoader created with {len(train_loader)} batches")
            
            # Test one batch
            for i, batch in enumerate(train_loader):
                print(f"   Batch {i}: {len(batch)} items")
                print(f"   First item keys: {list(batch[0].keys())}")
                
                # Test batch processing
                if len(batch) > 0 and isinstance(batch[0], dict) and 'input_ids' in batch[0]:
                    try:
                        input_ids = torch.stack([item['input_ids'] for item in batch])
                        attention_mask = torch.stack([item['attention_mask'] for item in batch])
                        labels = torch.stack([item['label'] for item in batch])
                        print(f"   Stacked input_ids shape: {input_ids.shape}")
                        print(f"   Stacked attention_mask shape: {attention_mask.shape}")
                        print(f"   Stacked labels shape: {labels.shape}")
                        print(f"   Labels: {labels}")
                        
                        # Test forward pass with batch
                        outputs = classifier(input_ids, attention_mask=attention_mask)
                        print(f"   Batch output shape: {outputs.shape}")
                        
                        # Test loss
                        loss = nn.CrossEntropyLoss()(outputs, labels)
                        print(f"   ✓ Batch loss: {loss.item()}")
                        
                    except Exception as e:
                        print(f"   ❌ Batch processing failed: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"   ⚠ Unexpected batch format: len={len(batch)}, type(batch[0])={type(batch[0]) if len(batch) > 0 else 'empty'}")
                
                if i >= 1:  # Only test first 2 batches
                    break
                    
        except Exception as e:
            print(f"   ❌ DataLoader creation failed: {e}")
    
    # 5. Test training epoch simulation
    print("\n5. Testing training epoch simulation...")
    if train_data and models:
        try:
            model = list(models.values())[0]
            task_config = GLUE_PLUS_TASKS[task_name]
            classifier = trainer._create_classification_model(model, task_config.num_labels)
            
            optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4)
            train_loader = trainer._create_dataloader(train_data[:4], batch_size=2, shuffle=False)  # Just 4 examples
            
            classifier.train()
            total_loss = 0
            num_batches = 0
            
            print(f"   Starting mini-epoch with {len(train_loader)} batches...")
            
            for batch_idx, batch in enumerate(train_loader):
                print(f"   Processing batch {batch_idx}...")
                optimizer.zero_grad()
                
                if len(batch) > 0 and isinstance(batch[0], dict) and 'input_ids' in batch[0]:
                    input_ids = torch.stack([item['input_ids'] for item in batch])
                    attention_mask = torch.stack([item['attention_mask'] for item in batch])
                    labels = torch.stack([item['label'] for item in batch])
                    
                    outputs = classifier(input_ids, attention_mask=attention_mask)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    print(f"     Batch {batch_idx} loss: {loss.item()}")
                else:
                    print(f"     Batch {batch_idx} skipped - unexpected format")
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"   ✓ Mini-epoch completed: avg_loss={avg_loss}, num_batches={num_batches}")
            
        except Exception as e:
            print(f"   ❌ Training epoch simulation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_training_pipeline()