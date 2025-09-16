#!/usr/bin/env python3
"""
Debug DataLoader batch structure
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader, Dataset
from comprehensive_qwen3_dizo_benchmark_2 import DiZODatasetCompatibilityLayer

def debug_dataloader():
    print("=== DEBUGGING DATALOADER STRUCTURE ===\n")
    
    # Load some data
    data_loader = DiZODatasetCompatibilityLayer()
    train_data = data_loader.load_glue_dataset('sst2', 'train', 4)  # Just 4 examples
    
    print(f"Train data type: {type(train_data)}")
    print(f"Train data length: {len(train_data)}")
    
    if train_data:
        print(f"First item type: {type(train_data[0])}")
        print(f"First item keys: {list(train_data[0].keys())}")
    
    # Create DataLoader
    class TaskDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = TaskDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    print(f"\nDataLoader created with {len(dataloader)} batches")
    
    # Test batch iteration
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}:")
        print(f"  Batch type: {type(batch)}")
        print(f"  Batch length: {len(batch) if hasattr(batch, '__len__') else 'No len'}")
        
        # Try different ways to access batch data
        try:
            print(f"  batch[0]: {type(batch[0])}")
            if hasattr(batch[0], 'keys'):
                print(f"    keys: {list(batch[0].keys())}")
        except Exception as e:
            print(f"  ❌ Cannot access batch[0]: {e}")
        
        # Check if batch is a dict of tensors (default collate_fn behavior)
        if isinstance(batch, dict):
            print(f"  Batch is dict with keys: {list(batch.keys())}")
            for key, value in batch.items():
                print(f"    {key}: {type(value)}, shape: {value.shape if hasattr(value, 'shape') else 'no shape'}")
        
        # Check if batch is a list
        elif isinstance(batch, list):
            print(f"  Batch is list with {len(batch)} items")
            for j, item in enumerate(batch):
                print(f"    Item {j}: {type(item)}")
                if isinstance(item, dict):
                    print(f"      Keys: {list(item.keys())}")
        
        # Check if batch is a tuple
        elif isinstance(batch, tuple):
            print(f"  Batch is tuple with {len(batch)} items")
            for j, item in enumerate(batch):
                print(f"    Item {j}: {type(item)}")
        
        if i >= 1:  # Only check first 2 batches
            break

def test_custom_collate_fn():
    print("\n=== TESTING CUSTOM COLLATE FUNCTION ===\n")
    
    def custom_collate_fn(batch):
        """Custom collate function to handle dict batches"""
        print(f"Custom collate called with batch type: {type(batch)}, length: {len(batch)}")
        
        if len(batch) == 0:
            return []
        
        # Check if batch items are dicts
        if isinstance(batch[0], dict):
            # Stack tensors for each key
            batched = {}
            for key in batch[0].keys():
                if key in ['input_ids', 'attention_mask', 'label']:
                    # Stack tensor values
                    batched[key] = torch.stack([item[key] for item in batch])
                else:
                    # Keep non-tensor values as list
                    batched[key] = [item[key] for item in batch]
            return batched
        else:
            return batch
    
    # Load data and test
    data_loader = DiZODatasetCompatibilityLayer()
    train_data = data_loader.load_glue_dataset('sst2', 'train', 4)
    
    class TaskDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = TaskDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)
    
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i} with custom collate:")
        print(f"  Type: {type(batch)}")
        
        if isinstance(batch, dict):
            print(f"  Keys: {list(batch.keys())}")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"    {key}: type={type(value)}, length={len(value) if hasattr(value, '__len__') else 'no len'}")
        
        if i >= 1:
            break

if __name__ == "__main__":
    debug_dataloader()
    test_custom_collate_fn()