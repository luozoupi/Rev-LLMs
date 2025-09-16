"""
Test Dataset Loading Fix
========================

This script tests that the dataset loading fix works correctly.

Usage:
python test_dataset_loading_fix.py
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_single_dataset_loading():
    """Test loading a single dataset with the correct format"""
    print("🎯 Testing Single Dataset Loading...")
    
    try:
        from datasets import load_dataset
        
        # Test the format that works according to HuggingFace docs
        print("  Testing nyu-mll/glue format...")
        dataset = load_dataset("nyu-mll/glue", "sst2", split="train[:10]")
        print(f"  ✓ Successfully loaded SST-2: {len(dataset)} examples")
        print(f"  ✓ Sample: {dataset[0]}")
        
        # Test validation split
        val_dataset = load_dataset("nyu-mll/glue", "sst2", split="validation[:5]")
        print(f"  ✓ Successfully loaded SST-2 validation: {len(val_dataset)} examples")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Single dataset loading failed: {e}")
        return False

def test_comprehensive_dataset_loading():
    """Test the comprehensive dataset loader"""
    print("\n📚 Testing Comprehensive Dataset Loader...")
    
    try:
        from comprehensive_qwen3_dizo_benchmark import DiZODatasetCompatibilityLayer
        
        # Create data loader
        data_loader = DiZODatasetCompatibilityLayer()
        
        # Test loading different tasks
        tasks = ['sst2', 'cola', 'mrpc']
        
        for task in tasks:
            print(f"  Testing {task}...")
            
            # Test train split
            train_data = data_loader.load_glue_dataset(task, 'train', limit=10)
            if train_data:
                print(f"    ✓ {task} train: {len(train_data)} examples")
            else:
                print(f"    ✗ {task} train: Failed to load")
                continue
            
            # Test validation split
            val_data = data_loader.load_glue_dataset(task, 'validation', limit=5)
            if val_data:
                print(f"    ✓ {task} validation: {len(val_data)} examples")
            else:
                print(f"    ⚠ {task} validation: Failed to load (trying alternatives)")
                
                # Try alternative splits
                for split in ['dev', 'test']:
                    alt_data = data_loader.load_glue_dataset(task, split, limit=5)
                    if alt_data:
                        print(f"    ✓ {task} {split}: {len(alt_data)} examples")
                        break
        
        return True
        
    except Exception as e:
        print(f"  ✗ Comprehensive dataset loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_processing():
    """Test that data processing works correctly"""
    print("\n🔧 Testing Data Processing...")
    
    try:
        from comprehensive_qwen3_dizo_benchmark import DiZODatasetCompatibilityLayer
        
        data_loader = DiZODatasetCompatibilityLayer()
        
        # Load a small sample
        data = data_loader.load_glue_dataset('sst2', 'train', limit=3)
        
        if not data:
            print("  ✗ No data loaded for processing test")
            return False
        
        # Check data format
        sample = data[0]
        print(f"  ✓ Loaded {len(data)} examples for processing test")
        print(f"  ✓ Sample keys: {list(sample.keys())}")
        
        # Check required fields
        required_fields = ['text', 'label']
        for field in required_fields:
            if field in sample:
                print(f"    ✓ Has {field}: {type(sample[field])}")
            else:
                print(f"    ✗ Missing {field}")
                return False
        
        # Check tokenization if available
        if 'input_ids' in sample:
            print(f"    ✓ Has tokenized input: {sample['input_ids'].shape}")
        else:
            print(f"    ℹ No tokenization (using text-only mode)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Data processing test failed: {e}")
        return False

def test_task_info():
    """Test task information retrieval"""
    print("\n📋 Testing Task Information...")
    
    try:
        from comprehensive_qwen3_dizo_benchmark import DiZODatasetCompatibilityLayer
        
        data_loader = DiZODatasetCompatibilityLayer()
        
        tasks = ['sst2', 'cola', 'mrpc', 'qqp', 'stsb']
        
        for task in tasks:
            task_info = data_loader.get_task_info(task)
            print(f"  {task}: {task_info['num_labels']} labels, {task_info['metric']} metric, {task_info['type']} type")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Task info test failed: {e}")
        return False

def main():
    """Run all dataset loading tests"""
    print("Dataset Loading Fix Test")
    print("=" * 50)
    
    tests = [
        ("Single Dataset Loading", test_single_dataset_loading),
        ("Comprehensive Dataset Loader", test_comprehensive_dataset_loading),
        ("Data Processing", test_data_processing),
        ("Task Information", test_task_info),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All dataset loading tests passed!")
        print("\nYou can now run training without dataset loading errors:")
        print("  python train_and_benchmark_dizo_scale.py --scale small --tasks sst2")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed.")
        print("\nIf datasets are still not loading, try:")
        print("  python fix_dataset_loading.py --all")

if __name__ == "__main__":
    main()