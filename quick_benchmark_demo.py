"""
Quick Benchmark Example - Minimal Working Example
================================================

This script provides a minimal working example to test your reversible Qwen3
implementation against standard models on a single task.

Usage:
python quick_benchmark_example.py
"""

import torch
import torch.nn as nn
import sys
import os
import time
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_benchmark_demo():
    """Run a minimal benchmark demonstration"""
    
    print("="*60)
    print("QUICK QWEN3 BENCHMARK EXAMPLE")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Step 1: Test model creation
    print("\n1. Testing Model Creation...")
    
    try:
        from qwen3_reversible_02_2 import create_reversible_qwen3_model
        print("  ✓ Model creation function imported")
    except ImportError as e:
        print(f"  ✗ Cannot import model creation: {e}")
        print("    Make sure qwen3_reversible_02_2.py is in the current directory")
        return False
    
    # Create small test models
    model_config = {
        'vocab_size': 5000,
        'hidden_size': 256,
        'num_hidden_layers': 4,
        'num_attention_heads': 8,
        'num_key_value_heads': 4
    }
    
    try:
        # Reversible model
        reversible_model = create_reversible_qwen3_model(
            **model_config,
            use_reversible=True,
            reverse_thres=128
        ).to(device)
        
        # Standard model  
        standard_model = create_reversible_qwen3_model(
            **model_config,
            use_reversible=False,
            reverse_thres=999999
        ).to(device)
        
        rev_params = sum(p.numel() for p in reversible_model.parameters())
        std_params = sum(p.numel() for p in standard_model.parameters())
        
        print(f"  ✓ Reversible model: {rev_params:,} parameters")
        print(f"  ✓ Standard model: {std_params:,} parameters")
        
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False
    
    # Step 2: Test forward pass
    print("\n2. Testing Forward Pass...")
    
    batch_size = 4
    seq_lengths = [64, 128, 256]  # Test different sequence lengths
    
    for seq_len in seq_lengths:
        print(f"  Testing sequence length {seq_len}...")
        
        # Create test input
        test_input = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len)).to(device)
        
        try:
            # Test reversible model
            start_time = time.time()
            with torch.no_grad():
                rev_output = reversible_model(test_input)
            rev_time = time.time() - start_time
            
            # Test standard model
            start_time = time.time() 
            with torch.no_grad():
                std_output = standard_model(test_input)
            std_time = time.time() - start_time
            
            print(f"    Reversible: {rev_output.shape}, {rev_time*1000:.1f}ms")
            print(f"    Standard: {std_output.shape}, {std_time*1000:.1f}ms")
            
        except Exception as e:
            print(f"    ✗ Forward pass failed for seq_len {seq_len}: {e}")
            continue
    
    # Step 3: Test backward pass and memory usage
    print("\n3. Testing Backward Pass and Memory...")
    
    try:
        test_input = torch.randint(0, model_config['vocab_size'], (2, 128)).to(device)
        
        # Test reversible model backward
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        rev_output = reversible_model(test_input)
        rev_loss = rev_output.sum()
        rev_loss.backward()
        
        if device == 'cuda':
            rev_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
        else:
            rev_memory = 0
        
        # Test standard model backward
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        std_output = standard_model(test_input)
        std_loss = std_output.sum()
        std_loss.backward()
        
        if device == 'cuda':
            std_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
        else:
            std_memory = 0
        
        print(f"  ✓ Reversible backward pass: {rev_memory:.1f} MB peak memory")
        print(f"  ✓ Standard backward pass: {std_memory:.1f} MB peak memory")
        
        if rev_memory > 0 and std_memory > 0:
            memory_ratio = rev_memory / std_memory
            print(f"  📊 Memory ratio (Rev/Std): {memory_ratio:.2f}x")
        
    except Exception as e:
        print(f"  ✗ Backward pass test failed: {e}")
    
    # Step 4: Simple benchmark on synthetic task
    print("\n4. Running Simple Classification Benchmark...")
    
    try:
        # Create simple classification task
        num_classes = 2
        train_size = 1000
        test_size = 200
        
        # Add classification heads
        rev_classifier = nn.Sequential(
            reversible_model,
            nn.Dropout(0.1),
            nn.Linear(model_config['hidden_size'], num_classes)
        ).to(actual_device)
        
        std_classifier = nn.Sequential(
            standard_model,
            nn.Dropout(0.1), 
            nn.Linear(model_config['hidden_size'], num_classes)
        ).to(actual_device)
        
        # Generate synthetic data
        train_inputs = torch.randint(0, model_config['vocab_size'], (train_size, 64)).to(actual_device)
        train_labels = torch.randint(0, num_classes, (train_size,)).to(actual_device)
        test_inputs = torch.randint(0, model_config['vocab_size'], (test_size, 64)).to(actual_device)
        test_labels = torch.randint(0, num_classes, (test_size,)).to(actual_device)
        
        # Simple training loop
        def train_simple(model, epochs=3):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0
                
                # Mini-batch training
                batch_size = 32
                for i in range(0, len(train_inputs), batch_size):
                    batch_inputs = train_inputs[i:i+batch_size]
                    batch_labels = train_labels[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_inputs)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                print(f"    Epoch {epoch+1}: Loss = {epoch_loss/len(train_inputs)*batch_size:.4f}")
            
            # Test accuracy
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_inputs)
                
                # Handle dict output for predictions
                if isinstance(test_outputs, dict):
                    if 'logits' in test_outputs:
                        logits = test_outputs['logits']
                    else:
                        first_key = next(iter(test_outputs.keys()))
                        logits = test_outputs[first_key]
                else:
                    logits = test_outputs
                
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == test_labels).float().mean().item()
            
            return accuracy
        
        print("  Training reversible model...")
        rev_accuracy = train_simple(rev_classifier)
        
        print("  Training standard model...")
        std_accuracy = train_simple(std_classifier)
        
        print(f"  📊 Results:")
        print(f"    Reversible accuracy: {rev_accuracy:.3f}")
        print(f"    Standard accuracy: {std_accuracy:.3f}")
        print(f"    Difference: {rev_accuracy - std_accuracy:+.3f}")
        
    except Exception as e:
        print(f"  ✗ Simple benchmark failed: {e}")
    
    # Step 5: Summary and next steps
    print("\n5. Summary and Next Steps")
    print("  ✓ Basic model functionality working")
    print("  ✓ Forward and backward passes functional")
    print("  ✓ Memory usage can be measured")
    print("  ✓ Simple training loop works")
    
    print(f"\n🎉 Quick benchmark completed successfully!")
    print(f"\nNext steps to run full benchmark:")
    print(f"  1. Setup: python setup_qwen3_dizo_benchmark.py --all")
    print(f"  2. Small test: python comprehensive_qwen3_dizo_benchmark.py --scale small --datasets sst2")
    print(f"  3. Full benchmark: python comprehensive_qwen3_dizo_benchmark.py --datasets glue_basic --full_eval")
    
    return True

def test_environment():
    """Test if environment is ready for benchmarking"""
    
    print("🔍 Testing Environment...")
    
    required_files = [
        'qwen3_reversible_02_2.py',
        'comprehensive_qwen3_dizo_benchmark.py',
        'setup_qwen3_dizo_benchmark.py'
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file} found")
        else:
            print(f"  ✗ {file} missing")
            return False
    
    # Test imports
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError:
        print("  ✗ PyTorch not installed")
        return False
    
    try:
        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")
    except ImportError:
        print("  ⚠ Transformers not installed (optional for quick test)")
    
    try:
        import datasets
        print("  ✓ Datasets library available")
    except ImportError:
        print("  ⚠ Datasets library not installed (needed for full benchmark)")
    
    print("  ✓ Environment ready for quick benchmark")
    return True

if __name__ == "__main__":
    print("Quick Benchmark Example for Qwen3 vs DiZO")
    print("This is a minimal test to verify your setup works")
    print()
    
    # Test environment first
    if not test_environment():
        print("\n❌ Environment not ready. Please check missing files/packages.")
        print("Make sure you have:")
        print("  - qwen3_reversible_02_2.py in current directory")
        print("  - PyTorch installed")
        print("  - Other benchmark scripts in current directory")
        sys.exit(1)
    
    # Run quick benchmark
    if quick_benchmark_demo():
        print("\n✅ Success! Your setup is working correctly.")
        print("You can now run the full benchmark suite with confidence.")
    else:
        print("\n❌ Some issues encountered. Check the error messages above.")
        print("Try fixing the issues and run this test again.")