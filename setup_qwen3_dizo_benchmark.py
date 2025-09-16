"""
Setup Script for Qwen3 vs DiZO Benchmark
========================================

This script helps you set up the environment and run initial tests to ensure
everything is working correctly before running the full benchmark.

Usage:
python setup_qwen3_dizo_benchmark.py --install_deps
python setup_qwen3_dizo_benchmark.py --test_setup
python setup_qwen3_dizo_benchmark.py --quick_demo
"""

import subprocess
import sys
import os
import torch
import json
from pathlib import Path
import argparse

def install_dependencies():
    """Install required dependencies for the benchmark"""
    
    print("🔧 Installing required dependencies...")
    
    packages = [
        'torch>=1.13.0',
        'transformers>=4.21.0',
        'datasets>=2.0.0',
        'evaluate>=0.4.0',
        'rouge-score>=0.1.0',
        'nltk>=3.8',
        'scikit-learn>=1.0.0',
        'tqdm>=4.64.0',
        'numpy>=1.21.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0'
    ]
    
    for package in packages:
        try:
            print(f"  Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"  ✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to install {package}: {e}")
            print(f"    You may need to install manually: pip install {package}")
    
    # Download NLTK data
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        print("  ✓ NLTK data downloaded")
    except Exception as e:
        print(f"  ⚠ NLTK data download failed: {e}")

def test_setup():
    """Test that all components are working correctly"""
    
    print("🧪 Testing benchmark setup...")
    
    # Test 1: Import dependencies
    print("\n1. Testing imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch: {torch.__version__}")
    except ImportError:
        print("  ✗ PyTorch not available")
        return False
    
    try:
        import transformers
        print(f"  ✓ Transformers: {transformers.__version__}")
    except ImportError:
        print("  ✗ Transformers not available")
    
    try:
        import datasets
        print(f"  ✓ Datasets: {datasets.__version__}")
    except ImportError:
        print("  ✗ Datasets not available")
    
    try:
        import evaluate
        print("  ✓ Evaluate library available")
    except ImportError:
        print("  ✗ Evaluate library not available")
    
    try:
        from rouge_score import rouge_scorer
        print("  ✓ ROUGE scorer available")
    except ImportError:
        print("  ✗ ROUGE scorer not available")
    
    # Test 2: GPU availability
    print("\n2. Testing GPU availability...")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠ CUDA not available, will use CPU (much slower)")
    
    # Test 3: Model creation
    print("\n3. Testing model creation...")
    
    try:
        # Test if we can import the model creation function
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from qwen3_reversible_02_2 import create_reversible_qwen3_model
        print("  ✓ Model creation function imported")
        
        # Test creating a small model
        try:
            model = create_reversible_qwen3_model(
                vocab_size=1000,
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                use_reversible=True
            )
            print(f"  ✓ Reversible model created ({sum(p.numel() for p in model.parameters()):,} params)")
            
            model = create_reversible_qwen3_model(
                vocab_size=1000,
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                use_reversible=False
            )
            print(f"  ✓ Standard model created ({sum(p.numel() for p in model.parameters()):,} params)")
            
        except Exception as e:
            print(f"  ✗ Model creation failed: {e}")
            return False
            
    except ImportError as e:
        print(f"  ✗ Cannot import model creation: {e}")
        print("    Make sure qwen3_reversible_02_2.py is in the current directory")
        return False
    
    # Test 4: Dataset loading
    print("\n4. Testing dataset loading...")
    
    try:
        from datasets import load_dataset
        dataset = load_dataset('glue', 'sst2', split='train[:10]')  # Load small sample
        print(f"  ✓ GLUE SST-2 dataset loaded: {len(dataset)} examples")
    except Exception as e:
        print(f"  ✗ Dataset loading failed: {e}")
    
    # Test 5: Benchmark components
    print("\n5. Testing benchmark components...")
    
    benchmark_components = [
        ('glue_plus_benchmark', 'GLUEPlusBenchmark'),
        ('memory_benchmark', 'MemoryBenchmark'),
        ('advanced_benchmarks', 'AdvancedBenchmarkSuite'),
        ('run_advanced_benchmarks', 'ComprehensiveBenchmarkRunner')
    ]
    
    available_components = []
    for module_name, class_name in benchmark_components:
        try:
            module = __import__(module_name)
            getattr(module, class_name)
            print(f"  ✓ {class_name} available")
            available_components.append(module_name)
        except (ImportError, AttributeError):
            print(f"  ⚠ {class_name} not available")
    
    print(f"\n📊 Setup Summary:")
    print(f"  - Available components: {len(available_components)}/{len(benchmark_components)}")
    print(f"  - GPU available: {'Yes' if torch.cuda.is_available() else 'No'}")
    print(f"  - Model creation: {'Working' if 'qwen3_reversible_02_2' in sys.modules else 'Needs check'}")
    
    return True

def run_quick_demo():
    """Run a quick demonstration of the benchmark"""
    
    print("🚀 Running quick demo...")
    
    try:
        from comprehensive_qwen3_dizo_benchmark import ComprehensiveQwen3DiZOBenchmark, BenchmarkConfig
        
        # Create a minimal config for demo
        config = BenchmarkConfig(
            scale='small',
            num_epochs=1,
            eval_size=100,
            batch_size=4
        )
        
        # Initialize benchmark
        benchmark = ComprehensiveQwen3DiZOBenchmark(config, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        print("  ✓ Benchmark initialized")
        
        # Test model creation
        models = benchmark.trainer.create_models()
        if models:
            print(f"  ✓ Created {len(models)} models")
            
            # Quick forward pass test
            # Get device from model parameters (models don't have .device attribute)
            first_model = next(iter(models.values()))
            try:
                device = next(first_model.parameters()).device
            except StopIteration:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            test_input = torch.randint(0, 1000, (1, 32)).to(device)
            
            for name, model in models.items():
                try:
                    with torch.no_grad():
                        output = model(test_input)
                    
                    # Handle both tensor and dictionary outputs
                    if isinstance(output, dict):
                        if 'logits' in output:
                            logits = output['logits']
                            print(f"  ✓ {name} forward pass: logits shape {logits.shape}")
                        else:
                            # Print all available keys in the output
                            keys = list(output.keys())
                            print(f"  ✓ {name} forward pass: dict with keys {keys}")
                            # Try to get the first tensor for shape info
                            first_tensor = next((v for v in output.values() if hasattr(v, 'shape')), None)
                            if first_tensor is not None:
                                print(f"    First tensor shape: {first_tensor.shape}")
                    elif hasattr(output, 'shape'):
                        print(f"  ✓ {name} forward pass: {output.shape}")
                    else:
                        print(f"  ✓ {name} forward pass: output type {type(output)}")
                        
                except Exception as e:
                    print(f"  ✗ {name} forward pass failed: {e}")
                    import traceback
                    print(f"    Error details: {traceback.format_exc()[:200]}...")
        else:
            print("  ✗ No models created")
            return False
        
        print("\n🎉 Quick demo completed successfully!")
        print("\nYou're ready to run the full benchmark!")
        print("\nNext steps:")
        print("  1. Run small scale test: python comprehensive_qwen3_dizo_benchmark.py --scale small")
        print("  2. Run basic GLUE tasks: python comprehensive_qwen3_dizo_benchmark.py --datasets glue_basic")
        print("  3. Run full evaluation: python comprehensive_qwen3_dizo_benchmark.py --full_eval")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_config_template():
    """Create a configuration template file"""
    
    config_template = {
        "benchmark_settings": {
            "scale": "small",
            "datasets": ["sst2", "cola", "mrpc"],
            "num_epochs": 3,
            "batch_size": 16,
            "learning_rate": 1e-4
        },
        "model_settings": {
            "vocab_size": 32000,
            "hidden_size": 768,
            "num_layers": 12,
            "num_attention_heads": 12,
            "max_seq_length": 2048
        },
        "evaluation_settings": {
            "run_glue": True,
            "run_memory": True,
            "run_advanced": True,
            "compare_dizo": True
        },
        "output_settings": {
            "save_models": False,
            "output_dir": "./benchmark_results",
            "save_detailed_logs": True
        }
    }
    
    with open('benchmark_config_template.json', 'w') as f:
        json.dump(config_template, f, indent=2)
    
    print("📁 Created benchmark_config_template.json")
    print("   You can modify this file and use it with: --config benchmark_config.json")

def check_system_requirements():
    """Check system requirements"""
    
    print("💻 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"  ✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"  ✗ Python {python_version.major}.{python_version.minor} (requires 3.8+)")
        return False
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"  ✓ System RAM: {memory.total / 1e9:.1f} GB")
        
        if memory.available < 8e9:  # Less than 8 GB available
            print("  ⚠ Warning: Less than 8 GB RAM available, consider using --scale small")
    except ImportError:
        print("  ⚠ Cannot check memory (install psutil for memory info)")
    
    # Check disk space
    try:
        import shutil
        disk_usage = shutil.disk_usage('.')
        free_gb = disk_usage.free / 1e9
        print(f"  ✓ Free disk space: {free_gb:.1f} GB")
        
        if free_gb < 10:
            print("  ⚠ Warning: Less than 10 GB free space, models and datasets need space")
    except:
        print("  ⚠ Cannot check disk space")
    
    return True

def main():
    """Main setup function"""
    
    parser = argparse.ArgumentParser(description="Setup Qwen3 vs DiZO Benchmark")
    parser.add_argument('--install_deps', action='store_true', help='Install required dependencies')
    parser.add_argument('--test_setup', action='store_true', help='Test setup and components')
    parser.add_argument('--quick_demo', action='store_true', help='Run quick demonstration')
    parser.add_argument('--check_system', action='store_true', help='Check system requirements')
    parser.add_argument('--create_config', action='store_true', help='Create configuration template')
    parser.add_argument('--all', action='store_true', help='Run all setup steps')
    
    args = parser.parse_args()
    
    print("="*60)
    print("QWEN3 vs DiZO BENCHMARK SETUP")
    print("="*60)
    
    if args.all or args.check_system:
        check_system_requirements()
        print()
    
    if args.all or args.install_deps:
        install_dependencies()
        print()
    
    if args.all or args.test_setup:
        test_setup()
        print()
    
    if args.all or args.create_config:
        create_config_template()
        print()
    
    if args.all or args.quick_demo:
        run_quick_demo()
        print()
    
    if not any(vars(args).values()):
        print("No action specified. Use --help for options.")
        print("\nQuick start:")
        print("  python setup_qwen3_dizo_benchmark.py --all")

if __name__ == "__main__":
    main()