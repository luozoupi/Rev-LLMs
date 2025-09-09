"""
Performance Testing Framework for Qwen3 Reversible Models
========================================================

This framework provides comprehensive testing for fine-tuning performance,
perplexity measurements, and memory efficiency analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class PerformanceMetrics:
    """Store comprehensive performance metrics"""
    model_name: str
    perplexity: float
    loss: float
    accuracy: float
    memory_peak_mb: float
    memory_reserved_mb: float
    training_time_per_epoch: float
    inference_time_per_token: float
    gradient_norm: float
    
    # Model-specific metrics
    attention_sparsity: Optional[float] = None
    pruning_ratio: Optional[float] = None
    reversible_layers_used: Optional[int] = None

class ReversibleQwenPerformanceTester:
    """Comprehensive performance testing for Reversible Qwen3 models"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
        
    def create_test_dataset(self, vocab_size=50000, seq_len=512, num_samples=1000):
        """Create synthetic dataset for testing"""
        
        # Create random sequences with realistic token distribution
        # Use power-law distribution to mimic natural language
        weights = np.array([1.0 / (i + 1) ** 0.8 for i in range(vocab_size)])
        weights /= weights.sum()
        
        data = []
        for _ in range(num_samples):
            sequence = np.random.choice(vocab_size, size=seq_len, p=weights)
            # Create targets (next token prediction)
            targets = np.roll(sequence, -1)
            data.append((torch.tensor(sequence), torch.tensor(targets)))
        
        return data
    
    def setup_models_for_comparison(self, vocab_size=10000, hidden_size=1024, 
                                  num_layers=6, num_heads=8):
        """Setup different model configurations for comparison"""
        
        from qwen3_reversible import create_reversible_qwen3_model
        
        models = {}
        
        # 1. Reversible with different attention types
        configs = [
            ("reversible_standard", "standard", True),
            ("reversible_candidate", "candidate_selection", True), 
            ("reversible_sparse", "native_sparse", True),
            ("non_reversible_standard", "standard", False)  # For comparison if memory allows
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
                    candidate_pr_ratio=0.5,
                    candidate_top_k=40
                )
                model = model.to(self.device)
                models[name] = model
                print(f"✅ Created {name} model ({sum(p.numel() for p in model.parameters()):,} params)")
                
            except Exception as e:
                print(f"❌ Failed to create {name}: {e}")
                
        return models
    
    def calculate_perplexity(self, model, dataloader, max_batches=50):
        """Calculate perplexity on test data"""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                    
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                try:
                    outputs = model(inputs)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    # Calculate loss for perplexity
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = targets[..., 1:].contiguous()
                    
                    loss = criterion(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    total_loss += loss.item() * inputs.numel()
                    total_tokens += inputs.numel()
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return perplexity.item(), avg_loss
    
    def fine_tune_model(self, model, train_dataloader, val_dataloader, 
                       epochs=30, lr=1e-3):
        """Fine-tune model and collect metrics"""
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        train_losses = []
        val_losses = []
        epoch_times = []
        memory_usage = []
        gradient_norms = []
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            epoch_start = time.time()
            epoch_loss = 0.0
            
            torch.cuda.reset_peak_memory_stats()
            
            for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataloader, 
                                                              desc=f"Epoch {epoch+1}")):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                
                try:
                    outputs = model(inputs)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    # Shift for next token prediction
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = targets[..., 1:].contiguous()
                    
                    loss = criterion(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    loss.backward()
                    
                    # Track gradient norm
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    gradient_norms.append(grad_norm.item())
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    # Limit batches for testing
                    if batch_idx >= 20:  # Limit for quick testing
                        break
                        
                except Exception as e:
                    print(f"Training error at batch {batch_idx}: {e}")
                    continue
            
            scheduler.step()
            
            # Record timing and memory
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            reserved_memory = torch.cuda.max_memory_reserved() / 1024**2  # MB
            memory_usage.append({'peak': peak_memory, 'reserved': reserved_memory})
            
            avg_train_loss = epoch_loss / min(len(train_dataloader), 21)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_dataloader):
                    if batch_idx >= 10:  # Limit validation batches
                        break
                        
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    try:
                        outputs = model(inputs)
                        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                        
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = targets[..., 1:].contiguous()
                        
                        loss = criterion(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                        
                        val_loss += loss.item()
                        val_batches += 1
                        
                    except Exception as e:
                        print(f"Validation error: {e}")
                        continue
            
            avg_val_loss = val_loss / max(val_batches, 1)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s, "
                  f"Memory: {peak_memory:.1f}MB")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epoch_times': epoch_times,
            'memory_usage': memory_usage,
            'gradient_norms': gradient_norms
        }
    
    def run_comprehensive_test(self, seq_len=512, vocab_size=10000):
        """Run comprehensive performance comparison"""
        
        print("="*60)
        print("COMPREHENSIVE REVERSIBLE QWEN3 PERFORMANCE TEST")
        print("="*60)
        
        # Create test datasets
        print("Creating test datasets...")
        train_data = self.create_test_dataset(vocab_size, seq_len, 200)
        val_data = self.create_test_dataset(vocab_size, seq_len, 50)
        test_data = self.create_test_dataset(vocab_size, seq_len, 100)
        
        train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
        
        # Setup models
        print("Setting up models...")
        models = self.setup_models_for_comparison(
            vocab_size=vocab_size,
            hidden_size=512,  # Smaller for testing
            num_layers=4,     # Fewer layers for quick testing
            num_heads=8
        )
        
        # Test each model
        all_results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*40}")
            print(f"Testing {model_name}")
            print(f"{'='*40}")
            
            try:
                # Calculate baseline perplexity
                print("Calculating baseline perplexity...")
                baseline_ppl, baseline_loss = self.calculate_perplexity(model, test_loader)
                print(f"Baseline perplexity: {baseline_ppl:.2f}")
                
                # Fine-tune model
                print("Starting fine-tuning...")
                training_results = self.fine_tune_model(model, train_loader, val_loader)
                
                # Calculate post-training perplexity
                print("Calculating post-training perplexity...")
                final_ppl, final_loss = self.calculate_perplexity(model, test_loader)
                print(f"Final perplexity: {final_ppl:.2f}")
                
                # Collect comprehensive metrics
                metrics = PerformanceMetrics(
                    model_name=model_name,
                    perplexity=final_ppl,
                    loss=final_loss,
                    accuracy=0.0,  # Would need additional calculation
                    memory_peak_mb=max([m['peak'] for m in training_results['memory_usage']]),
                    memory_reserved_mb=max([m['reserved'] for m in training_results['memory_usage']]),
                    training_time_per_epoch=np.mean(training_results['epoch_times']),
                    inference_time_per_token=0.0,  # Would need separate measurement
                    gradient_norm=np.mean(training_results['gradient_norms'])
                )
                
                all_results[model_name] = {
                    'metrics': metrics,
                    'training_curves': training_results,
                    'baseline_perplexity': baseline_ppl,
                    'improvement': baseline_ppl - final_ppl
                }
                
                print(f"✅ {model_name} completed successfully")
                
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return all_results
    
    def visualize_results(self, results):
        """Create comprehensive visualization of results"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        model_names = list(results.keys())
        
        # 1. Perplexity Comparison
        perplexities = [results[name]['metrics'].perplexity for name in model_names]
        axes[0, 0].bar(model_names, perplexities)
        axes[0, 0].set_title('Final Perplexity')
        axes[0, 0].set_ylabel('Perplexity')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Memory Usage
        memory_usage = [results[name]['metrics'].memory_peak_mb for name in model_names]
        axes[0, 1].bar(model_names, memory_usage)
        axes[0, 1].set_title('Peak Memory Usage')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Training Time
        train_times = [results[name]['metrics'].training_time_per_epoch for name in model_names]
        axes[0, 2].bar(model_names, train_times)
        axes[0, 2].set_title('Training Time per Epoch')
        axes[0, 2].set_ylabel('Time (seconds)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Perplexity Improvement
        improvements = [results[name]['improvement'] for name in model_names]
        axes[1, 0].bar(model_names, improvements)
        axes[1, 0].set_title('Perplexity Improvement')
        axes[1, 0].set_ylabel('PPL Reduction')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Training Loss Curves
        for name in model_names:
            if 'training_curves' in results[name]:
                curves = results[name]['training_curves']
                axes[1, 1].plot(curves['train_losses'], label=f'{name} (train)')
                axes[1, 1].plot(curves['val_losses'], '--', label=f'{name} (val)')
        axes[1, 1].set_title('Training Curves')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        
        # 6. Gradient Norm
        grad_norms = [results[name]['metrics'].gradient_norm for name in model_names]
        axes[1, 2].bar(model_names, grad_norms)
        axes[1, 2].set_title('Average Gradient Norm')
        axes[1, 2].set_ylabel('Gradient Norm')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def save_results(self, results, filename='reversible_qwen_performance_results.json'):
        """Save results to file"""
        
        # Convert to serializable format
        serializable_results = {}
        for name, result in results.items():
            serializable_results[name] = {
                'metrics': result['metrics'].__dict__,
                'baseline_perplexity': result['baseline_perplexity'],
                'improvement': result['improvement'],
                'training_curves': {
                    k: v for k, v in result['training_curves'].items()
                    if k != 'memory_usage'  # Skip complex nested structure
                }
            }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {filename}")



# Example usage and testing script
if __name__ == "__main__":
    # Quick test with small parameters
    tester = ReversibleQwenPerformanceTester()
    
    print("Running small-scale performance test...")
    results = tester.run_comprehensive_test(
        seq_len=256*5*2,  # Start small
        vocab_size=5000*2*2
    )
    
    if results:
        print("\n" + "="*60)
        print("SUMMARY OF RESULTS")
        print("="*60)
        
        for name, result in results.items():
            metrics = result['metrics']
            print(f"\n{name.upper()}:")
            print(f"  Final Perplexity: {metrics.perplexity:.2f}")
            print(f"  Peak Memory: {metrics.memory_peak_mb:.1f} MB")
            print(f"  Training Time/Epoch: {metrics.training_time_per_epoch:.1f}s")
            print(f"  Perplexity Improvement: {result['improvement']:.2f}")
        
        # Create visualization
        fig = tester.visualize_results(results)
        plt.savefig('reversible_qwen_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save results
        tester.save_results(results)
        
        print("\n✅ Performance testing completed!")
        print("📊 Visualization saved as 'reversible_qwen_performance_comparison.png'")
        print("💾 Results saved as 'reversible_qwen_performance_results.json'")

        # from qwen3_reversible import create_fixed_reversible_qwen3_model, get_reversible_training_config, train_reversible_model
        # model = create_fixed_reversible_qwen3_model(
        #     vocab_size=20000,
        #     hidden_size=768,
        #     num_hidden_layers=12,
        #     use_reversible=True,
        #     attention_type="candidate_selection"
        # )
        
        # # Get training config
        # config = get_reversible_training_config()
        
        # # Train with fixes
        # train_reversible_model(model, train_loader, val_loader, config)