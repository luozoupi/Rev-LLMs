"""
Updated Performance Testing Framework for Qwen3 Reversible Models
================================================================

This framework incorporates the fixes for reversible training issues.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
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
    """Comprehensive performance testing for Reversible Qwen3 models with fixes"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
        
        # Updated training configuration for reversible models
        self.training_configs = {
            'reversible_standard': {
                'learning_rate': 0.003,      # Higher LR for reversible
                'weight_decay': 0.01,
                'warmup_ratio': 0.3,         # 30% warmup
                'gradient_clip_norm': 0.5,   # Tighter clipping
                'epochs': 50,                # More epochs needed
                'patience': 15,              # Early stopping patience
                'scheduler_type': 'onecycle',
                'mixed_precision': True
            },
            'reversible_candidate': {
                'learning_rate': 0.002,      # Slightly lower for candidate selection
                'weight_decay': 0.01,
                'warmup_ratio': 0.3,
                'gradient_clip_norm': 0.5,
                'epochs': 50,
                'patience': 15,
                'scheduler_type': 'onecycle',
                'mixed_precision': True
            },
            'reversible_sparse': {
                'learning_rate': 0.0025,     # Middle ground for sparse attention
                'weight_decay': 0.01,
                'warmup_ratio': 0.3,
                'gradient_clip_norm': 0.5,
                'epochs': 50,
                'patience': 15,
                'scheduler_type': 'onecycle',
                'mixed_precision': True
            },
            'non_reversible_standard': {
                'learning_rate': 0.0005,     # Lower LR for standard training
                'weight_decay': 0.01,
                'warmup_ratio': 0.1,
                'gradient_clip_norm': 1.0,
                'epochs': 30,
                'patience': 10,
                'scheduler_type': 'cosine',
                'mixed_precision': True
            }
        }
        
    def create_test_dataset(self, vocab_size=50000, seq_len=512, num_samples=1000):
        """Create synthetic dataset for testing"""
        
        # Create random sequences with realistic token distribution
        weights = np.array([1.0 / (i + 1) ** 0.8 for i in range(vocab_size)])
        weights /= weights.sum()
        
        data = []
        for _ in range(num_samples):
            sequence = np.random.choice(vocab_size, size=seq_len, p=weights)
            targets = np.roll(sequence, -1)
            data.append((torch.tensor(sequence, dtype=torch.long), 
                        torch.tensor(targets, dtype=torch.long)))
        
        return data
    
    def init_reversible_model(self, model):
        """Apply proper initialization for reversible models"""
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Scaled Xavier initialization for reversible models
                torch.nn.init.xavier_normal_(m.weight, gain=0.5)  # Reduced gain
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, std=0.02)
        
        model.apply(init_weights)
        
        # Special initialization for gradient scale parameters
        for module in model.modules():
            if hasattr(module, 'grad_scale'):
                torch.nn.init.ones_(module.grad_scale)
        
        return model
    
    def setup_models_for_comparison(self, vocab_size=10000, hidden_size=1024, 
                                  num_layers=6, num_heads=8):
        """Setup different model configurations for comparison"""
        
        from qwen3_reversible import create_reversible_qwen3_model
        
        models = {}
        
        # Model configurations to test
        configs = [
            ("reversible_standard", "standard", True),
            ("reversible_candidate", "candidate_selection", True), 
            ("reversible_sparse", "native_sparse", True),
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
                    candidate_pr_ratio=0.5,
                    candidate_top_k=40
                )
                
                # Apply proper initialization for reversible models
                if use_reversible:
                    model = self.init_reversible_model(model)
                
                model = model.to(self.device)
                models[name] = model
                print(f"✅ Created {name} model ({sum(p.numel() for p in model.parameters()):,} params)")
                
            except Exception as e:
                print(f"❌ Failed to create {name}: {e}")
                import traceback
                traceback.print_exc()
                
        return models
    
    def create_optimizer_and_scheduler(self, model, config, steps_per_epoch):
        """Create optimizer and scheduler based on model type"""
        
        # Use AdamW with proper settings for reversible models
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Choose scheduler based on config
        if config['scheduler_type'] == 'onecycle':
            scheduler = OneCycleLR(
                optimizer,
                max_lr=config['learning_rate'],
                epochs=config['epochs'],
                steps_per_epoch=steps_per_epoch,
                pct_start=config['warmup_ratio'],
                anneal_strategy='cos',
                div_factor=25,      # Start with lr/25
                final_div_factor=1000,  # End with lr/1000
            )
            step_scheduler_per_batch = True
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config['epochs'] * steps_per_epoch,
                eta_min=config['learning_rate'] / 100
            )
            step_scheduler_per_batch = True
            
        return optimizer, scheduler, step_scheduler_per_batch
    
    def fine_tune_model(self, model, train_dataloader, val_dataloader, model_name):
        """Improved fine-tuning with proper configuration for reversible models"""
        
        # Get model-specific config
        config = self.training_configs.get(model_name, self.training_configs['reversible_standard'])
        
        # Setup optimizer and scheduler
        steps_per_epoch = len(train_dataloader)
        optimizer, scheduler, step_per_batch = self.create_optimizer_and_scheduler(
            model, config, steps_per_epoch
        )
        
        # Mixed precision training
        scaler = GradScaler() if config['mixed_precision'] else None
        
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Training tracking
        train_losses = []
        val_losses = []
        epoch_times = []
        memory_usage = []
        gradient_norms = []
        learning_rates = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training {model_name} with config: LR={config['learning_rate']}, "
              f"Epochs={config['epochs']}, Scheduler={config['scheduler_type']}")
        
        for epoch in range(config['epochs']):
            model.train()
            epoch_start = time.time()
            epoch_loss = 0.0
            batch_count = 0
            
            # Reset memory stats
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            for batch_idx, (inputs, targets) in enumerate(train_dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                try:
                    optimizer.zero_grad()
                    
                    # Forward pass with mixed precision
                    if scaler:
                        with autocast():
                            outputs = model(inputs)
                            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                            
                            # Prepare for next token prediction
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = targets[..., 1:].contiguous()
                            
                            loss = criterion(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1)
                            )
                        
                        # Backward pass with scaling
                        scaler.scale(loss).backward()
                        
                        # Unscale for gradient clipping
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            max_norm=config['gradient_clip_norm']
                        )
                        
                        scaler.step(optimizer)
                        scaler.update()
                        
                    else:
                        # Standard precision training
                        outputs = model(inputs)
                        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                        
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = targets[..., 1:].contiguous()
                        
                        loss = criterion(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                        
                        loss.backward()
                        
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            max_norm=config['gradient_clip_norm']
                        )
                        
                        optimizer.step()
                    
                    # Step scheduler if per-batch
                    if step_per_batch:
                        scheduler.step()
                    
                    # Track metrics
                    epoch_loss += loss.item()
                    gradient_norms.append(grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm)
                    learning_rates.append(optimizer.param_groups[0]['lr'])
                    batch_count += 1
                    
                    # Limit batches for testing (remove this for full training)
                    if batch_idx >= 20:
                        break
                        
                except Exception as e:
                    print(f"Training error at batch {batch_idx}: {e}")
                    continue
            
            # Step scheduler if per-epoch
            if not step_per_batch:
                scheduler.step()
            
            # Record timing and memory
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                reserved_memory = torch.cuda.max_memory_reserved() / 1024**2  # MB
                memory_usage.append({'peak': peak_memory, 'reserved': reserved_memory})
            
            avg_train_loss = epoch_loss / max(batch_count, 1)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            val_loss = self.validate_model(model, val_dataloader, criterion)
            val_losses.append(val_loss)
            
            # Print progress
            peak_mem = memory_usage[-1]['peak'] if memory_usage else 0
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s, "
                  f"Memory: {peak_mem:.1f}MB, LR: {learning_rates[-1]:.6f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    print(f"Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.4f})")
                    break
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epoch_times': epoch_times,
            'memory_usage': memory_usage,
            'gradient_norms': gradient_norms,
            'learning_rates': learning_rates,
            'best_val_loss': best_val_loss
        }
    
    def validate_model(self, model, val_dataloader, criterion):
        """Validation with proper error handling"""
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_dataloader):
                if batch_idx >= 10:  # Limit validation batches for testing
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
        
        return val_loss / max(val_batches, 1)
    
    def calculate_perplexity(self, model, dataloader, max_batches=50):
        """Calculate perplexity with improved error handling"""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
        
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
                    
                    # Count valid tokens (not padding)
                    valid_tokens = (shift_labels != -100).sum().item()
                    if valid_tokens == 0:
                        continue
                    
                    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), 
                                   shift_labels.view(-1))
                    
                    total_loss += loss.item()
                    total_tokens += valid_tokens
                    
                except Exception as e:
                    print(f"Perplexity calculation error: {e}")
                    continue
        
        if total_tokens == 0:
            return float('inf'), float('inf')
            
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity, avg_loss
    
    def run_comprehensive_test(self, seq_len=512, vocab_size=10000):
        """Run comprehensive performance comparison with fixes"""
        
        print("=" * 60)
        print("COMPREHENSIVE REVERSIBLE QWEN3 PERFORMANCE TEST (WITH FIXES)")
        print("=" * 60)
        
        # Create test datasets
        print("Creating test datasets...")
        train_data = self.create_test_dataset(vocab_size, seq_len, 400)  # More samples
        val_data = self.create_test_dataset(vocab_size, seq_len, 100)
        test_data = self.create_test_dataset(vocab_size, seq_len, 100)
        
        # Use smaller batch sizes for reversible models
        train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
        
        # Setup models
        print("Setting up models...")
        models = self.setup_models_for_comparison(
            vocab_size=vocab_size,
            hidden_size=512,  # Smaller for testing
            num_layers=6,     # Fewer layers for quick testing
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
                
                # Fine-tune model with appropriate config
                print("Starting fine-tuning...")
                training_results = self.fine_tune_model(model, train_loader, val_loader, model_name)
                
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
                    memory_peak_mb=max([m['peak'] for m in training_results['memory_usage']] + [0]),
                    memory_reserved_mb=max([m['reserved'] for m in training_results['memory_usage']] + [0]),
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
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        
        model_names = list(results.keys())
        
        # 1. Perplexity Comparison (Log scale for better visualization)
        perplexities = [results[name]['metrics'].perplexity for name in model_names]
        axes[0, 0].bar(model_names, perplexities)
        axes[0, 0].set_title('Final Perplexity (Lower is Better)')
        axes[0, 0].set_ylabel('Perplexity (log scale)')
        axes[0, 0].set_yscale('log')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add values on bars
        for i, v in enumerate(perplexities):
            axes[0, 0].text(i, v, f'{v:.1f}', ha='center', va='bottom')
        
        # 2. Memory Usage
        memory_usage = [results[name]['metrics'].memory_peak_mb for name in model_names]
        bars = axes[0, 1].bar(model_names, memory_usage)
        axes[0, 1].set_title('Peak Memory Usage')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Color bars differently for reversible vs non-reversible
        for i, name in enumerate(model_names):
            if 'reversible' in name and 'non_reversible' not in name:
                bars[i].set_color('green')
            else:
                bars[i].set_color('red')
        
        # 3. Training Time per Epoch
        train_times = [results[name]['metrics'].training_time_per_epoch for name in model_names]
        axes[0, 2].bar(model_names, train_times)
        axes[0, 2].set_title('Training Time per Epoch')
        axes[0, 2].set_ylabel('Time (seconds)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Perplexity Improvement
        improvements = [results[name]['improvement'] for name in model_names]
        bars = axes[1, 0].bar(model_names, improvements)
        axes[1, 0].set_title('Perplexity Improvement')
        axes[1, 0].set_ylabel('PPL Reduction')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Highlight negative improvements (worse performance)
        for i, imp in enumerate(improvements):
            if imp < 0:
                bars[i].set_color('red')
            else:
                bars[i].set_color('green')
        
        # 5. Training Loss Curves
        for name in model_names:
            if 'training_curves' in results[name]:
                curves = results[name]['training_curves']
                if 'train_losses' in curves and 'val_losses' in curves:
                    epochs = range(1, len(curves['train_losses']) + 1)
                    axes[1, 1].plot(epochs, curves['train_losses'], 
                                  label=f'{name} (train)', linewidth=2)
                    axes[1, 1].plot(epochs, curves['val_losses'], '--', 
                                  label=f'{name} (val)', linewidth=2)
        axes[1, 1].set_title('Training Loss Curves')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].set_yscale('log')  # Log scale for better visualization
        
        # 6. Learning Rate Schedule
        for name in model_names:
            if 'training_curves' in results[name]:
                curves = results[name]['training_curves']
                if 'learning_rates' in curves:
                    epochs = range(1, len(curves['learning_rates']) + 1)
                    axes[1, 2].plot(epochs, curves['learning_rates'], 
                                  label=name, linewidth=2)
        axes[1, 2].set_title('Learning Rate Schedule')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].legend()
        axes[1, 2].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('reversible_qwen_comparison_fixed.png', dpi=300, bbox_inches='tight')
        return fig
    
    def save_results(self, results, filename='reversible_qwen_performance_results_fixed.json'):
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


# Quick diagnostic test to verify fixes
def quick_gradient_test():
    """Quick test to verify gradient flow is working"""
    print("Running gradient flow diagnostic test...")
    
    from qwen3_reversible import create_reversible_qwen3_model
    
    # Create small test model
    model = create_reversible_qwen3_model(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        attention_type="standard",
        use_reversible=True,
        reverse_thres=0  # Force reversible mode
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create test input
    batch_size, seq_len = 2, 64
    inputs = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    targets = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    
    # Test forward and backward
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for step in range(5):
        optimizer.zero_grad()
        
        outputs = model(inputs)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), 
                        shift_labels.view(-1))
        
        loss.backward()
        
        # Check gradients
        total_grad_norm = 0
        param_count = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
                param_count += 1
        
        total_grad_norm = total_grad_norm ** 0.5
        
        optimizer.step()
        
        print(f"Step {step+1}: Loss = {loss.item():.4f}, "
              f"Grad Norm = {total_grad_norm:.4f}, "
              f"Params with grad = {param_count}")
    
    print("✅ Gradient flow test completed!")


# Example usage and testing script
if __name__ == "__main__":
    
    # First run diagnostic test
    quick_gradient_test()
    
    # Then run comprehensive test
    tester = ReversibleQwenPerformanceTester()
    
    print("\nRunning comprehensive performance test with fixes...")
    results = tester.run_comprehensive_test(
        seq_len=512,
        vocab_size=10000
    )
    
    if results:
        print("\n" + "="*60)
        print("SUMMARY OF RESULTS (AFTER FIXES)")
        print("="*60)
        
        for name, result in results.items():
            metrics = result['metrics']
            print(f"\n{name.upper()}:")
            print(f"  Final Perplexity: {metrics.perplexity:.2f}")
            print(f"  Peak Memory: {metrics.memory_peak_mb:.1f} MB")
            print(f"  Training Time/Epoch: {metrics.training_time_per_epoch:.1f}s")
            print(f"  Perplexity Improvement: {result['improvement']:.2f}")
            print(f"  Best Val Loss: {result['training_curves']['best_val_loss']:.4f}")
        
        # Create visualization
        fig = tester.visualize_results(results)
        plt.show()
        
        # Save results
        tester.save_results(results)
        
        print("\n🎉 Performance testing with fixes completed!")
        print("\nExpected improvements:")
        print("- Reversible models should now achieve perplexity < 100")
        print("- Training loss should decrease steadily")
        print("- Memory usage should remain low")