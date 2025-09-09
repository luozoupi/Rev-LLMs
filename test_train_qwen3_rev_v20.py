"""
Performance Testing Framework for Qwen3 Reversible Models - FIXED VERSION
========================================================================

This framework provides comprehensive testing for fine-tuning performance,
perplexity measurements, and memory efficiency analysis.
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
            data.append((torch.tensor(sequence, dtype=torch.long), 
                        torch.tensor(targets, dtype=torch.long)))
        
        return data
    
    def get_optimized_training_config(self, model_type="reversible"):
        """Get optimized training configuration based on model type"""
        
        if "reversible" in model_type.lower():
            return {
                # FIXED: Higher learning rate for reversible models
                'learning_rate': 0.003,  # 30x higher than original
                'min_learning_rate': 1e-6,
                
                # FIXED: Proper scheduler
                'scheduler_type': 'onecycle',  # Better than ReduceLROnPlateau
                'warmup_ratio': 0.1,  # 10% warmup
                'warmup_steps': 100,
                
                # FIXED: Better optimizer settings
                'optimizer': 'adamw',
                'weight_decay': 0.1,  # Stronger regularization
                'betas': (0.9, 0.95),  # Better momentum for reversible
                'eps': 1e-8,
                
                # FIXED: Gradient handling
                'gradient_clip_norm': 0.5,  # Tighter clipping
                'gradient_accumulation_steps': 2,
                
                # FIXED: Training duration
                'epochs': 50,  # More epochs needed
                'patience': 15,  # Early stopping patience
                
                # Mixed precision
                'use_amp': True,
                'amp_dtype': torch.float16,
                
                # Batch settings
                'batch_size': 8,
                'val_batch_size': 4,
            }
        else:
            # Standard model config
            return {
                'learning_rate': 0.0001,
                'min_learning_rate': 1e-6,
                'scheduler_type': 'cosine',
                'warmup_ratio': 0.05,
                'warmup_steps': 50,
                'optimizer': 'adamw',
                'weight_decay': 0.01,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'gradient_clip_norm': 1.0,
                'gradient_accumulation_steps': 1,
                'epochs': 30,
                'patience': 10,
                'use_amp': True,
                'amp_dtype': torch.float16,
                'batch_size': 8,
                'val_batch_size': 4,
            }
    
    def initialize_model_weights(self, model, model_type="reversible"):
        """Proper initialization for different model types"""
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                if "reversible" in model_type.lower():
                    # FIXED: Scaled initialization for reversible models
                    torch.nn.init.xavier_normal_(m.weight, gain=0.3)  # Much smaller gain
                else:
                    torch.nn.init.xavier_normal_(m.weight, gain=1.0)
                    
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                    
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                std = 0.01 if "reversible" in model_type.lower() else 0.02
                torch.nn.init.normal_(m.weight, std=std)
        
        model.apply(init_weights)
        
        # Special handling for reversible blocks
        for module in model.modules():
            if hasattr(module, 'grad_scale'):
                torch.nn.init.constant_(module.grad_scale, 0.1)  # Start small
                
        return model
    
    def setup_models_for_comparison(self, vocab_size=10000, hidden_size=1024, 
                                  num_layers=6, num_heads=8):
        """Setup different model configurations for comparison"""
        
        from qwen3_reversible_0 import create_reversible_qwen3_model
        
        models = {}
        
        # FIXED: Updated model configurations with better parameters
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
                    reverse_thres=128 if use_reversible else 999999,  # Lower threshold
                    candidate_pr_ratio=0.3,  # More conservative
                    candidate_top_k=32  # Smaller top-k
                )
                
                # FIXED: Proper initialization
                model = self.initialize_model_weights(model, name)
                model = model.to(self.device)
                models[name] = model
                
                param_count = sum(p.numel() for p in model.parameters())
                print(f"✅ Created {name} model ({param_count:,} params)")
                
            except Exception as e:
                print(f"❌ Failed to create {name}: {e}")
                
        return models
    
    def create_scheduler(self, optimizer, config, steps_per_epoch):
        """Create appropriate learning rate scheduler"""
        
        total_steps = config['epochs'] * steps_per_epoch
        
        if config['scheduler_type'] == 'onecycle':
            scheduler = OneCycleLR(
                optimizer,
                max_lr=config['learning_rate'],
                total_steps=total_steps,
                pct_start=config['warmup_ratio'],
                anneal_strategy='cos',
                div_factor=25.0,  # Start with lr/25
                final_div_factor=1000.0,  # End with lr/1000
                cycle_momentum=True
            )
        elif config['scheduler_type'] == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=config['min_learning_rate']
            )
        else:
            # Fallback to step scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
            
        return scheduler
    
    def fine_tune_model(self, model, train_loader, val_loader, max_epochs=50):
        """FIXED: Fine-tune model with proper reversible training techniques"""
        
        model_type = "reversible" if hasattr(model, 'use_reversible') and model.use_reversible else "standard"
        config = self.get_optimized_training_config(model_type)
        
        # FIXED: Better optimizer setup
        if config['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay'],
                betas=config['betas'],
                eps=config['eps']
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # FIXED: Proper scheduler
        scheduler = self.create_scheduler(optimizer, config, len(train_loader))
        
        # FIXED: Mixed precision setup
        scaler = GradScaler() if config['use_amp'] else None
        
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Training tracking
        train_losses = []
        val_losses = []
        epoch_times = []
        memory_usage = []
        gradient_norms = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training with config: LR={config['learning_rate']}, "
              f"Scheduler={config['scheduler_type']}, Epochs={min(max_epochs, config['epochs'])}")
        
        epochs_to_run = min(max_epochs, config['epochs'])
        
        for epoch in range(epochs_to_run):
            model.train()
            epoch_start = time.time()
            
            # Reset memory tracking
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            total_train_loss = 0
            num_train_batches = 0
            total_grad_norm = 0
            
            optimizer.zero_grad()
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                try:
                    # FIXED: Proper mixed precision training
                    if config['use_amp'] and scaler is not None:
                        with autocast(dtype=config['amp_dtype']):
                            outputs = model(inputs)
                            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                            
                            # Next token prediction loss
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = targets[..., 1:].contiguous()
                            loss = criterion(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1)
                            )
                        
                        # Scale and backward
                        scaled_loss = scaler.scale(loss / config['gradient_accumulation_steps'])
                        scaled_loss.backward()
                    else:
                        # Standard training
                        outputs = model(inputs)
                        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                        
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = targets[..., 1:].contiguous()
                        loss = criterion(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                        
                        (loss / config['gradient_accumulation_steps']).backward()
                    
                    total_train_loss += loss.item()
                    num_train_batches += 1
                    
                    # Update weights
                    if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                        
                        if config['use_amp'] and scaler is not None:
                            # Mixed precision gradient update
                            scaler.unscale_(optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(), config['gradient_clip_norm']
                            )
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            # Standard gradient update
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(), config['gradient_clip_norm']
                            )
                            optimizer.step()
                        
                        scheduler.step()  # Step every update for OneCycle
                        optimizer.zero_grad()
                        
                        total_grad_norm += grad_norm.item()
                    
                    # Progress reporting
                    if batch_idx % 10 == 0:
                        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else config['learning_rate']
                        print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
                        
                except Exception as e:
                    print(f"Training error at batch {batch_idx}: {e}")
                    continue
            
            avg_train_loss = total_train_loss / max(num_train_batches, 1)
            avg_grad_norm = total_grad_norm / max(num_train_batches // config['gradient_accumulation_steps'], 1)
            epoch_time = time.time() - epoch_start
            
            # Record memory usage
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
                reserved_memory = torch.cuda.max_memory_reserved() / (1024**2)
                memory_usage.append({'peak': peak_memory, 'reserved': reserved_memory})
            else:
                memory_usage.append({'peak': 0, 'reserved': 0})
            
            # Validation
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    try:
                        if config['use_amp']:
                            with autocast(dtype=config['amp_dtype']):
                                outputs = model(inputs)
                                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                                
                                shift_logits = logits[..., :-1, :].contiguous()
                                shift_labels = targets[..., 1:].contiguous()
                                loss = criterion(
                                    shift_logits.view(-1, shift_logits.size(-1)),
                                    shift_labels.view(-1)
                                )
                        else:
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
            
            # Record metrics
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            epoch_times.append(epoch_time)
            gradient_norms.append(avg_grad_norm)
            
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else config['learning_rate']
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s, "
                  f"Memory: {memory_usage[-1]['peak']:.1f}MB, "
                  f"Grad Norm: {avg_grad_norm:.4f}, LR: {current_lr:.6f}")
            
            # Early stopping with patience
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    print(f"Early stopping at epoch {epoch+1} (patience {config['patience']})")
                    break
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epoch_times': epoch_times,
            'memory_usage': memory_usage,
            'gradient_norms': gradient_norms
        }
    
    def calculate_perplexity(self, model, dataloader, max_batches=50):
        """Calculate perplexity on test data"""
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
                    
                    # Calculate loss for perplexity (next token prediction)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = targets[..., 1:].contiguous()
                    
                    loss = criterion(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    # Count valid tokens (not padded)
                    valid_tokens = (shift_labels.view(-1) != -100).sum().item()
                    
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
        
        print("="*60)
        print("COMPREHENSIVE REVERSIBLE QWEN3 PERFORMANCE TEST - FIXED")
        print("="*60)
        
        # Create test datasets
        print("Creating test datasets...")
        train_data = self.create_test_dataset(vocab_size, seq_len, 500)  # More data
        val_data = self.create_test_dataset(vocab_size, seq_len, 100)
        test_data = self.create_test_dataset(vocab_size, seq_len, 200)
        
        train_loader = DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
        
        # Setup models
        print("Setting up models...")
        models = self.setup_models_for_comparison(
            vocab_size=vocab_size,
            hidden_size=768,  # Slightly larger
            num_layers=6,     # Keep manageable for testing
            num_heads=12
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
                
                # Fine-tune model with fixes
                print("Starting fine-tuning...")
                training_results = self.fine_tune_model(model, train_loader, val_loader, max_epochs=50)
                
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
                    memory_peak_mb=max([m['peak'] for m in training_results['memory_usage']]) if training_results['memory_usage'] else 0,
                    memory_reserved_mb=max([m['reserved'] for m in training_results['memory_usage']]) if training_results['memory_usage'] else 0,
                    training_time_per_epoch=np.mean(training_results['epoch_times']) if training_results['epoch_times'] else 0,
                    inference_time_per_token=0.0,  # Would need separate measurement
                    gradient_norm=np.mean(training_results['gradient_norms']) if training_results['gradient_norms'] else 0
                )
                
                all_results[model_name] = {
                    'metrics': metrics,
                    'training_curves': training_results,
                    'baseline_perplexity': baseline_ppl,
                    'improvement': baseline_ppl - final_ppl
                }
                
                print(f"✅ {model_name} completed successfully")
                print(f"   Improvement: {baseline_ppl - final_ppl:.2f} perplexity reduction")
                
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return all_results
    
    def visualize_results(self, results):
        """Create comprehensive visualization of results"""
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        model_names = list(results.keys())
        
        # 1. Perplexity Comparison (Log scale for better visualization)
        perplexities = [results[name]['metrics'].perplexity for name in model_names]
        axes[0, 0].bar(model_names, perplexities)
        axes[0, 0].set_title('Final Perplexity (Lower is Better)')
        axes[0, 0].set_ylabel('Perplexity')
        axes[0, 0].set_yscale('log')  # Log scale for better comparison
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
        
        # 5. Training Loss Curves (Log scale)
        for name in model_names:
            if 'training_curves' in results[name]:
                curves = results[name]['training_curves']
                if curves['train_losses'] and curves['val_losses']:
                    axes[1, 1].plot(curves['train_losses'], label=f'{name} (train)')
                    axes[1, 1].plot(curves['val_losses'], '--', label=f'{name} (val)')
        axes[1, 1].set_title('Training Curves')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_yscale('log')  # Log scale for better visualization
        axes[1, 1].legend()
        
        # 6. Gradient Norm
        grad_norms = [results[name]['metrics'].gradient_norm for name in model_names]
        axes[1, 2].bar(model_names, grad_norms)
        axes[1, 2].set_title('Average Gradient Norm')
        axes[1, 2].set_ylabel('Gradient Norm')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
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


# DEBUGGING FUNCTIONS
# ===================

def debug_gradient_flow(model, sample_input):
    """Debug gradient flow in reversible models"""
    print("🔍 Debugging gradient flow...")
    
    # Enable gradient tracking
    for param in model.parameters():
        param.requires_grad_(True)
    
    model.train()
    
    # Forward pass
    outputs = model(sample_input)
    loss = outputs['logits'].sum() if isinstance(outputs, dict) else outputs.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    total_params = 0
    params_with_grad = 0
    grad_norms = []
    
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            params_with_grad += 1
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            
            if grad_norm == 0:
                print(f"⚠️  Zero gradient: {name}")
            elif torch.isnan(param.grad).any():
                print(f"⚠️  NaN gradient: {name}")
    
    print(f"Parameters with gradients: {params_with_grad}/{total_params}")
    print(f"Average gradient norm: {np.mean(grad_norms):.6f}")
    
    if params_with_grad < total_params * 0.8:
        print("❌ PROBLEM: Many parameters missing gradients!")
    else:
        print("✅ Gradient flow looks good")


def quick_convergence_test(model, train_loader):
    """Quick test to check if model can overfit small batch"""
    print("🚀 Running quick convergence test...")
    
    model.train()
    
    # Get one batch and try to overfit it
    inputs, targets = next(iter(train_loader))
    inputs, targets = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), targets.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    config = {
        'learning_rate': 0.01,  # High LR for overfitting test
        'optimizer': 'adamw',
        'weight_decay': 0.0,
        'gradient_clip_norm': 1.0,
        'use_amp': False  # Disable for debugging
    }
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    initial_loss = None
    
    for step in range(100):
        optimizer.zero_grad()
        
        outputs = model(inputs)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if step == 0:
            initial_loss = loss.item()
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step}: Loss {loss.item():.4f}")
    
    final_loss = loss.item()
    improvement = initial_loss - final_loss
    
    if improvement > initial_loss * 0.5:  # 50% improvement
        print(f"✅ Model can learn! Loss: {initial_loss:.4f} → {final_loss:.4f}")
        return True
    else:
        print(f"❌ Model struggling to learn. Loss: {initial_loss:.4f} → {final_loss:.4f}")
        return False


# MAIN TESTING SCRIPT
# ===================

if __name__ == "__main__":
    # Create tester
    tester = ReversibleQwenPerformanceTester()
    
    print("🧪 Running FIXED reversible training test...")
    
    # Smaller test first
    results = tester.run_comprehensive_test(
        seq_len=512,  # Reasonable size
        vocab_size=8000  # Smaller vocab for faster testing
    )
    
    if results:
        print("\n" + "="*60)
        print("SUMMARY OF FIXED RESULTS")
        print("="*60)
        
        for name, result in results.items():
            metrics = result['metrics']
            print(f"\n{name.upper()}:")
            print(f"  Final Perplexity: {metrics.perplexity:.2f}")
            print(f"  Peak Memory: {metrics.memory_peak_mb:.1f} MB")
            print(f"  Training Time/Epoch: {metrics.training_time_per_epoch:.1f}s")
            print(f"  Perplexity Improvement: {result['improvement']:.2f}")
            print(f"  Gradient Norm: {metrics.gradient_norm:.4f}")
        
        # Create visualization
        fig = tester.visualize_results(results)
        plt.savefig('reversible_qwen_performance_comparison_fixed.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save results
        tester.save_results(results, 'reversible_qwen_results_fixed.json')
        
        print("\n📊 Visualization saved as 'reversible_qwen_performance_comparison_fixed.png'")
        print("💾 Results saved as 'reversible_qwen_results_fixed.json'")
        print("\n✅ FIXED Performance testing completed!")
        
        # Success check
        reversible_models = [name for name in results.keys() if 'reversible' in name.lower()]
        if reversible_models:
            best_reversible_ppl = min([results[name]['metrics'].perplexity for name in reversible_models])
            if best_reversible_ppl < 100:  # Much better than before
                print("🎉 SUCCESS: Reversible models now showing proper convergence!")
            else:
                print("⚠️  Still some issues - may need further tuning")
    else:
        print("❌ No results generated - check for errors above")