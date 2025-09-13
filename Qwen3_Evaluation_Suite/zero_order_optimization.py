"""
Zero-Order Optimization for Qwen3 Models (DiZO-style)
=====================================================

This module implements zero-order optimization techniques similar to DiZO (DiZO/MeZO)
for training reversible and standard Qwen3 models with memory-efficient gradient estimation.

Key Features:
- Memory-efficient zero-order gradient estimation
- Forward-mode differentiation compatibility
- Reversible model optimization
- Statistical gradient estimation with variance reduction
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import random
from torch.utils.data import DataLoader
import time

logger = logging.getLogger(__name__)

@dataclass
class ZeroOrderConfig:
    """Configuration for zero-order optimization"""
    
    # Core ZO parameters
    zo_eps: float = 1e-3              # Perturbation magnitude
    zo_method: str = "forward"        # "forward" or "central" difference
    
    # Enhanced ZO parameters (DiZO-style)
    enhanced: bool = True             # Use enhanced ZO with variance reduction
    bits: int = 11                    # Bit precision for quantization
    rng_seed: int = 42               # Random seed for reproducibility
    pre_generated: bool = False       # Pre-generate perturbations
    handling: str = "standard"        # "standard" or "enhanced" handling
    variance_reduction: bool = True   # Apply variance reduction techniques
    
    # Memory optimization
    memory_efficient: bool = True     # Use memory-efficient implementation
    chunk_size: int = 1              # Process in chunks to save memory
    
    # Statistical parameters
    num_perturbations: int = 1        # Number of perturbations per step
    moving_average_decay: float = 0.9 # For gradient estimation smoothing
    gradient_clipping: float = 1.0    # Gradient clipping threshold

class ZeroOrderOptimizer:
    """
    Zero-order optimizer with DiZO-style enhancements for Qwen3 models
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: ZeroOrderConfig,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 0.01):
        
        self.model = model
        self.config = config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Get trainable parameters
        self.trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.param_shapes = [p.shape for p in self.trainable_params]
        self.param_sizes = [p.numel() for p in self.trainable_params]
        self.total_params = sum(self.param_sizes)
        
        logger.info(f"ZO Optimizer initialized with {self.total_params} parameters")
        logger.info(f"Config: eps={config.zo_eps}, method={config.zo_method}, enhanced={config.enhanced}")
        
        # Initialize random state
        self.rng = torch.Generator()
        self.rng.manual_seed(config.rng_seed)
        
        # Moving average for gradient estimation
        if config.enhanced:
            self.gradient_ma = None
            self.gradient_ma_decay = config.moving_average_decay
        
        # Pre-generated perturbations for efficiency
        if config.pre_generated:
            self.perturbation_bank = self._generate_perturbation_bank(1000)
            self.perturbation_idx = 0
        
        # Statistics tracking
        self.step_count = 0
        self.gradient_norms = []
        self.loss_history = []
    
    def _generate_perturbation_bank(self, num_perturbations: int) -> List[List[torch.Tensor]]:
        """Pre-generate perturbations for efficiency"""
        
        bank = []
        for _ in range(num_perturbations):
            perturbation = []
            for shape in self.param_shapes:
                if self.config.enhanced:
                    # Use Rademacher distribution for better properties
                    z = torch.randint(0, 2, shape, generator=self.rng) * 2 - 1
                else:
                    # Standard Gaussian perturbation
                    z = torch.randn(shape, generator=self.rng)
                perturbation.append(z.to(self.trainable_params[0].device))
            bank.append(perturbation)
        
        return bank
    
    def _get_perturbation(self) -> List[torch.Tensor]:
        """Get perturbation vector"""
        
        if self.config.pre_generated:
            # Use pre-generated perturbation
            perturbation = self.perturbation_bank[self.perturbation_idx % len(self.perturbation_bank)]
            self.perturbation_idx += 1
            return perturbation
        else:
            # Generate fresh perturbation
            perturbation = []
            for shape in self.param_shapes:
                if self.config.enhanced:
                    # Rademacher distribution
                    z = torch.randint(0, 2, shape, generator=self.rng) * 2 - 1
                else:
                    # Gaussian distribution
                    z = torch.randn(shape, generator=self.rng)
                perturbation.append(z.to(self.trainable_params[0].device))
            return perturbation
    
    def _apply_perturbation(self, perturbation: List[torch.Tensor], scale: float = 1.0):
        """Apply perturbation to model parameters"""
        
        for param, z in zip(self.trainable_params, perturbation):
            param.data.add_(z, alpha=scale * self.config.zo_eps)
    
    def _remove_perturbation(self, perturbation: List[torch.Tensor], scale: float = 1.0):
        """Remove perturbation from model parameters"""
        
        for param, z in zip(self.trainable_params, perturbation):
            param.data.sub_(z, alpha=scale * self.config.zo_eps)
    
    def _compute_loss(self, batch: Dict, loss_fn: Callable) -> torch.Tensor:
        """Compute loss for given batch"""
        
        # Move batch to correct device
        device = next(self.model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            outputs = self.model(**batch)
            loss = loss_fn(outputs, batch)
        
        return loss
    
    def _estimate_gradient_forward(self, 
                                 batch: Dict, 
                                 loss_fn: Callable,
                                 perturbation: List[torch.Tensor]) -> Tuple[float, List[torch.Tensor]]:
        """Estimate gradient using forward differences"""
        
        # Original loss
        loss_orig = self._compute_loss(batch, loss_fn).item()
        
        # Perturbed loss
        self._apply_perturbation(perturbation)
        loss_pert = self._compute_loss(batch, loss_fn).item()
        self._remove_perturbation(perturbation)
        
        # Compute gradient estimate
        grad_estimate = (loss_pert - loss_orig) / self.config.zo_eps
        
        # Scale perturbation by gradient estimate
        gradient = []
        for z in perturbation:
            gradient.append(grad_estimate * z)
        
        return loss_orig, gradient
    
    def _estimate_gradient_central(self, 
                                 batch: Dict, 
                                 loss_fn: Callable,
                                 perturbation: List[torch.Tensor]) -> Tuple[float, List[torch.Tensor]]:
        """Estimate gradient using central differences (more accurate but 2x cost)"""
        
        # Forward perturbation
        self._apply_perturbation(perturbation)
        loss_forward = self._compute_loss(batch, loss_fn).item()
        self._remove_perturbation(perturbation)
        
        # Backward perturbation
        self._apply_perturbation(perturbation, scale=-1.0)
        loss_backward = self._compute_loss(batch, loss_fn).item()
        self._remove_perturbation(perturbation, scale=-1.0)
        
        # Original loss (for tracking)
        loss_orig = self._compute_loss(batch, loss_fn).item()
        
        # Compute gradient estimate
        grad_estimate = (loss_forward - loss_backward) / (2 * self.config.zo_eps)
        
        # Scale perturbation by gradient estimate
        gradient = []
        for z in perturbation:
            gradient.append(grad_estimate * z)
        
        return loss_orig, gradient
    
    def _apply_variance_reduction(self, gradient: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply variance reduction techniques"""
        
        if not self.config.variance_reduction:
            return gradient
        
        reduced_gradient = []
        
        for g in gradient:
            # Simple variance reduction: moving average
            if self.gradient_ma is None:
                self.gradient_ma = [torch.zeros_like(g) for g in gradient]
            
        # Update moving average
        for i, g in enumerate(gradient):
            if i < len(self.gradient_ma):
                self.gradient_ma[i] = (self.gradient_ma_decay * self.gradient_ma[i] + 
                                     (1 - self.gradient_ma_decay) * g)
                reduced_gradient.append(self.gradient_ma[i])
            else:
                reduced_gradient.append(g)
        
        return reduced_gradient
    
    def step(self, batch: Dict, loss_fn: Callable) -> Dict[str, float]:
        """Perform one optimization step"""
        
        self.step_count += 1
        
        # Generate perturbation
        perturbation = self._get_perturbation()
        
        # Estimate gradient
        if self.config.zo_method == "forward":
            loss, gradient = self._estimate_gradient_forward(batch, loss_fn, perturbation)
        elif self.config.zo_method == "central":
            loss, gradient = self._estimate_gradient_central(batch, loss_fn, perturbation)
        else:
            raise ValueError(f"Unknown ZO method: {self.config.zo_method}")
        
        # Apply variance reduction
        if self.config.enhanced:
            gradient = self._apply_variance_reduction(gradient)
        
        # Compute gradient norm
        grad_norm = torch.sqrt(sum(torch.sum(g**2) for g in gradient)).item()
        
        # Apply gradient clipping
        if grad_norm > self.config.gradient_clipping:
            clip_factor = self.config.gradient_clipping / grad_norm
            gradient = [g * clip_factor for g in gradient]
            grad_norm = self.config.gradient_clipping
        
        # Update parameters
        with torch.no_grad():
            for param, grad in zip(self.trainable_params, gradient):
                # Apply weight decay
                if self.weight_decay > 0:
                    param.data.mul_(1 - self.learning_rate * self.weight_decay)
                
                # Apply gradient update
                param.data.sub_(grad, alpha=self.learning_rate)
        
        # Track statistics
        self.gradient_norms.append(grad_norm)
        self.loss_history.append(loss)
        
        return {
            'loss': loss,
            'grad_norm': grad_norm,
            'step': self.step_count,
            'lr': self.learning_rate,
        }
    
    def get_statistics(self) -> Dict:
        """Get optimization statistics"""
        
        return {
            'total_steps': self.step_count,
            'avg_grad_norm': np.mean(self.gradient_norms) if self.gradient_norms else 0,
            'final_loss': self.loss_history[-1] if self.loss_history else 0,
            'loss_reduction': (self.loss_history[0] - self.loss_history[-1]) if len(self.loss_history) > 1 else 0,
            'convergence_rate': self._compute_convergence_rate(),
        }
    
    def _compute_convergence_rate(self) -> float:
        """Compute convergence rate from loss history"""
        
        if len(self.loss_history) < 10:
            return 0.0
        
        # Fit exponential decay to loss curve
        recent_losses = self.loss_history[-50:]  # Last 50 steps
        x = np.arange(len(recent_losses))
        
        try:
            # Simple linear fit to log loss
            log_losses = np.log(np.maximum(recent_losses, 1e-10))
            if len(set(log_losses)) > 1:  # Check for variation
                slope = np.polyfit(x, log_losses, 1)[0]
                return -slope  # Negative slope means decreasing loss
        except:
            pass
        
        return 0.0

class ZeroOrderTrainer:
    """
    Trainer class for zero-order optimization with DiZO-style enhancements
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_dataloader: DataLoader,
                 eval_dataloader: Optional[DataLoader] = None,
                 zo_config: Optional[ZeroOrderConfig] = None,
                 learning_rate: float = 1e-3,
                 num_epochs: int = 3,
                 eval_steps: int = 100,
                 logging_steps: int = 50):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.num_epochs = num_epochs
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        
        # Initialize ZO optimizer
        if zo_config is None:
            zo_config = ZeroOrderConfig()
        
        self.optimizer = ZeroOrderOptimizer(model, zo_config, learning_rate)
        
        # Training state
        self.global_step = 0
        self.training_stats = []
        self.eval_stats = []
    
    def compute_loss(self, outputs, batch):
        """Compute loss from model outputs and batch"""
        
        if hasattr(outputs, 'loss'):
            return outputs.loss
        elif hasattr(outputs, 'logits'):
            # Standard classification loss
            logits = outputs.logits
            labels = batch.get('labels', batch.get('label'))
            if labels is not None:
                if logits.dim() > 2:
                    logits = logits.view(-1, logits.size(-1))
                    labels = labels.view(-1)
                return torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)
        
        raise ValueError("Cannot compute loss from outputs")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set"""
        
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0
        total_samples = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                # Move batch to device
                device = next(self.model.parameters()).device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = self.compute_loss(outputs, batch)
                
                total_loss += loss.item() * batch['input_ids'].size(0)
                total_samples += batch['input_ids'].size(0)
                
                # Compute accuracy if possible
                if hasattr(outputs, 'logits') and 'labels' in batch:
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    if predictions.dim() > 1:
                        predictions = predictions.view(-1)
                        labels = batch['labels'].view(-1)
                        mask = labels != -100
                        correct_predictions += (predictions[mask] == labels[mask]).sum().item()
                    else:
                        correct_predictions += (predictions == batch['labels']).sum().item()
        
        self.model.train()
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        return {
            'eval_loss': avg_loss,
            'eval_accuracy': accuracy,
            'eval_samples': total_samples,
        }
    
    def train(self) -> Dict:
        """Train model using zero-order optimization"""
        
        logger.info("Starting zero-order training")
        logger.info(f"Total epochs: {self.num_epochs}")
        logger.info(f"Train batches per epoch: {len(self.train_dataloader)}")
        
        self.model.train()
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Zero-order optimization step
                step_stats = self.optimizer.step(batch, self.compute_loss)
                
                epoch_loss += step_stats['loss']
                num_batches += 1
                self.global_step += 1
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    logger.info(f"Epoch {epoch}, Step {self.global_step}, "
                              f"Loss: {avg_loss:.4f}, Grad Norm: {step_stats['grad_norm']:.4f}")
                
                # Evaluation
                if self.eval_steps > 0 and self.global_step % self.eval_steps == 0:
                    eval_stats = self.evaluate()
                    if eval_stats:
                        logger.info(f"Eval - Loss: {eval_stats['eval_loss']:.4f}, "
                                  f"Accuracy: {eval_stats['eval_accuracy']:.4f}")
                        self.eval_stats.append({
                            'step': self.global_step,
                            **eval_stats
                        })
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s, "
                       f"Avg Loss: {avg_epoch_loss:.4f}")
            
            self.training_stats.append({
                'epoch': epoch,
                'avg_loss': avg_epoch_loss,
                'epoch_time': epoch_time,
            })
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        # Final evaluation
        final_eval = self.evaluate()
        
        # Get optimizer statistics
        optimizer_stats = self.optimizer.get_statistics()
        
        return {
            'training_stats': self.training_stats,
            'eval_stats': self.eval_stats,
            'final_eval': final_eval,
            'optimizer_stats': optimizer_stats,
            'total_training_time': total_time,
        }

def create_zo_trainer_for_qwen3(model: nn.Module,
                               train_dataloader: DataLoader,
                               eval_dataloader: Optional[DataLoader] = None,
                               enhanced_zo: bool = True,
                               **kwargs) -> ZeroOrderTrainer:
    """
    Create a zero-order trainer optimized for Qwen3 models
    """
    
    # ZO configuration optimized for Qwen3
    zo_config = ZeroOrderConfig(
        zo_eps=1e-3,
        zo_method="forward",
        enhanced=enhanced_zo,
        bits=11,
        variance_reduction=True,
        memory_efficient=True,
        num_perturbations=1,
        gradient_clipping=1.0,
        **kwargs
    )
    
    return ZeroOrderTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        zo_config=zo_config,
        **kwargs
    )
