"""
Model Utilities for Robust Handling
===================================

This module provides utility functions to handle common issues with PyTorch models
including device detection and output format variations.
"""

import torch
import torch.nn as nn
from typing import Union, Dict, Any, Tuple

def get_model_device(model: nn.Module) -> torch.device:
    """
    Safely get the device of a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        torch.device: The device the model is on
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        # Model has no parameters, return default device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_logits(output: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
    """
    Extract logits from model output, handling both tensor and dict formats.
    
    Args:
        output: Model output (tensor or dict)
        
    Returns:
        torch.Tensor: The logits tensor
    """
    if isinstance(output, dict):
        # Try common keys for logits
        if 'logits' in output:
            return output['logits']
        elif 'last_hidden_state' in output:
            return output['last_hidden_state']
        elif 'prediction_logits' in output:
            return output['prediction_logits']
        else:
            # Get the first tensor value from the dict
            first_key = next(iter(output.keys()))
            return output[first_key]
    else:
        # Already a tensor
        return output

def safe_model_forward(model: nn.Module, input_ids: torch.Tensor, 
                      extract_logits_only: bool = True) -> torch.Tensor:
    """
    Perform a safe forward pass that handles device and output format issues.
    
    Args:
        model: PyTorch model
        input_ids: Input tensor
        extract_logits_only: Whether to extract logits from dict output
        
    Returns:
        torch.Tensor: Model output (logits if extract_logits_only=True)
    """
    # Ensure input is on the same device as model
    model_device = get_model_device(model)
    input_ids = input_ids.to(model_device)
    
    # Forward pass
    output = model(input_ids)
    
    # Extract logits if needed
    if extract_logits_only:
        return extract_logits(output)
    else:
        return output

def safe_loss_calculation(model: nn.Module, input_ids: torch.Tensor, 
                         labels: torch.Tensor, loss_fn=None) -> torch.Tensor:
    """
    Calculate loss safely, handling device and output format issues.
    
    Args:
        model: PyTorch model
        input_ids: Input tensor
        labels: Target labels
        loss_fn: Loss function (default: CrossEntropyLoss)
        
    Returns:
        torch.Tensor: Calculated loss
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    # Get model device and move tensors
    model_device = get_model_device(model)
    input_ids = input_ids.to(model_device)
    labels = labels.to(model_device)
    
    # Forward pass and extract logits
    output = model(input_ids)
    logits = extract_logits(output)
    
    # Calculate loss
    return loss_fn(logits, labels)

def get_output_shape_info(output: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Tuple[torch.Size, str]:
    """
    Get shape information from model output.
    
    Args:
        output: Model output (tensor or dict)
        
    Returns:
        Tuple[torch.Size, str]: (shape, description)
    """
    if isinstance(output, dict):
        if 'logits' in output:
            return output['logits'].shape, 'logits'
        elif 'last_hidden_state' in output:
            return output['last_hidden_state'].shape, 'last_hidden_state'
        else:
            first_key = next(iter(output.keys()))
            return output[first_key].shape, first_key
    else:
        return output.shape, 'tensor'

class RobustModelWrapper(nn.Module):
    """
    A wrapper that makes any model robust to device and output format issues.
    
    This wrapper provides:
    - Safe device property access
    - Automatic logits extraction
    - Consistent tensor output format
    """
    
    def __init__(self, model: nn.Module, auto_extract_logits: bool = True):
        super().__init__()
        self.model = model
        self.auto_extract_logits = auto_extract_logits
    
    @property 
    def device(self) -> torch.device:
        """Safe device property"""
        return get_model_device(self.model)
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Safe forward with automatic output extraction"""
        # Ensure input is on correct device
        input_ids = input_ids.to(self.device)
        
        # Forward pass
        output = self.model(input_ids, **kwargs)
        
        # Extract logits if requested
        if self.auto_extract_logits:
            return extract_logits(output)
        else:
            return output
    
    def safe_forward(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Forward pass that returns detailed information"""
        input_ids = input_ids.to(self.device)
        output = self.model(input_ids, **kwargs)
        
        shape, description = get_output_shape_info(output)
        
        return {
            'output': output,
            'logits': extract_logits(output),
            'shape': shape,
            'format': description,
            'device': str(self.device)
        }

def wrap_models(models: Dict[str, nn.Module]) -> Dict[str, RobustModelWrapper]:
    """
    Wrap a dictionary of models with RobustModelWrapper.
    
    Args:
        models: Dictionary of model_name -> model
        
    Returns:
        Dict[str, RobustModelWrapper]: Dictionary of wrapped models
    """
    return {name: RobustModelWrapper(model) for name, model in models.items()}

def test_model_compatibility(model: nn.Module, vocab_size: int = 1000, seq_len: int = 64) -> bool:
    """
    Test if a model is compatible with the benchmark framework.
    
    Args:
        model: Model to test
        vocab_size: Vocabulary size for test input
        seq_len: Sequence length for test input
        
    Returns:
        bool: True if compatible
    """
    try:
        # Test device detection
        device = get_model_device(model)
        print(f"✓ Device detection: {device}")
        
        # Test forward pass
        test_input = torch.randint(0, vocab_size, (2, seq_len)).to(device)
        
        with torch.no_grad():
            output = model(test_input)
        
        # Test output extraction
        logits = extract_logits(output)
        shape, description = get_output_shape_info(output)
        
        print(f"✓ Forward pass: {shape} ({description})")
        
        # Test backward pass
        if hasattr(model, 'train'):
            model.train()
            output = model(test_input)
            logits = extract_logits(output)
            loss = logits.sum()
            loss.backward()
            print("✓ Backward pass successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("Model Utilities - Robust Handling for PyTorch Models")
    print("="*60)
    print("Available functions:")
    print("- get_model_device(model): Safe device detection")
    print("- extract_logits(output): Extract logits from any output format")
    print("- safe_model_forward(model, input): Robust forward pass")
    print("- safe_loss_calculation(model, input, labels): Robust loss calculation")
    print("- RobustModelWrapper(model): Make any model robust")
    print("- test_model_compatibility(model): Test model compatibility")
    print("\nImport this module to use these utilities in your benchmarks!")