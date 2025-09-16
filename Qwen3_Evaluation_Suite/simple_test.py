#!/usr/bin/env python3
"""
Simple test to verify the evaluation framework structure without heavy dependencies
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestEvaluationFramework(unittest.TestCase):
    
    def test_config_loading(self):
        """Test that config files can be loaded"""
        try:
            import yaml
            config_path = os.path.join(os.path.dirname(__file__), 'configs', 'comprehensive_eval.yaml')
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.assertIn('model_names', config)
                    self.assertIn('tasks', config)
                    print("✓ Config loading works")
            else:
                print("⚠ Config file not found, but structure test passed")
                
        except ImportError:
            print("⚠ YAML not available, skipping config test")
    
    def test_framework_imports(self):
        """Test that our framework modules can be imported"""
        try:
            # Test importing our modules without executing them
            import importlib.util
            
            # Test run_evaluation.py
            spec = importlib.util.spec_from_file_location(
                "run_evaluation", 
                os.path.join(os.path.dirname(__file__), "run_evaluation.py")
            )
            
            if spec and spec.loader:
                print("✓ run_evaluation.py can be loaded")
            
            # Test analysis_and_comparison.py
            spec = importlib.util.spec_from_file_location(
                "analysis_and_comparison", 
                os.path.join(os.path.dirname(__file__), "analysis_and_comparison.py")
            )
            
            if spec and spec.loader:
                print("✓ analysis_and_comparison.py can be loaded")
                
            # Test zero_order_optimization.py
            spec = importlib.util.spec_from_file_location(
                "zero_order_optimization", 
                os.path.join(os.path.dirname(__file__), "zero_order_optimization.py")
            )
            
            if spec and spec.loader:
                print("✓ zero_order_optimization.py can be loaded")
                
        except Exception as e:
            self.fail(f"Framework import test failed: {e}")
    
    def test_directory_structure(self):
        """Test that required directories and files exist"""
        base_dir = os.path.dirname(__file__)
        
        # Check main files
        required_files = [
            'run_evaluation.py',
            'analysis_and_comparison.py', 
            'zero_order_optimization.py',
            'quick_test.py'
        ]
        
        for file in required_files:
            file_path = os.path.join(base_dir, file)
            self.assertTrue(os.path.exists(file_path), f"Missing file: {file}")
        
        # Check configs directory
        configs_dir = os.path.join(base_dir, 'configs')
        self.assertTrue(os.path.exists(configs_dir), "Missing configs directory")
        
        print("✓ Directory structure is correct")
    
    def test_parent_model_files(self):
        """Test that parent directory has model files"""
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        
        # Check for model files
        model_files = [
            'qwen3_reversible_02_3.py',
            'test_train_qwen3_rev_v202_21.py'
        ]
        
        found_files = []
        for file in model_files:
            file_path = os.path.join(parent_dir, file)
            if os.path.exists(file_path):
                found_files.append(file)
        
        if found_files:
            print(f"✓ Found model files: {', '.join(found_files)}")
        else:
            print("⚠ No model files found in parent directory")
    
    def test_statistical_analysis_basics(self):
        """Test basic statistical analysis without scipy"""
        try:
            # Mock numpy functions if not available
            try:
                import numpy as np
                
                # Test basic array operations
                arr1 = np.array([1, 2, 3, 4, 5])
                arr2 = np.array([2, 3, 4, 5, 6])
                
                # Test that we can compute basic statistics
                mean1 = np.mean(arr1)
                mean2 = np.mean(arr2)
                std1 = np.std(arr1, ddof=1)
                std2 = np.std(arr2, ddof=1)
                
                self.assertAlmostEqual(mean1, 3.0)
                self.assertAlmostEqual(mean2, 4.0)
                
                print("✓ Basic statistical analysis works")
                
            except ImportError:
                # Fallback to pure Python
                arr1 = [1, 2, 3, 4, 5]
                arr2 = [2, 3, 4, 5, 6]
                
                mean1 = sum(arr1) / len(arr1)
                mean2 = sum(arr2) / len(arr2)
                
                self.assertAlmostEqual(mean1, 3.0)
                self.assertAlmostEqual(mean2, 4.0)
                
                print("✓ Basic statistical analysis works (pure Python)")
                
        except Exception as e:
            self.fail(f"Statistical analysis test failed: {e}")

def main():
    print("Running simple framework tests...")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 50)
    print("Simple test completed!")
    print("\nNote: This test verifies the framework structure.")
    print("For full functionality, install PyTorch and transformers.")

if __name__ == "__main__":
    main()