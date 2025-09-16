"""
DiZO Setup Helper for Comprehensive Benchmark
============================================

This script helps set up and validate the DiZO environment for 
fair comparison with Qwen3 models.
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiZOSetupHelper:
    """Helper class to set up and validate DiZO environment"""
    
    def __init__(self, dizo_path: str = "/home/yul23028/DiZO/large_models"):
        self.dizo_path = Path(dizo_path)
        self.required_files = [
            "run.py",
            "dizo.sh", 
            "tasks.py",
            "trainer.py",
            "utils.py",
            "metrics.py",
            "dataset.py"
        ]
        
    def validate_dizo_installation(self) -> bool:
        """Validate that DiZO is properly installed and accessible"""
        logger.info("Validating DiZO installation...")
        
        if not self.dizo_path.exists():
            logger.error(f"DiZO path not found: {self.dizo_path}")
            return False
        
        missing_files = []
        for required_file in self.required_files:
            file_path = self.dizo_path / required_file
            if not file_path.exists():
                missing_files.append(required_file)
        
        if missing_files:
            logger.error(f"Missing DiZO files: {missing_files}")
            return False
        
        logger.info("✓ DiZO installation validated")
        return True
    
    def test_dizo_import(self) -> bool:
        """Test importing DiZO modules"""
        logger.info("Testing DiZO module imports...")
        
        try:
            # Add DiZO path to sys.path temporarily
            if str(self.dizo_path) not in sys.path:
                sys.path.insert(0, str(self.dizo_path))
            
            # Try importing key modules
            import tasks
            import utils
            import metrics
            import trainer
            
            logger.info("✓ DiZO modules imported successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import DiZO modules: {e}")
            return False
    
    def create_dizo_config(self, output_path: str = "dizo_config.json") -> bool:
        """Create configuration file for DiZO benchmark runs"""
        logger.info("Creating DiZO configuration...")
        
        # Default configurations for different scales
        config = {
            "small": {
                "model": "facebook/opt-350m",
                "learning_rate": 1e-6,
                "eps": 1e-3,
                "steps": 2000,
                "eval_steps": 500,
                "batch_size": 16,
                "num_train": 500,
                "num_dev": 200,
                "num_eval": 500,
                "enhanced": "zo"
            },
            "medium": {
                "model": "facebook/opt-1.3b", 
                "learning_rate": 1e-6,
                "eps": 1e-3,
                "steps": 4000,
                "eval_steps": 500,
                "batch_size": 16,
                "num_train": 1000,
                "num_dev": 500,
                "num_eval": 1000,
                "enhanced": "zo"
            },
            "large": {
                "model": "facebook/opt-2.7b",
                "learning_rate": 1e-6,
                "eps": 1e-3,
                "steps": 8000,
                "eval_steps": 500,
                "batch_size": 16,
                "num_train": 2000,
                "num_dev": 1000,
                "num_eval": 2000,
                "enhanced": "zo"
            }
        }
        
        # Task-specific configurations
        task_configs = {
            "SST2": {"train_as_classification": True},
            "COLA": {"train_as_classification": True},
            "MRPC": {"train_as_classification": True},
            "QQP": {"train_as_classification": True},
            "MNLI": {"train_as_classification": True},
            "QNLI": {"train_as_classification": True},
            "RTE": {"train_as_classification": True},
            "WNLI": {"train_as_classification": True},
            "Copa": {"train_as_classification": False, "num_dev": 100},
            "ReCoRD": {"train_as_classification": False},
            "DROP": {"train_as_classification": False},
            "SQuAD": {"train_as_classification": False},
            "CB": {"num_dev": 100}
        }
        
        full_config = {
            "scale_configs": config,
            "task_configs": task_configs,
            "dizo_path": str(self.dizo_path)
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(full_config, f, indent=2)
            logger.info(f"✓ DiZO configuration saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save DiZO config: {e}")
            return False
    
    def run_dizo_test(self, task: str = "SST2", scale: str = "small") -> bool:
        """Run a quick DiZO test to ensure everything works"""
        logger.info(f"Running DiZO test with task={task}, scale={scale}")
        
        if not self.validate_dizo_installation():
            return False
        
        # Load configuration
        try:
            with open("dizo_config.json", 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            logger.warning("DiZO config not found, creating default...")
            if not self.create_dizo_config():
                return False
            with open("dizo_config.json", 'r') as f:
                config = json.load(f)
        
        scale_config = config["scale_configs"][scale]
        task_config = config["task_configs"].get(task, {})
        
        # Set environment variables
        env = os.environ.copy()
        env.update({
            'MODEL': scale_config['model'],
            'TASK': task,
            'MODE': 'ft',
            'LR': str(scale_config['learning_rate']),
            'EPS': str(scale_config['eps']),
            'STEPS': '100',  # Short test run
            'EVAL_STEPS': '50',
            'BS': str(scale_config['batch_size']),
            'TRAIN': '50',  # Small dataset for test
            'DEV': '20',
            'EVAL': '20',
            'ENHANCED': scale_config['enhanced'],
            'SEED': '26'
        })
        
        # Add task-specific arguments
        extra_args = []
        if not task_config.get('train_as_classification', True):
            extra_args.append('--train_as_classification False')
        
        try:
            # Change to DiZO directory
            original_cwd = os.getcwd()
            os.chdir(self.dizo_path)
            
            # Run test
            result = subprocess.run(
                ['bash', 'dizo.sh'] + extra_args,
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for test
            )
            
            os.chdir(original_cwd)
            
            if result.returncode == 0:
                logger.info("✓ DiZO test completed successfully")
                return True
            else:
                logger.error(f"DiZO test failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("DiZO test timed out")
            return False
        except Exception as e:
            logger.error(f"DiZO test exception: {e}")
            return False
    
    def check_model_availability(self, models: List[str]) -> Dict[str, bool]:
        """Check if specified OPT models are available"""
        logger.info("Checking OPT model availability...")
        
        availability = {}
        
        for model in models:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model)
                availability[model] = True
                logger.info(f"✓ {model} available")
            except Exception as e:
                availability[model] = False
                logger.warning(f"✗ {model} not available: {e}")
        
        return availability
    
    def setup_comprehensive_benchmark(self) -> bool:
        """Set up everything needed for comprehensive benchmark"""
        logger.info("Setting up comprehensive benchmark environment...")
        
        success = True
        
        # Validate DiZO installation
        if not self.validate_dizo_installation():
            success = False
        
        # Test DiZO imports
        if not self.test_dizo_import():
            success = False
        
        # Create configuration
        if not self.create_dizo_config():
            success = False
        
        # Check model availability
        models_to_check = [
            "facebook/opt-350m",
            "facebook/opt-1.3b", 
            "facebook/opt-2.7b"
        ]
        availability = self.check_model_availability(models_to_check)
        if not any(availability.values()):
            logger.error("No OPT models available")
            success = False
        
        # Run quick test
        if success:
            logger.info("Running quick DiZO test...")
            if not self.run_dizo_test():
                logger.warning("DiZO test failed, but setup may still work")
        
        if success:
            logger.info("✓ Comprehensive benchmark setup completed successfully")
        else:
            logger.error("✗ Benchmark setup failed")
        
        return success

def create_run_script(output_path: str = "run_comprehensive_benchmark.sh"):
    """Create a convenient run script for the benchmark"""
    script_content = '''#!/bin/bash

# Comprehensive DiZO OPT vs Qwen3 Benchmark Runner
# Usage: ./run_comprehensive_benchmark.sh [scale] [tasks] [models]

SCALE=${1:-small}
TASKS=${2:-sst2,cola}
MODELS=${3:-all}

echo "Running comprehensive benchmark..."
echo "Scale: $SCALE"
echo "Tasks: $TASKS" 
echo "Models: $MODELS"

python comprehensive_dizo_qwen3_benchmark.py \\
    --scale $SCALE \\
    --tasks $TASKS \\
    --models $MODELS \\
    --full_eval \\
    --output_dir ./benchmark_results_$(date +%Y%m%d_%H%M%S)

echo "Benchmark completed!"
'''
    
    try:
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(output_path, 0o755)
        
        logger.info(f"✓ Run script created: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create run script: {e}")
        return False

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DiZO Setup Helper")
    parser.add_argument('--dizo_path', type=str, default="/home/yul23028/DiZO/large_models",
                        help='Path to DiZO installation')
    parser.add_argument('--test', action='store_true',
                        help='Run DiZO test after setup')
    parser.add_argument('--create_run_script', action='store_true',
                        help='Create convenient run script')
    
    args = parser.parse_args()
    
    # Initialize helper
    helper = DiZOSetupHelper(args.dizo_path)
    
    # Run setup
    success = helper.setup_comprehensive_benchmark()
    
    # Run test if requested
    if args.test and success:
        logger.info("Running extended test...")
        helper.run_dizo_test()
    
    # Create run script if requested
    if args.create_run_script:
        create_run_script()
    
    if success:
        print("\n" + "="*60)
        print("DiZO SETUP COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nYou can now run the comprehensive benchmark:")
        print("python comprehensive_dizo_qwen3_benchmark.py --scale small --tasks sst2,cola --models all")
        print("\nOr use the convenience script (if created):")
        print("./run_comprehensive_benchmark.sh small sst2,cola all")
        print("\n" + "="*60)
    else:
        print("\n" + "="*60)
        print("SETUP FAILED - Please check the errors above")
        print("="*60)

if __name__ == "__main__":
    main()