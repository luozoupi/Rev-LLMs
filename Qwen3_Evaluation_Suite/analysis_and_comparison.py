"""
Comprehensive Analysis and Comparison Module for Qwen3 Evaluation
================================================================

This module provides detailed analysis and comparison capabilities for 
reversible vs standard Qwen3 models, similar to DiZO's evaluation framework.

Features:
- Statistical significance testing
- Performance gap analysis
- Memory efficiency comparison
- Training dynamics analysis
- Visualization and reporting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """Statistical analysis for model comparison"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def compare_performance(self, 
                          reversible_scores: List[float], 
                          standard_scores: List[float],
                          metric_name: str = "accuracy") -> Dict:
        """
        Compare performance between reversible and standard models with statistical testing
        """
        
        if not reversible_scores or not standard_scores:
            return {"error": "Empty score lists"}
        
        # Convert to numpy arrays to avoid ambiguous truth value issues
        reversible_scores = np.asarray(reversible_scores)
        standard_scores = np.asarray(standard_scores)
        
        # Basic statistics
        rev_mean = np.mean(reversible_scores)
        rev_std = np.std(reversible_scores, ddof=1)  # Use sample standard deviation
        std_mean = np.mean(standard_scores)
        std_std = np.std(standard_scores, ddof=1)  # Use sample standard deviation
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(reversible_scores) - 1) * rev_std**2 + 
                             (len(standard_scores) - 1) * std_std**2) / 
                            (len(reversible_scores) + len(standard_scores) - 2))
        cohens_d = (rev_mean - std_mean) / pooled_std if pooled_std > 0 else 0
        
        # Statistical tests
        # 1. Welch's t-test (assumes unequal variances)
        t_stat, t_pvalue = stats.ttest_ind(reversible_scores, standard_scores, equal_var=False)
        
        # 2. Mann-Whitney U test (non-parametric)
        u_stat, u_pvalue = stats.mannwhitneyu(reversible_scores, standard_scores, 
                                             alternative='two-sided')
        
        # 3. Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(reversible_scores, standard_scores)
        
        # Confidence intervals
        rev_se = rev_std / np.sqrt(len(reversible_scores))
        std_se = std_std / np.sqrt(len(standard_scores))
        
        rev_ci = stats.t.interval(self.confidence_level, len(reversible_scores)-1, 
                                 loc=rev_mean, scale=rev_se)
        std_ci = stats.t.interval(self.confidence_level, len(standard_scores)-1,
                                 loc=std_mean, scale=std_se)
        
        # Interpretation
        significant = t_pvalue < self.alpha
        effect_size_magnitude = self._interpret_effect_size(abs(cohens_d))
        
        return {
            'metric': metric_name,
            'reversible': {
                'mean': rev_mean,
                'std': rev_std,
                'n': len(reversible_scores),
                'ci_lower': rev_ci[0],
                'ci_upper': rev_ci[1],
            },
            'standard': {
                'mean': std_mean,
                'std': std_std,
                'n': len(standard_scores),
                'ci_lower': std_ci[0],
                'ci_upper': std_ci[1],
            },
            'comparison': {
                'difference': rev_mean - std_mean,
                'relative_difference': (rev_mean - std_mean) / std_mean if std_mean != 0 else 0,
                'cohens_d': cohens_d,
                'effect_size_magnitude': effect_size_magnitude,
            },
            'statistical_tests': {
                'welch_t_test': {
                    'statistic': t_stat,
                    'p_value': t_pvalue,
                    'significant': significant,
                },
                'mann_whitney_u': {
                    'statistic': u_stat,
                    'p_value': u_pvalue,
                    'significant': u_pvalue < self.alpha,
                },
                'kolmogorov_smirnov': {
                    'statistic': ks_stat,
                    'p_value': ks_pvalue,
                    'significant': ks_pvalue < self.alpha,
                }
            }
        }
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"

class MemoryAnalyzer:
    """Analyze memory usage patterns"""
    
    def __init__(self):
        pass
    
    def analyze_memory_scaling(self, memory_profiles: Dict) -> Dict:
        """Analyze how memory usage scales with sequence length"""
        
        scaling_analysis = {}
        
        for model_name, profiles in memory_profiles.items():
            if 'memory_profiles' not in profiles:
                continue
            
            lengths = []
            memory_usage = []
            
            for length_key, memory_info in profiles['memory_profiles'].items():
                if 'error' in memory_info:
                    continue
                
                try:
                    length = int(length_key.split('_')[1])
                    memory_mb = memory_info.get('memory_used_mb', 0)
                    
                    lengths.append(length)
                    memory_usage.append(memory_mb)
                except:
                    continue
            
            if len(lengths) >= 3:
                # Fit power law: memory = a * length^b
                log_lengths = np.log(lengths)
                log_memory = np.log(np.maximum(memory_usage, 1e-6))
                
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(log_lengths, log_memory)
                    
                    scaling_analysis[model_name] = {
                        'scaling_exponent': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'memory_complexity': self._interpret_scaling(slope),
                        'lengths': lengths,
                        'memory_usage_mb': memory_usage,
                    }
                except:
                    scaling_analysis[model_name] = {'error': 'Could not fit scaling law'}
        
        return scaling_analysis
    
    def _interpret_scaling(self, exponent: float) -> str:
        """Interpret memory scaling exponent"""
        if exponent < 1.2:
            return "sub-linear (efficient)"
        elif exponent < 1.8:
            return "linear"
        elif exponent < 2.2:
            return "quadratic"
        else:
            return "super-quadratic (inefficient)"
    
    def compare_memory_efficiency(self, reversible_profiles: Dict, standard_profiles: Dict) -> Dict:
        """Compare memory efficiency between model types"""
        
        comparison = {
            'max_supported_length': {},
            'memory_at_length': {},
            'efficiency_ratio': {},
        }
        
        # Find maximum supported sequence length
        for model_type, profiles in [("reversible", reversible_profiles), ("standard", standard_profiles)]:
            max_length = 0
            for profile in profiles.values():
                if 'memory_profiles' not in profile:
                    continue
                for length_key, memory_info in profile['memory_profiles'].items():
                    if 'error' not in memory_info:
                        try:
                            length = int(length_key.split('_')[1])
                            max_length = max(max_length, length)
                        except:
                            continue
            comparison['max_supported_length'][model_type] = max_length
        
        # Compare memory usage at common lengths
        common_lengths = [512, 1024, 2048]
        for length in common_lengths:
            rev_memory = self._get_average_memory_at_length(reversible_profiles, length)
            std_memory = self._get_average_memory_at_length(standard_profiles, length)
            
            if rev_memory > 0 and std_memory > 0:
                comparison['memory_at_length'][f'length_{length}'] = {
                    'reversible_mb': rev_memory,
                    'standard_mb': std_memory,
                    'ratio': rev_memory / std_memory,
                    'savings_percent': (1 - rev_memory / std_memory) * 100,
                }
        
        return comparison
    
    def _get_average_memory_at_length(self, profiles: Dict, target_length: int) -> float:
        """Get average memory usage at specific length"""
        memory_values = []
        
        for profile in profiles.values():
            if 'memory_profiles' not in profile:
                continue
            
            length_key = f'length_{target_length}'
            if length_key in profile['memory_profiles']:
                memory_info = profile['memory_profiles'][length_key]
                if 'error' not in memory_info:
                    memory_mb = memory_info.get('memory_used_mb', 0)
                    if memory_mb > 0:
                        memory_values.append(memory_mb)
        
        return np.mean(memory_values) if memory_values else 0

class PerformanceAnalyzer:
    """Analyze training and inference performance"""
    
    def __init__(self):
        self.stat_analyzer = StatisticalAnalyzer()
    
    def analyze_training_efficiency(self, results: Dict) -> Dict:
        """Analyze training efficiency across models"""
        
        training_analysis = {
            'convergence_rates': {},
            'training_times': {},
            'efficiency_comparison': {},
        }
        
        reversible_times = []
        standard_times = []
        reversible_convergence = []
        standard_convergence = []
        
        for model_name, result in results.items():
            if 'error' in result:
                continue
            
            # Extract training metrics
            training_time = 0
            convergence_rate = 0
            
            # From task performance
            for task_name, task_result in result.get('task_performance', {}).items():
                if 'training_time' in task_result:
                    training_time += task_result['training_time']
                
                # From optimizer stats if available
                if 'optimizer_stats' in task_result:
                    convergence_rate = max(convergence_rate, 
                                         task_result['optimizer_stats'].get('convergence_rate', 0))
            
            # Categorize by model type
            if 'reversible' in model_name.lower():
                if training_time > 0:
                    reversible_times.append(training_time)
                if convergence_rate > 0:
                    reversible_convergence.append(convergence_rate)
            else:
                if training_time > 0:
                    standard_times.append(training_time)
                if convergence_rate > 0:
                    standard_convergence.append(convergence_rate)
            
            training_analysis['training_times'][model_name] = training_time
            training_analysis['convergence_rates'][model_name] = convergence_rate
        
        # Statistical comparison
        if reversible_times and standard_times:
            time_comparison = self.stat_analyzer.compare_performance(
                reversible_times, standard_times, "training_time"
            )
            training_analysis['efficiency_comparison']['training_time'] = time_comparison
        
        if reversible_convergence and standard_convergence:
            convergence_comparison = self.stat_analyzer.compare_performance(
                reversible_convergence, standard_convergence, "convergence_rate"
            )
            training_analysis['efficiency_comparison']['convergence_rate'] = convergence_comparison
        
        return training_analysis

class VisualizationEngine:
    """Create visualizations for evaluation results"""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def create_performance_comparison_plot(self, comparison_results: Dict, save_path: str = None):
        """Create performance comparison plots"""
        
        if not comparison_results.get('performance_gaps'):
            logger.warning("No performance gaps data available for plotting")
            return
        
        # Extract data
        tasks = []
        rev_scores = []
        std_scores = []
        gaps = []
        
        for task, gap_info in comparison_results['performance_gaps'].items():
            tasks.append(task.upper())
            rev_scores.append(gap_info['reversible_avg'])
            std_scores.append(gap_info['standard_avg'])
            gaps.append(gap_info['gap'])
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance comparison
        x = np.arange(len(tasks))
        width = 0.35
        
        ax1.bar(x - width/2, rev_scores, width, label='Reversible', alpha=0.8)
        ax1.bar(x + width/2, std_scores, width, label='Standard', alpha=0.8)
        
        ax1.set_xlabel('Tasks')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Performance Comparison: Reversible vs Standard')
        ax1.set_xticks(x)
        ax1.set_xticklabels(tasks, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Performance gaps
        colors = ['green' if gap > 0 else 'red' for gap in gaps]
        bars = ax2.bar(tasks, gaps, color=colors, alpha=0.7)
        
        ax2.set_xlabel('Tasks')
        ax2.set_ylabel('Performance Gap (Reversible - Standard)')
        ax2.set_title('Performance Gaps')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, gap in zip(bars, gaps):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{gap:+.3f}', ha='center', va='bottom' if gap > 0 else 'top')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "performance_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance comparison plot saved to {save_path}")
    
    def create_memory_scaling_plot(self, memory_analysis: Dict, save_path: str = None):
        """Create memory scaling visualization"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Memory scaling curves
        for model_name, scaling_info in memory_analysis.items():
            if 'error' in scaling_info:
                continue
            
            lengths = scaling_info['lengths']
            memory_usage = scaling_info['memory_usage_mb']
            exponent = scaling_info['scaling_exponent']
            
            ax1.loglog(lengths, memory_usage, 'o-', label=f'{model_name} (exp={exponent:.2f})', alpha=0.8)
        
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.set_title('Memory Scaling with Sequence Length')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scaling exponents comparison
        model_names = []
        exponents = []
        
        for model_name, scaling_info in memory_analysis.items():
            if 'error' not in scaling_info:
                model_names.append(model_name.replace('_', '\n'))
                exponents.append(scaling_info['scaling_exponent'])
        
        if model_names and exponents:
            colors = ['green' if exp < 1.5 else 'orange' if exp < 2.0 else 'red' for exp in exponents]
            bars = ax2.bar(model_names, exponents, color=colors, alpha=0.7)
            
            ax2.set_ylabel('Scaling Exponent')
            ax2.set_title('Memory Scaling Exponents')
            ax2.axhline(y=1.0, color='blue', linestyle='--', alpha=0.5, label='Linear')
            ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Quadratic')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, exp in zip(bars, exponents):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{exp:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "memory_scaling.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Memory scaling plot saved to {save_path}")
    
    def create_training_dynamics_plot(self, results: Dict, save_path: str = None):
        """Create training dynamics visualization"""
        
        # Extract training curves
        training_curves = {}
        
        for model_name, result in results.items():
            if 'error' in result:
                continue
            
            # Look for training statistics
            for task_name, task_result in result.get('task_performance', {}).items():
                if 'training_stats' in task_result:
                    key = f"{model_name}_{task_name}"
                    training_curves[key] = task_result['training_stats']
                    break  # Use first task for simplicity
        
        if not training_curves:
            logger.warning("No training curves data available for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training loss curves
        for curve_name, stats in training_curves.items():
            if isinstance(stats, list) and len(stats) > 0:
                epochs = [s.get('epoch', i) for i, s in enumerate(stats)]
                losses = [s.get('avg_loss', 0) for s in stats]
                
                ax1.plot(epochs, losses, 'o-', label=curve_name, alpha=0.8)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Training time per epoch
        for curve_name, stats in training_curves.items():
            if isinstance(stats, list) and len(stats) > 0:
                epochs = [s.get('epoch', i) for i, s in enumerate(stats)]
                times = [s.get('epoch_time', 0) for s in stats]
                
                ax2.plot(epochs, times, 's-', label=curve_name, alpha=0.8)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Time per Epoch (s)')
        ax2.set_title('Training Time per Epoch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "training_dynamics.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training dynamics plot saved to {save_path}")

class ComprehensiveReportGenerator:
    """Generate comprehensive evaluation reports"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stat_analyzer = StatisticalAnalyzer()
        self.memory_analyzer = MemoryAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.visualizer = VisualizationEngine(str(self.output_dir / "figures"))
    
    def generate_full_report(self, evaluation_results: Dict) -> str:
        """Generate comprehensive evaluation report"""
        
        logger.info("Generating comprehensive evaluation report")
        
        # Analyze results
        detailed_results = evaluation_results.get('detailed', {})
        
        # Separate reversible and standard results
        reversible_results = {k: v for k, v in detailed_results.items() if 'reversible' in k.lower()}
        standard_results = {k: v for k, v in detailed_results.items() if 'reversible' not in k.lower()}
        
        # Statistical analysis
        statistical_comparisons = self._perform_statistical_analysis(reversible_results, standard_results)
        
        # Memory analysis
        memory_analysis = self.memory_analyzer.analyze_memory_scaling(detailed_results)
        memory_comparison = self.memory_analyzer.compare_memory_efficiency(reversible_results, standard_results)
        
        # Performance analysis
        performance_analysis = self.performance_analyzer.analyze_training_efficiency(detailed_results)
        
        # Create visualizations
        if evaluation_results.get('comparisons'):
            self.visualizer.create_performance_comparison_plot(evaluation_results['comparisons'])
        
        if memory_analysis:
            self.visualizer.create_memory_scaling_plot(memory_analysis)
        
        self.visualizer.create_training_dynamics_plot(detailed_results)
        
        # Generate report
        report_content = self._create_report_content(
            evaluation_results,
            statistical_comparisons,
            memory_analysis,
            memory_comparison,
            performance_analysis
        )
        
        # Save report
        report_path = self.output_dir / "comprehensive_evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Comprehensive report saved to {report_path}")
        return str(report_path)
    
    def _perform_statistical_analysis(self, reversible_results: Dict, standard_results: Dict) -> Dict:
        """Perform statistical analysis comparing model types"""
        
        comparisons = {}
        
        # Extract performance metrics for common tasks
        common_tasks = set()
        for result in reversible_results.values():
            if 'task_performance' in result:
                common_tasks.update(result['task_performance'].keys())
        
        for result in standard_results.values():
            if 'task_performance' in result:
                common_tasks = common_tasks.intersection(result['task_performance'].keys())
        
        # Compare each task
        for task in common_tasks:
            rev_scores = []
            std_scores = []
            
            # Extract scores
            for result in reversible_results.values():
                if task in result.get('task_performance', {}):
                    task_result = result['task_performance'][task]
                    if 'eval_metrics' in task_result:
                        eval_metrics = task_result['eval_metrics']
                        # Try different metric names
                        for metric_key in ['eval_accuracy', 'eval_f1', 'eval_matthews_correlation']:
                            if metric_key in eval_metrics:
                                rev_scores.append(eval_metrics[metric_key])
                                break
            
            for result in standard_results.values():
                if task in result.get('task_performance', {}):
                    task_result = result['task_performance'][task]
                    if 'eval_metrics' in task_result:
                        eval_metrics = task_result['eval_metrics']
                        for metric_key in ['eval_accuracy', 'eval_f1', 'eval_matthews_correlation']:
                            if metric_key in eval_metrics:
                                std_scores.append(eval_metrics[metric_key])
                                break
            
            # Perform comparison
            if rev_scores and std_scores:
                comparison = self.stat_analyzer.compare_performance(rev_scores, std_scores, task)
                comparisons[task] = comparison
        
        return comparisons
    
    def _create_report_content(self, 
                             evaluation_results: Dict,
                             statistical_comparisons: Dict,
                             memory_analysis: Dict,
                             memory_comparison: Dict,
                             performance_analysis: Dict) -> str:
        """Create markdown report content"""
        
        report = []
        
        # Header
        report.append("# Qwen3 Evaluation Report: Reversible vs Standard Models")
        report.append("=" * 60)
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        
        summary = evaluation_results.get('summary', {})
        total_models = summary.get('total_models_evaluated', 0)
        successful = summary.get('successful_evaluations', 0)
        failed = summary.get('failed_evaluations', 0)
        
        report.append(f"- **Total Models Evaluated**: {total_models}")
        report.append(f"- **Successful Evaluations**: {successful}")
        report.append(f"- **Failed Evaluations**: {failed}")
        report.append(f"- **Success Rate**: {successful/total_models*100:.1f}%" if total_models > 0 else "- **Success Rate**: N/A")
        report.append("")
        
        # Performance Comparison
        report.append("## Performance Comparison")
        report.append("")
        
        if statistical_comparisons:
            report.append("### Statistical Analysis Results")
            report.append("")
            report.append("| Task | Reversible | Standard | Gap | Effect Size | Significant |")
            report.append("|------|------------|----------|-----|-------------|-------------|")
            
            for task, comparison in statistical_comparisons.items():
                rev_mean = comparison['reversible']['mean']
                std_mean = comparison['standard']['mean']
                gap = comparison['comparison']['difference']
                effect_size = comparison['comparison']['cohens_d']
                significant = comparison['statistical_tests']['welch_t_test']['significant']
                
                report.append(f"| {task.upper()} | {rev_mean:.3f} | {std_mean:.3f} | {gap:+.3f} | {effect_size:.3f} | {'✓' if significant else '✗'} |")
            
            report.append("")
        
        # Memory Analysis
        report.append("## Memory Analysis")
        report.append("")
        
        if memory_analysis:
            report.append("### Memory Scaling Analysis")
            report.append("")
            
            for model_name, scaling_info in memory_analysis.items():
                if 'error' not in scaling_info:
                    exponent = scaling_info['scaling_exponent']
                    complexity = scaling_info['memory_complexity']
                    r_squared = scaling_info['r_squared']
                    
                    report.append(f"- **{model_name}**: Scaling exponent = {exponent:.2f} ({complexity}), R² = {r_squared:.3f}")
            
            report.append("")
        
        if memory_comparison:
            report.append("### Memory Efficiency Comparison")
            report.append("")
            
            max_lengths = memory_comparison.get('max_supported_length', {})
            if max_lengths:
                rev_max = max_lengths.get('reversible', 0)
                std_max = max_lengths.get('standard', 0)
                report.append(f"- **Maximum Sequence Length**:")
                report.append(f"  - Reversible: {rev_max}")
                report.append(f"  - Standard: {std_max}")
                report.append("")
            
            memory_at_length = memory_comparison.get('memory_at_length', {})
            if memory_at_length:
                report.append("- **Memory Usage Comparison**:")
                for length_key, memory_info in memory_at_length.items():
                    length = length_key.split('_')[1]
                    rev_mb = memory_info['reversible_mb']
                    std_mb = memory_info['standard_mb']
                    savings = memory_info['savings_percent']
                    
                    report.append(f"  - Length {length}: {rev_mb:.1f} MB vs {std_mb:.1f} MB ({savings:+.1f}% savings)")
            
            report.append("")
        
        # Training Efficiency
        report.append("## Training Efficiency")
        report.append("")
        
        if performance_analysis:
            efficiency_comparison = performance_analysis.get('efficiency_comparison', {})
            
            if 'training_time' in efficiency_comparison:
                time_comp = efficiency_comparison['training_time']
                rev_time = time_comp['reversible']['mean']
                std_time = time_comp['standard']['mean']
                time_diff = time_comp['comparison']['relative_difference']
                
                report.append(f"- **Average Training Time**:")
                report.append(f"  - Reversible: {rev_time:.1f}s")
                report.append(f"  - Standard: {std_time:.1f}s")
                report.append(f"  - Relative Difference: {time_diff:+.1%}")
                report.append("")
            
            if 'convergence_rate' in efficiency_comparison:
                conv_comp = efficiency_comparison['convergence_rate']
                rev_conv = conv_comp['reversible']['mean']
                std_conv = conv_comp['standard']['mean']
                conv_diff = conv_comp['comparison']['relative_difference']
                
                report.append(f"- **Convergence Rate**:")
                report.append(f"  - Reversible: {rev_conv:.4f}")
                report.append(f"  - Standard: {std_conv:.4f}")
                report.append(f"  - Relative Difference: {conv_diff:+.1%}")
                report.append("")
        
        # Detailed Results
        report.append("## Detailed Results")
        report.append("")
        
        detailed_results = evaluation_results.get('detailed', {})
        
        for model_name, result in detailed_results.items():
            report.append(f"### {model_name}")
            report.append("")
            
            if 'error' in result:
                report.append(f"**Error**: {result['error']}")
                report.append("")
                continue
            
            task_performance = result.get('task_performance', {})
            for task_name, task_result in task_performance.items():
                if 'error' in task_result:
                    report.append(f"- **{task_name}**: Error - {task_result['error']}")
                else:
                    eval_metrics = task_result.get('eval_metrics', {})
                    if eval_metrics:
                        # Find primary metric
                        primary_metric = None
                        for metric_key in ['eval_accuracy', 'eval_f1', 'eval_matthews_correlation']:
                            if metric_key in eval_metrics:
                                primary_metric = eval_metrics[metric_key]
                                break
                        
                        if primary_metric is not None:
                            report.append(f"- **{task_name}**: {primary_metric:.3f}")
                        
                        training_time = task_result.get('training_time', 0)
                        if training_time > 0:
                            report.append(f"  - Training time: {training_time:.1f}s")
            
            report.append("")
        
        # Conclusions
        report.append("## Conclusions")
        report.append("")
        
        # Generate conclusions based on analysis
        conclusions = self._generate_conclusions(statistical_comparisons, memory_comparison, performance_analysis)
        for conclusion in conclusions:
            report.append(f"- {conclusion}")
        
        report.append("")
        
        # Appendix
        report.append("## Appendix")
        report.append("")
        report.append("### Methodology")
        report.append("- Statistical significance tested using Welch's t-test")
        report.append("- Effect sizes calculated using Cohen's d")
        report.append("- Memory scaling analyzed using power-law regression")
        report.append("- Multiple random seeds used for robust evaluation")
        report.append("")
        
        return "\n".join(report)
    
    def _generate_conclusions(self, statistical_comparisons: Dict, memory_comparison: Dict, performance_analysis: Dict) -> List[str]:
        """Generate conclusions based on analysis results"""
        
        conclusions = []
        
        # Performance conclusions
        if statistical_comparisons:
            significant_tasks = [task for task, comp in statistical_comparisons.items() 
                               if comp['statistical_tests']['welch_t_test']['significant']]
            
            if significant_tasks:
                rev_better = [task for task, comp in statistical_comparisons.items() 
                             if comp['comparison']['difference'] > 0 and 
                             comp['statistical_tests']['welch_t_test']['significant']]
                
                if rev_better:
                    conclusions.append(f"Reversible models significantly outperform standard models on {len(rev_better)} task(s): {', '.join(rev_better)}")
                else:
                    conclusions.append("Standard models generally perform better than reversible models on evaluated tasks")
            else:
                conclusions.append("No statistically significant performance differences found between model types")
        
        # Memory conclusions
        if memory_comparison:
            memory_at_length = memory_comparison.get('memory_at_length', {})
            if memory_at_length:
                avg_savings = np.mean([info['savings_percent'] for info in memory_at_length.values()])
                if avg_savings > 5:
                    conclusions.append(f"Reversible models achieve average memory savings of {avg_savings:.1f}%")
                elif avg_savings < -5:
                    conclusions.append(f"Reversible models use {-avg_savings:.1f}% more memory on average")
                else:
                    conclusions.append("Memory usage is comparable between model types")
        
        # Training efficiency conclusions
        if performance_analysis:
            efficiency_comparison = performance_analysis.get('efficiency_comparison', {})
            
            if 'training_time' in efficiency_comparison:
                time_diff = efficiency_comparison['training_time']['comparison']['relative_difference']
                if time_diff > 0.1:
                    conclusions.append(f"Reversible models take {time_diff:.1%} longer to train")
                elif time_diff < -0.1:
                    conclusions.append(f"Reversible models train {-time_diff:.1%} faster")
                else:
                    conclusions.append("Training times are comparable between model types")
        
        if not conclusions:
            conclusions.append("Further evaluation needed to draw definitive conclusions")
        
        return conclusions
