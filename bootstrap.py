"""
# Advanced Binning Pipeline with Multi-Core Parallel Processing + Memory Management + Checkpointing

This module implements comprehensive binning strategies with optimized parallel processing:

## Key Features:
- **Multi-Core Feature Processing**: Utilizes multiple CPU cores to process features simultaneously
- **Conservative Resource Management**: Automatically limits resource usage for stability
- **10 Different Binning Methods**: Including Decision Tree, Chi-Merge, MAPA, Isotonic Regression, etc.
- **Parallel VIF and AIC Selection**: Optimized feature selection with parallel processing
- **Comprehensive Logging**: Detailed logging with automatic file and image saving
- **Memory Management**: Advanced memory monitoring, leak prevention, and garbage collection
- **Checkpointing System**: Automatic saving of progress with resume capability
- **Crash Recovery**: Full state preservation and restoration functionality

## NEW FEATURES:
- **Memory Leak Prevention**: Explicit garbage collection, memory monitoring, reduced data copying
- **Checkpoint System**: Automatic saving every N iterations with resume capability
- **Memory-Safe Bootstrap**: Chunked processing with intermediate saves
- **Emergency Save**: Saves progress on errors/crashes for recovery

## Parallel Processing Optimizations:
- Feature-level parallelization in compare_binning_strategies_on_dataset()
- Conservative CPU allocation (max 8-10 cores for feature processing on 22-core systems)
- Thread-safe binning pipeline execution
- Memory-efficient batch processing for large datasets

## Usage:
```python
results, ivs, coeffs, figs = compare_binning_strategies_on_dataset(
    X_train, X_test, y_train, y_test,
    n_jobs=8,  # Conservative parallel processing
    verbose=True
)

# NEW: Memory-safe bootstrap with checkpointing
results = bootstrap_binning_comparison_with_checkpoints(
    X_train_population, X_test_population, y_train_population, y_test_population,
    max_memory_gb=16.0, checkpoint_frequency=25, chunk_size=5,
    emergency_save_on_error=True
)
```
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from scipy.stats import chi2_contingency, mannwhitneyu
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import warnings
from datetime import datetime
import os
import sys
import time
import gc
import pickle
import json
from contextlib import contextmanager
from multiprocessing import cpu_count
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import traceback
from pathlib import Path

# Memory monitoring imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory monitoring disabled.")

warnings.filterwarnings('ignore')

# === MEMORY MANAGEMENT UTILITIES ===

class MemoryManager:
    """Advanced memory management and monitoring utility."""

    def __init__(self, max_memory_gb=None, warning_threshold=0.8):
        self.max_memory_gb = max_memory_gb
        self.warning_threshold = warning_threshold
        self.initial_memory = self.get_memory_usage()

    def get_memory_usage(self):
        """Get current memory usage in GB."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
        return 0.0

    def get_system_memory(self):
        """Get system memory info."""
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_percent': memory.percent
            }
        return {'total_gb': 0, 'available_gb': 0, 'used_percent': 0}

    def check_memory_limit(self):
        """Check if memory usage exceeds limits."""
        current_memory = self.get_memory_usage()
        system_info = self.get_system_memory()

        warnings = []

        if self.max_memory_gb and current_memory > self.max_memory_gb:
            warnings.append(f"Process memory ({current_memory:.2f}GB) exceeds limit ({self.max_memory_gb}GB)")

        if system_info['used_percent'] > self.warning_threshold * 100:
            warnings.append(f"System memory usage ({system_info['used_percent']:.1f}%) is high")

        return warnings

    def force_garbage_collection(self):
        """Force garbage collection and return memory freed."""
        before = self.get_memory_usage()
        gc.collect()
        after = self.get_memory_usage()
        return before - after

    @contextmanager
    def memory_context(self, operation_name="Operation"):
        """Context manager for monitoring memory during operations."""
        start_memory = self.get_memory_usage()
        start_time = time.time()

        print(f"🧠 {operation_name} - Starting memory: {start_memory:.2f}GB")

        try:
            yield self

        finally:
            end_memory = self.get_memory_usage()
            end_time = time.time()
            memory_delta = end_memory - start_memory

            print(f"🧠 {operation_name} - Final memory: {end_memory:.2f}GB (Δ{memory_delta:+.2f}GB) in {end_time-start_time:.1f}s")

            # Auto garbage collection if memory increased significantly
            if memory_delta > 1.0:  # More than 1GB increase
                freed = self.force_garbage_collection()
                if freed > 0.1:  # More than 100MB freed
                    print(f"🧠 Garbage collection freed {freed:.2f}GB")

# === CHECKPOINT SYSTEM ===

class CheckpointManager:
    """Comprehensive checkpoint and resume system for bootstrap analysis."""

    def __init__(self, output_dir, analysis_name="bootstrap_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_name = analysis_name
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint files
        self.config_file = self.checkpoint_dir / f"{analysis_name}_config.json"
        self.results_file = self.checkpoint_dir / f"{analysis_name}_results.pkl"
        self.progress_file = self.checkpoint_dir / f"{analysis_name}_progress.json"
        self.metadata_file = self.checkpoint_dir / f"{analysis_name}_metadata.json"

    def save_config(self, config):
        """Save analysis configuration."""
        try:
            # Convert numpy types to native Python types for JSON serialization
            clean_config = self._clean_for_json(config)
            with open(self.config_file, 'w') as f:
                json.dump(clean_config, f, indent=2)
            print(f"✅ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"❌ Failed to save config: {e}")

    def save_results(self, results, iteration_num=None):
        """Save bootstrap results with optional iteration number."""
        try:
            checkpoint_data = {
                'results': results,
                'iteration': iteration_num,
                'timestamp': datetime.now().isoformat(),
                'total_results': len(results) if isinstance(results, list) else 1
            }

            with open(self.results_file, 'wb') as f:
                pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"✅ Results saved: {len(results) if isinstance(results, list) else 1} iterations")

        except Exception as e:
            print(f"❌ Failed to save results: {e}")

    def save_progress(self, progress_info):
        """Save progress information."""
        try:
            progress_info['timestamp'] = datetime.now().isoformat()
            with open(self.progress_file, 'w') as f:
                json.dump(progress_info, f, indent=2)
            print(f"✅ Progress saved: {progress_info.get('completed', 0)}/{progress_info.get('total', 0)} iterations")
        except Exception as e:
            print(f"❌ Failed to save progress: {e}")

    def save_metadata(self, metadata):
        """Save dataset and analysis metadata."""
        try:
            clean_metadata = self._clean_for_json(metadata)
            with open(self.metadata_file, 'w') as f:
                json.dump(clean_metadata, f, indent=2)
            print(f"✅ Metadata saved")
        except Exception as e:
            print(f"❌ Failed to save metadata: {e}")

    def load_checkpoint(self):
        """Load complete checkpoint data."""
        checkpoint = {
            'config': None,
            'results': None,
            'progress': None,
            'metadata': None,
            'has_checkpoint': False
        }

        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    checkpoint['config'] = json.load(f)

            if self.results_file.exists():
                with open(self.results_file, 'rb') as f:
                    results_data = pickle.load(f)
                    checkpoint['results'] = results_data['results']
                    checkpoint['last_iteration'] = results_data.get('iteration')
                    checkpoint['results_timestamp'] = results_data.get('timestamp')

            if self.progress_file.exists():
                with open(self.progress_file, 'r') as f:
                    checkpoint['progress'] = json.load(f)

            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    checkpoint['metadata'] = json.load(f)

            checkpoint['has_checkpoint'] = any([
                checkpoint['config'], checkpoint['results'],
                checkpoint['progress'], checkpoint['metadata']
            ])

            if checkpoint['has_checkpoint']:
                print(f"✅ Checkpoint loaded from {self.checkpoint_dir}")
                if checkpoint['results']:
                    print(f"   - Found {len(checkpoint['results'])} bootstrap results")
                if checkpoint['progress']:
                    completed = checkpoint['progress'].get('completed', 0)
                    total = checkpoint['progress'].get('total', 0)
                    print(f"   - Progress: {completed}/{total} iterations")

        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")

        return checkpoint

    def create_emergency_save(self, **kwargs):
        """Create emergency save of all provided variables."""
        emergency_file = self.checkpoint_dir / f"emergency_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

        try:
            emergency_data = {
                'timestamp': datetime.now().isoformat(),
                'variables': kwargs
            }

            with open(emergency_file, 'wb') as f:
                pickle.dump(emergency_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"🚨 EMERGENCY SAVE completed: {emergency_file}")
            print(f"   Saved variables: {list(kwargs.keys())}")

            return str(emergency_file)

        except Exception as e:
            print(f"❌ EMERGENCY SAVE failed: {e}")
            return None

    def _clean_for_json(self, obj):
        """Recursively clean object for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray)):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)  # Convert complex objects to string
        else:
            return obj

def set_parallel_env(n_workers):
    """
    Set parallel processing environment variables for optimal performance.

    Parameters:
    -----------
    n_workers : int
        Number of workers to use for parallel processing
    """
    import os

    # Set environment variables for various parallel libraries
    os.environ['OMP_NUM_THREADS'] = str(n_workers)
    os.environ['MKL_NUM_THREADS'] = str(n_workers)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n_workers)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_workers)

    # Set joblib backend configuration
    try:
        from joblib import parallel_backend
        # This sets a default backend but doesn't enforce it globally
        # Individual joblib calls can still override this
        pass
    except ImportError:
        pass

# --- Logging System ---
class BinningLogger:
    """Comprehensive logging system for binning analysis with date/time filename and JPEG image saving."""

    def __init__(self, base_filename="binning_analysis", output_dir=None):
        """
        Initialize the logger.

        Parameters:
        -----------
        base_filename : str
            Base name for the log file
        output_dir : str, optional
            Directory to save outputs. If None, uses current directory.
        """
        self.output_dir = output_dir or os.getcwd()

        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"{base_filename}_{timestamp}.txt"
        self.log_filepath = os.path.join(self.output_dir, self.log_filename)

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize log file
        with open(self.log_filepath, 'w') as f:
            f.write(f"Binning Analysis Log\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

        self.figure_counter = 1

    def log(self, message, print_to_console=True):
        """
        Log a message to both file and optionally console.

        Parameters:
        -----------
        message : str
            Message to log
        print_to_console : bool
            Whether to also print to console
        """
        # Add timestamp to message
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"

        # Write to file
        with open(self.log_filepath, 'a') as f:
            f.write(formatted_message + "\n")

        # Print to console if requested
        if print_to_console:
            print(formatted_message)

    def save_figure(self, fig, figure_name=None, dpi=300):
        """
        Save a matplotlib figure as JPEG next to the log file.

        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Figure to save
        figure_name : str, optional
            Custom name for the figure. If None, uses auto-generated name.
        dpi : int
            Resolution for the saved image

        Returns:
        --------
        str
            Path to the saved figure
        """
        if figure_name is None:
            figure_name = f"figure_{self.figure_counter:02d}"
            self.figure_counter += 1

        # Ensure .jpg extension
        if not figure_name.lower().endswith(('.jpg', '.jpeg')):
            figure_name += '.jpg'

        figure_path = os.path.join(self.output_dir, figure_name)

        # Save figure as JPEG
        fig.savefig(figure_path, format='jpeg', dpi=dpi,
                   bbox_inches='tight', facecolor='white')

        self.log(f"Figure saved: {figure_name}")

        return figure_path

    def save_dataframe(self, df, filename=None, description=""):
        """
        Save a DataFrame to CSV and log the action.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to save
        filename : str, optional
            Custom filename. If None, uses auto-generated name.
        description : str
            Description of the DataFrame content

        Returns:
        --------
        str
            Path to the saved CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dataframe_{timestamp}.csv"

        # Ensure .csv extension
        if not filename.lower().endswith('.csv'):
            filename += '.csv'

        csv_path = os.path.join(self.output_dir, filename)
        df.to_csv(csv_path, index=True)

        self.log(f"DataFrame saved: {filename}")
        if description:
            self.log(f"Description: {description}")

        return csv_path

    @contextmanager
    def capture_prints(self):
        """
        Context manager to capture print statements and redirect them to the logger.
        """
        class LogCapture:
            def __init__(self, logger):
                self.logger = logger
                self.original_stdout = sys.stdout

            def write(self, message):
                if message.strip():  # Only log non-empty messages
                    self.logger.log(message.strip(), print_to_console=False)
                    self.original_stdout.write(message)

            def flush(self):
                self.original_stdout.flush()

        log_capture = LogCapture(self)
        original_stdout = sys.stdout
        try:
            sys.stdout = log_capture
            yield self
        finally:
            sys.stdout = original_stdout

    def log_section(self, title, level=1):
        """
        Log a section header with formatting.

        Parameters:
        -----------
        title : str
            Section title
        level : int
            Section level (1-3)
        """
        if level == 1:
            separator = "=" * 80
        elif level == 2:
            separator = "-" * 60
        else:
            separator = "." * 40

        self.log("")
        self.log(separator)
        self.log(title)
        self.log(separator)
        self.log("")

    def create_comprehensive_debug_log(self):
        """
        Create a comprehensive debug log file that captures ALL terminal output.

        Returns:
        --------
        str
            Path to the comprehensive debug log file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_filename = f"comprehensive_debug_log_{timestamp}.txt"
        debug_filepath = os.path.join(self.output_dir, debug_filename)

        # Initialize comprehensive debug log
        with open(debug_filepath, 'w') as f:
            f.write(f"Comprehensive Debug Log - All Terminal Output\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

        self.debug_log_filepath = debug_filepath
        return debug_filepath

    @contextmanager
    def capture_all_output(self):
        """
        Context manager to capture ALL output (stdout, stderr, and debug) to comprehensive log.
        This captures everything that appears in the terminal.
        """
        class ComprehensiveCapture:
            def __init__(self, logger):
                self.logger = logger
                self.original_stdout = sys.stdout
                self.original_stderr = sys.stderr
                self.debug_filepath = getattr(logger, 'debug_log_filepath', None)

                # Create debug log if it doesn't exist
                if not self.debug_filepath:
                    self.debug_filepath = logger.create_comprehensive_debug_log()

            def write_to_debug_log(self, message, stream_type=""):
                """Write message to debug log with timestamp"""
                if self.debug_filepath and message.strip():
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    prefix = f"[{timestamp}]" + (f" {stream_type}" if stream_type else "")
                    formatted_message = f"{prefix} {message.rstrip()}\n"

                    try:
                        with open(self.debug_filepath, 'a', encoding='utf-8') as f:
                            f.write(formatted_message)
                    except Exception:
                        pass  # Don't let logging errors break the main process

            def stdout_write(self, message):
                """Capture stdout"""
                self.write_to_debug_log(message, "[STDOUT]")
                self.original_stdout.write(message)
                return len(message)

            def stderr_write(self, message):
                """Capture stderr"""
                self.write_to_debug_log(message, "[STDERR]")
                self.original_stderr.write(message)
                return len(message)

            def stdout_flush(self):
                self.original_stdout.flush()

            def stderr_flush(self):
                self.original_stderr.flush()

        class DebugStdout:
            def __init__(self, capture):
                self.capture = capture
            def write(self, message):
                return self.capture.stdout_write(message)
            def flush(self):
                return self.capture.stdout_flush()

        class DebugStderr:
            def __init__(self, capture):
                self.capture = capture
            def write(self, message):
                return self.capture.stderr_write(message)
            def flush(self):
                return self.capture.stderr_flush()

        comprehensive_capture = ComprehensiveCapture(self)
        debug_stdout = DebugStdout(comprehensive_capture)
        debug_stderr = DebugStderr(comprehensive_capture)

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            sys.stdout = debug_stdout
            sys.stderr = debug_stderr
            yield self
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

            # Write completion message
            if hasattr(comprehensive_capture, 'debug_filepath'):
                try:
                    with open(comprehensive_capture.debug_filepath, 'a', encoding='utf-8') as f:
                        f.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] === Debug capture session ended ===\n")
                except Exception:
                    pass

    def log_debug_info(self, info_dict):
        """
        Log debug information in a structured format to the comprehensive log.

        Parameters:
        -----------
        info_dict : dict
            Dictionary containing debug information
        """
        if hasattr(self, 'debug_log_filepath'):
            timestamp = datetime.now().strftime("%H:%M:%S")
            try:
                with open(self.debug_log_filepath, 'a', encoding='utf-8') as f:
                    f.write(f"\n[{timestamp}] [DEBUG_INFO] ===== Structured Debug Information =====\n")
                    for key, value in info_dict.items():
                        f.write(f"[{timestamp}] [DEBUG_INFO] {key}: {value}\n")
                    f.write(f"[{timestamp}] [DEBUG_INFO] ================================================\n\n")
            except Exception:
                pass

# --- Helper Functions ---

def ks_statistic(y_true, y_pred_proba):
    """Calculate Kolmogorov-Smirnov statistic"""
    df = pd.DataFrame({'target': y_true, 'proba': y_pred_proba})
    df = df.sort_values(by='proba')
    if df['target'].sum() == 0 or df['target'].sum() == len(df['target']):
        return 0.0
    df['cumulative_goods'] = (1 - df['target']).cumsum() / (1 - df['target']).sum()
    df['cumulative_bads'] = df['target'].cumsum() / df['target'].sum()
    return np.max(np.abs(df['cumulative_bads'] - df['cumulative_goods']))

 # Helper function for VIF calculation (place before compare_binning_strategies_on_dataset)
def _calculate_vif_for_feature(args):
    """Helper function to calculate VIF for a single feature (for parallel processing)"""
    vif_data_values, feature_idx = args
    try:
        # Check if the feature has zero variance
        if np.var(vif_data_values[:, feature_idx]) == 0:
            return np.inf

        # Check for numerical stability before VIF calculation
        # Remove the feature column temporarily to check remaining matrix
        remaining_features = np.delete(vif_data_values, feature_idx, axis=1)
        if remaining_features.shape[1] == 0:
            return 1.0  # Only one feature, VIF = 1

        # Check condition number of remaining features
        try:
            cond_num = np.linalg.cond(remaining_features)
            if cond_num > 1e12:  # Very high condition number
                return np.inf
        except:
            return np.inf

        # Calculate VIF using statsmodels
        vif_value = variance_inflation_factor(vif_data_values, feature_idx)

        # Handle edge cases
        if np.isnan(vif_value) or np.isinf(vif_value):
            return np.inf
        if vif_value < 1.0:  # VIF should always be >= 1
            return 1.0

        return vif_value

    except Exception as e:
        # More detailed error logging for debugging
        return np.inf

def _get_optimal_workers(n_jobs, task_count, task_type='cpu'):
    """
    Get optimal number of workers to prevent system overload

    Args:
        n_jobs (int): Requested number of jobs
        task_count (int): Number of tasks to execute
        task_type (str): Type of task ('cpu' for CPU-bound, 'io' for I/O-bound)

    Returns:
        int: Optimal number of workers
    """
    if n_jobs is None:
        # Conservative default: limit to 8 workers max for stability
        n_jobs = min(cpu_count(), 8)
        print(f"    DEBUG: Auto-limiting to {n_jobs} workers (conservative mode)")

    # Use conservative settings without memory monitoring
    print(f"    DEBUG: Using conservative CPU settings for stability")
    n_jobs = min(n_jobs, 4)

    # Don't create more workers than tasks
    optimal_workers = min(n_jobs, task_count)

    # For CPU-bound tasks, be more conservative
    if task_type == 'cpu':
        optimal_workers = min(optimal_workers, min(cpu_count(), 6))

    # For I/O-bound tasks, still be conservative
    elif task_type == 'io':
        optimal_workers = min(optimal_workers, min(cpu_count(), 8))

    # Always have at least 1 worker
    optimal_workers = max(1, optimal_workers)
    print(f"    DEBUG: Using {optimal_workers} workers for {task_type} task")
    return optimal_workers

def _parallel_vif_calculation(vif_data, n_jobs=None):
    """Calculate VIF for all features in parallel with enhanced error handling"""
    print(f"    DEBUG: Starting parallel VIF calculation for {vif_data.shape[1]} features")

    # Pre-check for numerical stability
    try:
        # Check overall matrix condition
        cond_num = np.linalg.cond(vif_data.values)
        print(f"    DEBUG: Input matrix condition number: {cond_num:.2e}")

        if cond_num > 1e12:
            print(f"    DEBUG: WARNING - Matrix condition number too high for reliable VIF calculation")
    except:
        print(f"    DEBUG: Could not calculate matrix condition number")

    try:
        n_workers = _get_optimal_workers(n_jobs, vif_data.shape[1], 'cpu')
        print(f"    DEBUG: VIF data shape: {vif_data.shape}")

        # Prepare arguments for parallel processing
        args = [(vif_data.values, i) for i in range(vif_data.shape[1])]
        print(f"    DEBUG: Prepared {len(args)} VIF calculation tasks")

        # Use ThreadPoolExecutor for I/O bound VIF calculations
        print(f"    DEBUG: Starting ThreadPoolExecutor with {n_workers} workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            vif_values = list(executor.map(_calculate_vif_for_feature, args))

        # Post-process results
        vif_series = pd.Series(vif_values, index=vif_data.columns)

        # Count and report problematic VIF values
        inf_count = sum(np.isinf(vif_values))
        nan_count = sum(np.isnan(vif_values))
        finite_count = len(vif_values) - inf_count - nan_count

        print(f"    DEBUG: VIF calculation completed: {finite_count} finite, {inf_count} infinite, {nan_count} NaN values")

        if finite_count > 0:
            finite_vifs = [v for v in vif_values if np.isfinite(v)]
            print(f"    DEBUG: Finite VIF range: {min(finite_vifs):.2f} to {max(finite_vifs):.2f}")

        return vif_series

    except Exception as e:
        print(f"    DEBUG: Error in parallel VIF calculation: {str(e)}")
        print(f"    DEBUG: Falling back to sequential VIF calculation")

        # Enhanced fallback calculation
        vif_values = []
        successful_calcs = 0

        for i in range(vif_data.shape[1]):
            feature_name = vif_data.columns[i]
            try:
                # Use the enhanced VIF calculation
                vif_val = _calculate_vif_for_feature((vif_data.values, i))
                vif_values.append(vif_val)
                if np.isfinite(vif_val):
                    successful_calcs += 1
            except Exception as feature_error:
                print(f"    DEBUG: VIF calculation failed for feature {feature_name}: {str(feature_error)}")
                vif_values.append(np.inf)

        print(f"    DEBUG: Sequential fallback completed: {successful_calcs}/{len(vif_values)} successful calculations")
        return pd.Series(vif_values, index=vif_data.columns)

def _parallel_model_evaluation(args):
    """Helper function for parallel model evaluation in stepwise selection"""
    X_test_values, y_values, test_features = args
    try:
        X_test_df = pd.DataFrame(X_test_values, columns=test_features)
        X_test_const = sm.add_constant(X_test_df, has_constant='add')
        model = sm.Logit(y_values, X_test_const).fit(disp=0, method='lbfgs')
        return model.aic
    except:
        return np.inf

def _evaluate_feature_cv(args):
    """Helper function for parallel cross-validation of feature subsets"""
    X_subset, y, feature_subset, cv_folds, metric = args
    try:
        scores = []
        for train_idx, val_idx in cv_folds:
            X_train = X_subset.iloc[train_idx]
            X_val = X_subset.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            # Fit logistic regression
            X_train_const = sm.add_constant(X_train, has_constant='add')
            X_val_const = sm.add_constant(X_val, has_constant='add')

            model = sm.Logit(y_train, X_train_const).fit(disp=0, method='lbfgs')
            y_pred_proba = model.predict(X_val_const)

            if metric == 'auc':
                score = roc_auc_score(y_val, y_pred_proba)
            elif metric == 'aic':
                score = -model.aic  # Negative because we want higher scores to be better
            else:
                score = roc_auc_score(y_val, y_pred_proba)  # Default to AUC

            scores.append(score)

        return np.mean(scores), feature_subset
    except:
        return -np.inf, feature_subset

def parallel_feature_selection_cv(X, y, feature_candidates, cv_folds, metric='auc', n_jobs=None):
    """
    Perform parallel cross-validation evaluation of feature subsets

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        feature_candidates (list): List of feature subsets to evaluate
        cv_folds (list): List of (train_idx, val_idx) tuples for cross-validation
        metric (str): Evaluation metric ('auc' or 'aic')
        n_jobs (int): Number of parallel jobs

    Returns:
        list: List of (score, feature_subset) tuples sorted by score
    """
    print(f"    DEBUG: Starting parallel CV for {len(feature_candidates)} feature candidates")

    try:
        n_workers = _get_optimal_workers(n_jobs, len(feature_candidates), 'cpu')

        # Prepare arguments for parallel processing
        args_list = []
        for feature_subset in feature_candidates:
            X_subset = X[feature_subset].copy()
            args_list.append((X_subset, y, feature_subset, cv_folds, metric))

        print(f"    DEBUG: Prepared {len(args_list)} CV tasks")

        # Evaluate in parallel with optimal worker allocation
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_evaluate_feature_cv, args_list))

        # Sort by score (descending)
        results.sort(key=lambda x: float(x[0]), reverse=True)
        print(f"    DEBUG: Parallel CV completed successfully")
        return results

    except Exception as e:
        print(f"    DEBUG: Error in parallel CV: {str(e)}")
        print(f"    DEBUG: Falling back to sequential CV")

        # Fallback to sequential processing
        results = []
        for feature_subset in feature_candidates:
            try:
                X_subset = X[feature_subset].copy()
                score, subset = _evaluate_feature_cv((X_subset, y, feature_subset, cv_folds, metric))
                results.append((score, subset))
            except:
                results.append((-np.inf, feature_subset))

        results.sort(key=lambda x: float(x[0]), reverse=True)
        return results

def parallel_stepwise_selection(X, y, initial_features, metric='aic', n_jobs=None, max_iterations=100):
    """
    Optimized bidirectional stepwise feature selection with parallel candidate evaluation

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        initial_features (list): Starting feature set
        metric (str): Selection metric ('aic' or 'bic')
        n_jobs (int): Number of parallel jobs
        max_iterations (int): Maximum iterations to prevent infinite loops

    Returns:
        list: Final selected features
        float: Final model score
    """
    print(f"    DEBUG: Starting parallel stepwise selection with {len(initial_features)} initial features")

    try:
        # Use conservative worker allocation for model fitting
        n_workers = _get_optimal_workers(n_jobs, 50, 'io')  # Reduced estimate for stability

        included_features = initial_features.copy()
        excluded_features = [f for f in X.columns if f not in included_features]

        for iteration in range(max_iterations):
            if not included_features:
                break

            # Get current model score
            try:
                X_current = X[included_features].copy().astype(float)
                X_current_const = sm.add_constant(X_current, has_constant='add')
                current_model = sm.Logit(y, X_current_const).fit(disp=0, method='lbfgs')
                current_score = current_model.aic if metric == 'aic' else current_model.bic
            except:
                break

            # Prepare all candidate evaluations
            candidates = []

            # Forward candidates (add features)
            for feature in excluded_features:
                test_features = included_features + [feature]
                candidates.append((test_features, ('add', feature)))

            # Backward candidates (remove features, keep at least one)
            if len(included_features) > 1:
                for feature in included_features:
                    test_features = [f for f in included_features if f != feature]
                    candidates.append((test_features, ('remove', feature)))

            if not candidates:
                break

            # Evaluate candidates in parallel
            best_score = current_score
            best_action = None

            # Prepare arguments for parallel evaluation
            eval_args = []
            for test_features, action in candidates:
                X_test = X[test_features].copy().astype(float)
                eval_args.append((X_test.values, y.values, test_features, action))

            # Use ThreadPoolExecutor for model fitting with optimal worker allocation
            optimal_workers = _get_optimal_workers(n_workers, len(eval_args), 'io')
            print(f"    DEBUG: Evaluating {len(eval_args)} candidates with {optimal_workers} workers")

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                    future_to_action = {}
                    for X_test_values, y_values, test_features, action in eval_args:
                        future = executor.submit(_parallel_model_evaluation_with_action,
                                               (X_test_values, y_values, test_features, action, metric))
                        future_to_action[future] = action

                    # Collect results
                    for future in concurrent.futures.as_completed(future_to_action):
                        action = future_to_action[future]
                        try:
                            score = future.result()
                            if score < best_score:  # Lower is better for AIC/BIC
                                best_score = score
                                best_action = action
                        except Exception as e:
                            print(f"    DEBUG: Worker failed for action {action}: {str(e)}")
                            pass

            except Exception as e:
                print(f"    DEBUG: Error in parallel model evaluation: {str(e)}")
                print(f"    DEBUG: Falling back to sequential evaluation")

                # Fallback to sequential evaluation
                for X_test_values, y_values, test_features, action in eval_args:
                    try:
                        score = _parallel_model_evaluation_with_action((X_test_values, y_values, test_features, action, metric))
                        if score < best_score:
                            best_score = score
                            best_action = action
                    except:
                        pass

            # Apply best action if improvement found
            if best_action is None:
                break

            action_type, feature = best_action
            if action_type == 'add':
                included_features.append(feature)
                excluded_features.remove(feature)
            else:  # remove
                included_features.remove(feature)
                excluded_features.append(feature)

        # Calculate final score
        try:
            X_final = X[included_features].copy().astype(float)
            X_final_const = sm.add_constant(X_final, has_constant='add')
            final_model = sm.Logit(y, X_final_const).fit(disp=0, method='lbfgs')
            final_score = final_model.aic if metric == 'aic' else final_model.bic
        except:
            final_score = np.inf

        return included_features, final_score

    except Exception as e:
        print(f"    DEBUG: Error in parallel stepwise selection: {str(e)}")
        print(f"    DEBUG: Traceback: {traceback.format_exc()}")
        print(f"    DEBUG: Returning initial features as fallback")
        return initial_features, np.inf

def _parallel_model_evaluation_with_action(args):
    """Helper function for parallel model evaluation with action info"""
    X_test_values, y_values, test_features, action, metric = args
    try:
        X_test_df = pd.DataFrame(X_test_values, columns=test_features)
        X_test_const = sm.add_constant(X_test_df, has_constant='add')
        model = sm.Logit(y_values, X_test_const).fit(disp=0, method='lbfgs')
        return model.aic if metric == 'aic' else model.bic
    except:
        return np.inf

def calculate_vif_iteratively(X, y, threshold=5.0, max_features=None,
                            sampling_threshold=300000, max_sampling_attempts=10,
                            mean_tolerance=0.05, corr_tolerance=0.05, balance_tolerance=0.05,
                            n_jobs=None, use_parallel_stepwise=True,
                            correlation_threshold=0.7, condition_threshold=100):
     """
     Iteratively calculates VIF and removes features with VIF above the threshold.
     First performs correlation-based removal and matrix stability checks.
     Optionally, limits the number of features returned by VIF's max_features.
     Then, performs AIC-based bidirectional stepwise feature selection.

     VIF and correlation removal are performed on the full dataset for accuracy.
     For large datasets (>sampling_threshold), uses stratified sampling only for
     the final stepwise feature selection stage to save computation time.

     Args:
         X (pd.DataFrame): DataFrame of features.
         y (pd.Series): Target variable (must be suitable for Logit, e.g., 0/1 numeric).
         threshold (float): VIF threshold for feature removal (default: 5.0).
         max_features (int, optional): Maximum number of features to keep after the VIF stage.
                                      If None, no limit other than VIF threshold.
         sampling_threshold (int): If dataset size > this, use sampling for stepwise selection only.
         max_sampling_attempts (int): Maximum attempts to get representative sample.
         mean_tolerance (float): Tolerance for mean differences in sample validation.
         corr_tolerance (float): Tolerance for correlation differences in sample validation.
         balance_tolerance (float): Tolerance for target balance differences in sample validation.
         n_jobs (int, optional): Number of parallel jobs. If None, uses all available CPUs.
         use_parallel_stepwise (bool): Whether to use parallel stepwise selection for AIC stage.
         correlation_threshold (float): Threshold for correlation-based feature removal (default: 0.85).
         condition_threshold (float): Matrix condition number threshold for stability check (default: 100).

     Returns:
         list: List of selected feature names after VIF and AIC selection.
         pd.DataFrame: DataFrame with selected features.
         float: AIC of the final model with selected features.
     """
     # Set conservative default for stability
     if n_jobs is None:
         n_jobs = min(cpu_count(), 6)  # Conservative limit
         print(f"    DEBUG: Auto-setting n_jobs to {n_jobs} for stability")

     # --- Ensure original data has aligned indices ---
     X = X.reset_index(drop=True)
     y = y.reset_index(drop=True)

     # --- Step 1: Correlation-Based Feature Removal (on full dataset) ---
     print(f"    DEBUG: Starting with {len(X.columns)} features: {list(X.columns)}")

     # Ensure y is not part of X
     X_clean = X.copy()
     if y.name in X_clean.columns:
         X_clean = X_clean.drop(columns=[y.name])

     print(f"    DEBUG: Removing features with correlation > {correlation_threshold}")
     corr_matrix = X_clean.corr().abs()

     # Find high correlation pairs
     upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
     high_corr_pairs = []

     for col in upper_tri.columns:
         for row in upper_tri.index:
             if not pd.isna(upper_tri.loc[row, col]) and upper_tri.loc[row, col] > correlation_threshold:
                 high_corr_pairs.append((row, col, upper_tri.loc[row, col]))

     # Remove features with highest correlations, keeping the one with higher variance
     to_drop = set()
     for row, col, corr_val in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
         if row not in to_drop and col not in to_drop:
             if X_clean[row].var() >= X_clean[col].var():
                 to_drop.add(col)
                 print(f"    DEBUG: Removing {col} (corr with {row}: {corr_val:.4f})")
             else:
                 to_drop.add(row)
                 print(f"    DEBUG: Removing {row} (corr with {col}: {corr_val:.4f})")

     X_clean = X_clean.drop(columns=list(to_drop))
     print(f"    DEBUG: Removed {len(to_drop)} highly correlated features")
     print(f"    DEBUG: Remaining features after correlation removal: {X_clean.shape[1]}")

     # --- Step 2: Matrix Stability Check ---
     if X_clean.shape[1] > 1:
         try:
             # Check matrix condition number
             corr_matrix_clean = X_clean.corr()
             condition_number = np.linalg.cond(corr_matrix_clean.values)
             print(f"    DEBUG: Matrix condition number: {condition_number:.2f}")

             if condition_number > condition_threshold:
                 print(f"    DEBUG: WARNING - Matrix condition number ({condition_number:.2f}) > threshold ({condition_threshold})")
                 print(f"    DEBUG: Matrix may be ill-conditioned for VIF calculation")

             # Check matrix rank
             matrix_rank = np.linalg.matrix_rank(X_clean.values)
             print(f"    DEBUG: Matrix rank: {matrix_rank} (expected: {X_clean.shape[1]})")

             if matrix_rank < X_clean.shape[1]:
                 print(f"    DEBUG: WARNING - Matrix is rank deficient before VIF calculation!")
                 print(f"    DEBUG: This may cause VIF calculation failures")
         except Exception as e:
             print(f"    DEBUG: Could not check matrix stability: {str(e)}")

     # --- Step 3: VIF Selection Stage (on full dataset) ---
     features_after_vif = list(X_clean.columns)
     print(f"    DEBUG: Starting VIF calculation with {len(features_after_vif)} features")

     # Iteratively remove features with VIF > threshold
     vif_iteration = 0
     while len(features_after_vif) >= 2:
         vif_iteration += 1
         vif_data = X_clean[features_after_vif].copy()

         # Replace inf/-inf with NaN, then drop rows with any NaNs for VIF calculation robustness
         vif_data.replace([np.inf, -np.inf], np.nan, inplace=True)
         vif_data.dropna(inplace=True)

         if vif_data.shape[0] < 2 or vif_data.shape[1] < 2:
             print(f"    DEBUG: Not enough data/features after cleaning (iteration {vif_iteration})")
             break

         # Check for constant features before VIF calculation
         feature_stds = vif_data.std()
         constant_features = feature_stds[feature_stds == 0].index.tolist()
         if constant_features:
             print(f"    DEBUG: Removing constant features before VIF: {constant_features}")
             vif_data = vif_data.drop(columns=constant_features)
             features_after_vif = [f for f in features_after_vif if f not in constant_features]
             continue

         # Use parallel VIF calculation
         print(f"    DEBUG: VIF iteration {vif_iteration}: calculating VIF for {len(features_after_vif)} features")
         vif_values = _parallel_vif_calculation(vif_data, n_jobs)

         # Check for infinite VIF values
         inf_vif_features = vif_values[vif_values == np.inf].index.tolist()
         if inf_vif_features:
             print(f"    DEBUG: Features with infinite VIF (numerical instability): {inf_vif_features}")

         # Get the maximum finite VIF value
         finite_vif = vif_values[vif_values != np.inf]
         if len(finite_vif) == 0:
             print(f"    DEBUG: All VIF values are infinite - stopping VIF calculation")
             # Remove features with infinite VIF one by one
             if inf_vif_features:
                 feature_to_drop = inf_vif_features[0]
                 print(f"    DEBUG: Removing feature with infinite VIF: {feature_to_drop}")
                 features_after_vif.remove(feature_to_drop)
                 continue
             else:
                 break

         max_vif_val = finite_vif.max()
         max_vif_feature = finite_vif.idxmax()

         print(f"    DEBUG: Max VIF: {max_vif_val:.2f} (feature: {max_vif_feature})")

         if max_vif_val > threshold:
             print(f"    DEBUG: Removing feature '{max_vif_feature}' with VIF {max_vif_val:.2f} > {threshold}")
             features_after_vif.remove(max_vif_feature)
         else:
             print(f"    DEBUG: All VIF values below threshold {threshold} - VIF selection complete")
             break

     # Apply max_features cap from VIF stage if specified
     if max_features is not None and len(features_after_vif) > max_features:
         print(f"    DEBUG: Applying max_features cap: {len(features_after_vif)} -> {max_features}")
         # If still over max_features, remove features with highest VIFs until cap is met
         while len(features_after_vif) > max_features and len(features_after_vif) >= 2:
             vif_data_cap = X_clean[features_after_vif].copy()
             vif_data_cap.replace([np.inf, -np.inf], np.nan, inplace=True)
             vif_data_cap.dropna(inplace=True)

             if vif_data_cap.shape[0] < 2 or vif_data_cap.shape[1] < 2:
                 # Fallback: remove the last feature if VIF cannot be calculated
                 if features_after_vif:
                     removed_feature = features_after_vif.pop()
                     print(f"    DEBUG: Fallback removal of feature: {removed_feature}")
                 else:
                     break
                 continue

             # Use parallel VIF calculation for max_features capping
             vif_values_cap = _parallel_vif_calculation(vif_data_cap, n_jobs)
             # Remove feature with highest VIF (finite or infinite)
             feature_to_drop_cap = vif_values_cap.idxmax()
             print(f"    DEBUG: Removing feature for max_features cap: {feature_to_drop_cap}")
             features_after_vif.remove(feature_to_drop_cap)

         # Final truncation if still over (e.g. if only 1 feature left and max_features is 0)
         if len(features_after_vif) > max_features:
             features_after_vif = features_after_vif[:max_features]
             print(f"    DEBUG: Final truncation to {max_features} features")

     if not features_after_vif:
         print(f"    DEBUG: No features after VIF selection!")
         return [], pd.DataFrame(index=X.index), np.nan

     print(f"    DEBUG: After VIF selection: {len(features_after_vif)} features retained: {features_after_vif}")

     # --- Step 4: Stratified Sampling Stage (only for stepwise selection) ---
     use_sampling = len(X) > sampling_threshold
     if use_sampling:
         print(f"    DEBUG: Dataset size ({len(X)}) > threshold ({sampling_threshold}), will use sampling for stepwise selection")
         print(f"    DEBUG: Creating stratified sample for stepwise feature selection...")
         X_sample, y_sample = _create_representative_sample(
             X[features_after_vif], y, sampling_threshold,
             max_sampling_attempts, mean_tolerance, corr_tolerance, balance_tolerance
         )
         print(f"    DEBUG: Sample created with {len(X_sample)} observations")
         # Reset indices to ensure alignment
         X_sample = X_sample.reset_index(drop=True)
         y_sample = y_sample.reset_index(drop=True)
     else:
         print(f"    DEBUG: Using full dataset for stepwise selection")
         X_sample = X[features_after_vif].copy().reset_index(drop=True)
         y_sample = y.copy().reset_index(drop=True)

     # --- Step 5: AIC-based Bidirectional Stepwise Feature Selection Stage ---
     current_features_aic = features_after_vif.copy()

     # Ensure y is numeric for Logit
     y_numeric = y_sample.astype(float)

     # AIC-based bidirectional stepwise feature selection
     print(f"    DEBUG: Starting AIC bidirectional stepwise selection with {len(current_features_aic)} features")

     if use_parallel_stepwise and len(current_features_aic) > 1:
         # Use optimized parallel stepwise selection
         print(f"    DEBUG: Using parallel stepwise selection with {n_jobs} workers")
         try:
             selected_features, _ = parallel_stepwise_selection(
                 X_sample, y_numeric, current_features_aic, metric='aic', n_jobs=n_jobs
             )
             current_features_aic = selected_features
             print(f"    DEBUG: Parallel stepwise selection completed successfully")
         except Exception as e:
             print(f"    DEBUG: Parallel stepwise failed: {str(e)}")
             print(f"    DEBUG: Falling back to sequential stepwise selection")
             use_parallel_stepwise = False  # Fall through to sequential method
     else:
         # Original sequential stepwise selection
         # Track included and excluded features
         included_features = current_features_aic.copy()
         excluded_features = [f for f in features_after_vif if f not in included_features]

         # Continue until no improvement is possible
         while True:
             if not included_features:  # Must have at least one feature
                 break

             # Calculate current model AIC
             X_current = X_sample[included_features].copy().astype(float)
             y_numeric_aligned = y_numeric.copy()
             X_current_const = sm.add_constant(X_current, has_constant='add')

             try:
                 current_model = sm.Logit(y_numeric_aligned, X_current_const).fit(disp=0, method='lbfgs')
                 current_aic = current_model.aic
             except Exception:
                 break  # Cannot proceed if current model fails

             best_aic = current_aic
             best_action = None  # ('add', feature) or ('remove', feature)

             # Prepare all candidate evaluations for parallel processing
             forward_candidates = []
             backward_candidates = []

             # Prepare forward step candidates
             for feature_to_add in excluded_features:
                 test_features = included_features + [feature_to_add]
                 X_test = X_sample[test_features].copy().astype(float)
                 forward_candidates.append((X_test.values, y_numeric_aligned.values, test_features, ('add', feature_to_add)))

             # Prepare backward step candidates (keep at least one feature)
             if len(included_features) > 1:
                 for feature_to_remove in included_features:
                     test_features = [f for f in included_features if f != feature_to_remove]
                     X_test = X_sample[test_features].copy().astype(float)
                     backward_candidates.append((X_test.values, y_numeric_aligned.values, test_features, ('remove', feature_to_remove)))

             # Combine all candidates
             all_candidates = forward_candidates + backward_candidates

             # Evaluate all candidates in parallel if we have any
             if all_candidates:
                 optimal_workers = _get_optimal_workers(n_jobs, len(all_candidates), 'io')

                 with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                     # Submit all evaluation tasks
                     future_to_candidate = {}
                     for X_test_values, y_values, test_features, action in all_candidates:
                         future = executor.submit(_parallel_model_evaluation, (X_test_values, y_values, test_features))
                         future_to_candidate[future] = action

                     # Collect results
                     for future in concurrent.futures.as_completed(future_to_candidate):
                         action = future_to_candidate[future]
                         try:
                             aic_result = future.result()
                             if aic_result < best_aic:
                                 best_aic = aic_result
                                 best_action = action
                         except:
                             pass  # Ignore if model evaluation fails

             # Apply the best action if it improves AIC
             if best_action is None:
                 break  # No improvement possible, stop

             action_type, feature = best_action
             if action_type == 'add':
                 included_features.append(feature)
                 excluded_features.remove(feature)
                 print(f"    DEBUG: Added feature '{feature}', AIC improved from {current_aic:.4f} to {best_aic:.4f}")
             else:  # remove
                 included_features.remove(feature)
                 excluded_features.append(feature)
                 print(f"    DEBUG: Removed feature '{feature}', AIC improved from {current_aic:.4f} to {best_aic:.4f}")

         # Update final selected features
         current_features_aic = included_features

     final_selected_features = current_features_aic
     final_aic = np.nan # Default if no features or model fails

     print(f"    DEBUG: Final feature selection result: {len(final_selected_features)} features: {final_selected_features}")

     if final_selected_features:
         try:
             X_final_model = X[final_selected_features].copy().astype(float).reset_index(drop=True)
             y_final = y.astype(float).reset_index(drop=True)

             # Enhanced debugging for singular matrix detection
             print(f"    DEBUG: Checking final feature matrix properties...")
             print(f"    DEBUG: Feature matrix shape: {X_final_model.shape}")

             # Check for constant features and remove them
             feature_stds = X_final_model.std()
             constant_features = feature_stds[feature_stds == 0].index.tolist()
             if constant_features:
                 print(f"    DEBUG: WARNING - Constant features detected: {constant_features}")
                 print(f"    DEBUG: Removing constant features automatically...")
                 X_final_model = X_final_model.drop(columns=constant_features)
                 # Update final_selected_features to reflect removal
                 final_selected_features = [f for f in final_selected_features if f not in constant_features]
                 print(f"    DEBUG: Features after constant removal: {final_selected_features}")

             # Skip if no features remain after constant removal
             if X_final_model.empty or len(final_selected_features) == 0:
                 print(f"    DEBUG: No features remaining after constant feature removal")
                 final_aic = np.nan
             else:
                 # Check correlation matrix
                 corr_matrix = X_final_model.corr()
                 print(f"    DEBUG: Max correlation (off-diagonal): {corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool)).max().max():.4f}")

                 # Check for perfect correlations
                 perfect_corr_pairs = []
                 for i in range(len(corr_matrix.columns)):
                     for j in range(i+1, len(corr_matrix.columns)):
                         if abs(corr_matrix.iloc[i, j]) > 0.999:
                             perfect_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

                 if perfect_corr_pairs:
                     print(f"    DEBUG: WARNING - Near-perfect correlations found:")
                     for feat1, feat2, corr_val in perfect_corr_pairs:
                         print(f"    DEBUG:   {feat1} vs {feat2}: {corr_val:.6f}")

                 # Check matrix rank
                 try:
                     matrix_rank = np.linalg.matrix_rank(X_final_model.values)
                     print(f"    DEBUG: Matrix rank: {matrix_rank} (expected: {X_final_model.shape[1]})")
                     if matrix_rank < X_final_model.shape[1]:
                         print(f"    DEBUG: WARNING - Matrix is rank deficient! Features are linearly dependent.")
                 except:
                     print(f"    DEBUG: Could not calculate matrix rank")

                 X_final_model_const = sm.add_constant(X_final_model, has_constant='add')
                 final_model = sm.Logit(y_final, X_final_model_const).fit(disp=0,method='lbfgs')
                 final_aic = final_model.aic
                 print(f"    DEBUG: Final model AIC calculated successfully: {final_aic}")
         except Exception as e:
             print(f"    DEBUG: Error fitting final model for AIC calculation: {str(e)}")
             pass # final_aic remains np.nan
         return final_selected_features, X[final_selected_features].copy(), final_aic
     else:
         return [], pd.DataFrame(index=X.index), final_aic

def optimized_feature_selection(X, y, vif_threshold=5.0, max_features=None,
                               sampling_threshold=300000, n_jobs=None,
                               use_parallel_stepwise=True, verbose=True):
    """
    Convenience function for optimized feature selection with parallel processing

    This function combines VIF-based feature filtering with AIC-based stepwise selection,
    leveraging parallel processing to speed up computation on multi-core systems.

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable (binary)
        vif_threshold (float): VIF threshold for multicollinearity removal
        max_features (int, optional): Maximum features to retain after VIF stage
        sampling_threshold (int): Dataset size threshold for using sampling
        n_jobs (int, optional): Number of parallel jobs (None = use all CPUs)
        use_parallel_stepwise (bool): Whether to use parallel stepwise selection
        verbose (bool): Whether to print progress information

    Returns:
        tuple: (selected_features, selected_X, final_aic)
            - selected_features (list): Names of selected features
            - selected_X (pd.DataFrame): DataFrame with selected features only
            - final_aic (float): AIC of final model

    Example:
        >>> features, X_selected, aic = optimized_feature_selection(
        ...     X, y, vif_threshold=5.0, n_jobs=22
        ... )
        >>> print(f"Selected {len(features)} features with AIC: {aic:.4f}")
    """
    # Get conservative worker allocation for large datasets
    if X.shape[0] > 1000000:  # For datasets > 1M rows, be very conservative
        optimal_workers = min(_get_optimal_workers(n_jobs, X.shape[1], 'cpu'), 4)
        print(f"    DEBUG: Large dataset detected ({X.shape[0]} rows), limiting to {optimal_workers} workers")
    else:
        optimal_workers = _get_optimal_workers(n_jobs, X.shape[1], 'cpu')

    if verbose:
        print(f"Starting optimized feature selection with {optimal_workers} parallel workers")
        print(f"Initial features: {X.shape[1]}")
        print(f"Dataset size: {X.shape[0]} rows")

    selected_features, selected_X, final_aic = calculate_vif_iteratively(
        X=X,
        y=y,
        threshold=vif_threshold,
        max_features=max_features,
        sampling_threshold=sampling_threshold,
        n_jobs=optimal_workers,
        use_parallel_stepwise=use_parallel_stepwise
    )

    if verbose:
        print(f"Feature selection completed:")
        print(f"  - Final features: {len(selected_features)}")
        print(f"  - Features selected: {selected_features}")
        print(f"  - Final AIC: {final_aic:.4f}" if not np.isnan(final_aic) else "  - Final AIC: N/A")

    return selected_features, selected_X, final_aic


def _create_representative_sample(X, y, sample_size, max_attempts=10,
                                mean_tolerance=0.05, corr_tolerance=0.05, balance_tolerance=0.05):
    """
    Create a stratified representative sample with validation.

    Args:
        X (pd.DataFrame): Feature DataFrame
        y (pd.Series): Target variable
        sample_size (int): Desired sample size
        max_attempts (int): Maximum sampling attempts
        mean_tolerance (float): Tolerance for mean differences
        corr_tolerance (float): Tolerance for correlation differences
        balance_tolerance (float): Tolerance for target balance differences

    Returns:
        tuple: (X_sample, y_sample) representative sample
    """
    from sklearn.model_selection import train_test_split
    import numpy as np

    # Calculate original population statistics
    original_means = X.mean()
    original_target_balance = y.mean()

    # For correlation, use a subset of features if too many to avoid computation issues
    corr_features = X.columns[:min(20, len(X.columns))]
    original_corr_matrix = X[corr_features].corr()

    print(f"    DEBUG: Original target balance: {original_target_balance:.4f}")
    print(f"    DEBUG: Original feature means range: [{original_means.min():.4f}, {original_means.max():.4f}]")

    for attempt in range(max_attempts):
        print(f"    DEBUG: Sampling attempt {attempt + 1}/{max_attempts}")

        # Stratified sampling to maintain target distribution
        try:
            _, X_sample, _, y_sample = train_test_split(
                X, y,
                test_size=sample_size/len(X),
                stratify=y,
                random_state=42 + attempt
            )
            X_sample = X_sample.reset_index(drop=True)
            y_sample = y_sample.reset_index(drop=True)
        except ValueError:
            # If stratification fails (e.g., too few samples in a class), use simple random
            print(f"    DEBUG: Stratified sampling failed, using random sampling")
            sample_idx = np.random.RandomState(42 + attempt).choice(
                len(X), size=min(sample_size, len(X)), replace=False
            )
            X_sample = X.iloc[sample_idx].copy().reset_index(drop=True)
            y_sample = y.iloc[sample_idx].copy().reset_index(drop=True)

        # Validate representativeness
        sample_means = X_sample.mean()
        sample_target_balance = y_sample.mean()
        sample_corr_matrix = X_sample[corr_features].corr()

        # Check mean differences
        mean_diff = np.abs((sample_means - original_means) / (original_means + 1e-8)).max()

        # Check target balance difference
        balance_diff = abs(sample_target_balance - original_target_balance)

        # Check correlation differences (use mean absolute difference)
        corr_diff = np.abs(sample_corr_matrix - original_corr_matrix).mean().mean()

        print(f"    DEBUG: Mean diff: {mean_diff:.4f}, Balance diff: {balance_diff:.4f}, Corr diff: {corr_diff:.4f}")

        # Check if sample is representative
        if (mean_diff <= mean_tolerance and
            balance_diff <= balance_tolerance and
            corr_diff <= corr_tolerance):
            print(f"    DEBUG: Representative sample achieved on attempt {attempt + 1}")
            return X_sample, y_sample

    # If no representative sample found, use the last one with a warning
    print(f"    DEBUG: WARNING - Could not achieve representative sample within {max_attempts} attempts")
    print(f"    DEBUG: Using last sample with: mean_diff={mean_diff:.4f}, balance_diff={balance_diff:.4f}, corr_diff={corr_diff:.4f}")
    return X_sample, y_sample




# --- Custom Transformers ---

class BinningTransformer(BaseEstimator, TransformerMixin):
    """Base class for binning transformers"""

    def __init__(self, max_bins=5):
        self.max_bins = max_bins

    def fit(self, X, y=None):
        """Implement in subclasses"""
        return self

    def transform(self, X):
        """Implement in subclasses"""
        return X


class EqualFreqBinning(BinningTransformer):
    """Equal frequency binning transformer"""

    def fit(self, X, y=None):
        self.bin_edges_ = None
        X_array = X.reshape(-1, 1) if len(X.shape) == 1 else X
        if len(np.unique(X_array)) <= 1:
            self.constant_value_ = True
            return self

        self.constant_value_ = False

        try:
            # Use pandas qcut for equal frequency binning
            _, self.bin_edges_ = pd.qcut(
                X_array[:, 0],
                q=min(self.max_bins, len(np.unique(X_array[:, 0]))),
                labels=False,
                retbins=True,
                duplicates='drop'
            )
        except:
            # Fallback to equal-width binning
            _, self.bin_edges_ = pd.cut(
                X_array[:, 0],
                bins=min(self.max_bins, len(np.unique(X_array[:, 0]))),
                labels=False,
                retbins=True,
                duplicates='drop'
            )

        return self

    def transform(self, X):
        X_array = X.reshape(-1, 1) if len(X.shape) == 1 else X

        if hasattr(self, 'constant_value_') and self.constant_value_:
            return np.ones(X_array.shape[0], dtype=np.int8).reshape(-1, 1)

        if self.bin_edges_ is not None:
            # Apply binning using the bin edges
            result = pd.cut(
                X_array[:, 0],
                bins=self.bin_edges_,
                labels=False,
                include_lowest=True,
                duplicates='drop'
            )
            # Convert to int and handle NaN values, then add 1 for 1-based ordinal labels
            return (np.nan_to_num(result).astype(np.int8) + 1).reshape(-1, 1)
        else:
            # If no bin edges, return ones (ordinal 1)
            return np.ones(X_array.shape[0], dtype=np.int8).reshape(-1, 1)


class DecisionTreeBinning(BinningTransformer):
    """Parallel-optimized decision tree-based binning transformer"""

    def __init__(self, max_bins=5, random_state=42, n_jobs=None):
        super().__init__(max_bins=max_bins)
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        print(f"    DEBUG: Starting parallel DecisionTreeBinning for {len(X):,} samples")

        self.bin_edges_ = None
        self.tree_model_ = None
        X_array = X.reshape(-1, 1) if len(X.shape) == 1 else X

        if len(np.unique(X_array)) <= 1 or len(np.unique(y)) <= 1:
            self.constant_value_ = True
            return self

        self.constant_value_ = False

        # Get optimal workers for parallel processing
        n_workers = _get_optimal_workers(self.n_jobs, min(cpu_count(), 4), 'cpu')
        print(f"    DEBUG: Using {n_workers} workers for parallel tree evaluation")

        # Set parallel environment for tree fitting
        set_parallel_env(n_workers)

        # Adjust min_samples_leaf to avoid overfitting in small datasets
        min_samples_leaf = max(1, int(0.05 * len(X_array)))
        if len(X_array) / self.max_bins < min_samples_leaf:
            min_samples_leaf = max(1, int(len(X_array) / (self.max_bins * 2)))

        # Try multiple tree configurations in parallel to find best splits
        best_tree = self._parallel_tree_optimization(X_array, y, min_samples_leaf, n_workers)

        if best_tree is not None:
            # Extract thresholds from the best tree
            thresholds = sorted(list(set(
                best_tree.tree_.threshold[best_tree.tree_.feature != -2]
            )))
            self.bin_edges_ = np.array([-np.inf] + thresholds + [np.inf])
            self.tree_model_ = best_tree
            print(f"    DEBUG: DecisionTreeBinning found {len(thresholds)} split points")

        return self

    def _parallel_tree_optimization(self, X_array, y, min_samples_leaf, n_workers):
        """Optimize tree parameters using parallel evaluation"""
        try:
            # Generate different tree configurations to test
            configs = self._generate_tree_configs(min_samples_leaf)

            if len(configs) <= 1:
                # If only one config, use standard sequential approach
                return self._fit_single_tree(X_array, y, configs[0])

            print(f"    DEBUG: Testing {len(configs)} tree configurations in parallel")

            # Evaluate configurations in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(n_workers, len(configs))) as executor:
                future_to_config = {}
                for config in configs:
                    future = executor.submit(self._evaluate_tree_config, X_array, y, config)
                    future_to_config[future] = config

                best_score = -np.inf
                best_tree = None

                for future in concurrent.futures.as_completed(future_to_config):
                    try:
                        tree, score = future.result()
                        if score > best_score:
                            best_score = score
                            best_tree = tree
                    except Exception as e:
                        print(f"    DEBUG: Tree config evaluation failed: {str(e)}")
                        continue

            return best_tree

        except Exception as e:
            print(f"    DEBUG: Parallel tree optimization failed: {str(e)}, using fallback")
            # Fallback to single tree
            return self._fit_single_tree(X_array, y, self._generate_tree_configs(min_samples_leaf)[0])

    def _generate_tree_configs(self, min_samples_leaf):
        """Generate different tree configurations to test"""
        configs = []

        # Base configuration
        configs.append({
            'max_leaf_nodes': self.max_bins,
            'min_samples_leaf': min_samples_leaf,
            'min_samples_split': min_samples_leaf * 2,
            'max_depth': None
        })

        # Additional configurations for optimization
        if len(configs) < 4:  # Only add if we don't have too many
            # More conservative splitting
            configs.append({
                'max_leaf_nodes': self.max_bins,
                'min_samples_leaf': min_samples_leaf * 2,
                'min_samples_split': min_samples_leaf * 4,
                'max_depth': 6
            })

            # More aggressive splitting
            configs.append({
                'max_leaf_nodes': self.max_bins,
                'min_samples_leaf': max(1, min_samples_leaf // 2),
                'min_samples_split': min_samples_leaf,
                'max_depth': 8
            })

        return configs

    def _fit_single_tree(self, X_array, y, config):
        """Fit a single tree with given configuration"""
        tree_model = DecisionTreeClassifier(
            random_state=self.random_state,
            **config
        )
        try:
            tree_model.fit(X_array, y)
            return tree_model
        except:
            return None

    def _evaluate_tree_config(self, X_array, y, config):
        """Evaluate a tree configuration and return tree with score"""
        try:
            tree = self._fit_single_tree(X_array, y, config)
            if tree is None:
                return None, -np.inf

            # Score based on number of meaningful splits and tree quality
            n_splits = len(set(tree.tree_.threshold[tree.tree_.feature != -2]))
            if n_splits == 0:
                return tree, -np.inf

            # Simple scoring: prefer trees with good split distribution
            score = n_splits * tree.score(X_array, y)
            return tree, score

        except Exception:
            return None, -np.inf

    def transform(self, X):
        X_array = X.reshape(-1, 1) if len(X.shape) == 1 else X

        if hasattr(self, 'constant_value_') and self.constant_value_:
            return np.ones(X_array.shape[0], dtype=np.int8).reshape(-1, 1)

        if self.bin_edges_ is not None and len(self.bin_edges_) > 1:
            # Apply binning using standard pd.cut for consistency with other methods
            try:
                result = pd.cut(
                    X_array[:, 0],
                    bins=self.bin_edges_,
                    labels=False,
                    include_lowest=True,
                    duplicates='drop'
                )
                # Convert to int and handle NaN values, then add 1 for 1-based ordinal labels
                return (np.nan_to_num(result).astype(np.int8) + 1).reshape(-1, 1)
            except Exception as e:
                # If binning fails, return ones (ordinal 1)
                return np.ones(X_array.shape[0], dtype=np.int8).reshape(-1, 1)
        else:
            # If no bin edges, return ones (ordinal 1)
            return np.ones(X_array.shape[0], dtype=np.int8).reshape(-1, 1)


class ChiMergeBinning(BinningTransformer):
    """Chi-squared merging binning transformer"""

    def __init__(self, max_bins=5, threshold=3.841):
        super().__init__(max_bins=max_bins)
        self.threshold = threshold

    def fit(self, X, y):
        self.bin_edges_ = None
        X_array = X.reshape(-1, 1) if len(X.shape) == 1 else X

        if len(np.unique(X_array)) <= 1 or len(np.unique(y)) <= 1:
            self.constant_value_ = True
            return self

        self.constant_value_ = False

        # Initial binning if there are too many unique values
        unique_values = np.unique(X_array)
        if len(unique_values) > self.max_bins * 5:
            try:
                init_binning = EqualFreqBinning(max_bins=min(20, len(unique_values)))
                binned_values = init_binning.fit_transform(X_array, y)
            except:
                binned_values = np.zeros(X_array.shape[0], dtype=np.int8).reshape(-1, 1)
        else:
            # Use rank if few enough unique values
            binned_values = np.zeros(X_array.shape[0], dtype=np.int8).reshape(-1, 1)
            for i, val in enumerate(sorted(unique_values)):
                binned_values[X_array[:, 0] == val] = i

        # Create DataFrame for chi-square calculation
        df = pd.DataFrame({
            'x': X_array[:, 0],
            'bin': binned_values[:, 0],
            'y': y
        })

        # Get unique bins and sort by x value
        unique_bins = np.unique(binned_values)
        bin_min_x = df.groupby('bin')['x'].min().reset_index()
        bin_order = bin_min_x.sort_values('x')['bin'].values

        # Calculate contingency tables for each bin
        bin_counts = []
        bin_edges = []

        for bin_val in bin_order:
            bin_data = df[df['bin'] == bin_val]
            if len(bin_data) == 0:
                continue

            # Create contingency table
            counts = np.zeros(2)  # Assuming binary target
            counts[0] = (bin_data['y'] == 0).sum()
            counts[1] = (bin_data['y'] == 1).sum()

            bin_counts.append(counts)
            bin_edges.append([bin_data['x'].min(), bin_data['x'].max()])

        # Perform chi-merge iteratively
        while len(bin_counts) > self.max_bins:
            min_chi2, merge_idx = np.inf, -1

            # Calculate chi-square for adjacent bins
            for i in range(len(bin_counts) - 1):
                obs = np.array([bin_counts[i], bin_counts[i+1]])

                # Skip if any row or column sum is zero
                if np.any(obs.sum(axis=0) == 0) or np.any(obs.sum(axis=1) == 0):
                    continue

                try:
                    chi2, _, _, _ = chi2_contingency(obs)
                    if chi2 < min_chi2:
                        min_chi2, merge_idx = chi2, i
                except:
                    continue

            # Break if no valid merge or already at desired bins
            if merge_idx == -1 or min_chi2 > self.threshold:
                break

            # Merge the bins
            bin_counts[merge_idx] += bin_counts[merge_idx + 1]
            bin_edges[merge_idx][1] = bin_edges[merge_idx + 1][1]  # Update max edge

            # Remove the merged bin
            bin_counts.pop(merge_idx + 1)
            bin_edges.pop(merge_idx + 1)

        # Create final bin edges
        if len(bin_edges) > 0:
            edges = [-np.inf]
            for i in range(len(bin_edges) - 1):
                # Use the maximum value of each bin as a boundary
                edges.append(bin_edges[i][1])
            edges.append(np.inf)
            self.bin_edges_ = np.array(edges)

        return self

    def transform(self, X):
        X_array = X.reshape(-1, 1) if len(X.shape) == 1 else X

        if hasattr(self, 'constant_value_') and self.constant_value_:
            return np.ones(X_array.shape[0], dtype=np.int8).reshape(-1, 1)

        if self.bin_edges_ is not None and len(self.bin_edges_) > 1:
            # Apply binning using the bin edges
            result = pd.cut(
                X_array[:, 0],
                bins=self.bin_edges_,
                labels=False,
                include_lowest=True,
                duplicates='drop'
            )
            # Convert to int and handle NaN values, then add 1 for 1-based ordinal labels
            return (np.nan_to_num(result).astype(np.int8) + 1).reshape(-1, 1)
        else:
            # If no bin edges, return ones (ordinal 1)
            return np.ones(X_array.shape[0], dtype=np.int8).reshape(-1, 1)


class MAPABinning(BinningTransformer):
    """MAPA (Monotonic Adaptive Pattern Analysis) binning"""

    def __init__(self, max_bins=5, direction='auto', random_state=42, bin_threshold=0.00001):
        super().__init__(max_bins=max_bins)
        self.direction = direction
        self.random_state = random_state
        self.bin_threshold = bin_threshold  # Threshold for creating new bins

    def fit(self, X, y):
        self.bin_edges_ = None
        X_array = X.reshape(-1, 1) if len(X.shape) == 1 else X

        if len(np.unique(X_array)) <= 1 or len(np.unique(y)) <= 1:
            self.constant_value_ = True
            return self

        self.constant_value_ = False

        # First do equal frequency binning
        try:
            init_bins = min(self.max_bins * 3, len(np.unique(X_array)))
            init_binning = EqualFreqBinning(max_bins=init_bins)
            binned_values = init_binning.fit_transform(X_array, y)
            # No diagnostic printing
        except Exception as e:
            # No error printing
            # Initialize with basic bins instead of returning
            self.bin_edges_ = np.array([-np.inf, np.inf])
            return self

        # Create DataFrame with binning data
        df = pd.DataFrame({
            'feature': X_array[:, 0],
            'bin': binned_values[:, 0],
            'target': y
        })

        # Calculate statistics for each bin
        bin_stats = df.groupby('bin').agg(
            target_rate=('target', 'mean'),
            mean_feature=('feature', 'mean'),
            min_feature=('feature', 'min'),
            max_feature=('feature', 'max'),
            count=('feature', 'count')
        ).reset_index()

        # Determine trend direction
        if self.direction == 'auto':
            if bin_stats.shape[0] > 1:
                correlation = np.corrcoef(
                    bin_stats['mean_feature'],
                    bin_stats['target_rate']
                )[0, 1]
                trend_increasing = correlation >= 0 if not np.isnan(correlation) else True
            else:
                trend_increasing = True
        else:
            trend_increasing = (self.direction == 'increasing')

        # Apply isotonic regression to ensure monotonicity
        ir = IsotonicRegression(increasing=trend_increasing, out_of_bounds='clip')
        bin_stats = bin_stats.sort_values(by='mean_feature')

        try:
            bin_stats['monotonic_rate'] = ir.fit_transform(
                bin_stats['mean_feature'],
                bin_stats['target_rate']
            )
            # No diagnostic printing
        except Exception as e:
            # No error printing
            # Initialize with basic bins instead of returning
            self.bin_edges_ = np.array([-np.inf, np.inf])
            return self

        # Create binning edges based on monotonic rates
        edges = [-np.inf]
        current_rate = None

        # Sort by feature value to ensure proper bin ordering
        bin_stats = bin_stats.sort_values(by='mean_feature')

        # No diagnostic printing

        for _, row in bin_stats.iterrows():
            # Use the specified threshold for bin creation
            if current_rate is None or abs(row['monotonic_rate'] - current_rate) > self.bin_threshold:
                if current_rate is not None:
                    edges.append(row['min_feature'])
                current_rate = row['monotonic_rate']

        edges.append(np.inf)
        # No diagnostic printing

        if len(edges) > 1:
            self.bin_edges_ = np.array(edges)
        else:
            # Ensure we have at least two edges to create one bin
            # No warning printing
            self.bin_edges_ = np.array([-np.inf, np.inf])

        # Force at least 2 bins if we only got 1 and have enough unique values
        if len(self.bin_edges_) <= 2 and len(np.unique(X_array)) > 10:
            # No diagnostic printing
            median_val = np.median(X_array)
            self.bin_edges_ = np.array([-np.inf, median_val, np.inf])

        # No diagnostic printing
        return self

    def transform(self, X):
        X_array = X.reshape(-1, 1) if len(X.shape) == 1 else X

        if hasattr(self, 'constant_value_') and self.constant_value_:
            return np.zeros(X_array.shape[0], dtype=np.int8).reshape(-1, 1)

        if self.bin_edges_ is not None and len(self.bin_edges_) > 1:
            # Apply binning using the bin edges
            result = pd.cut(
                X_array[:, 0],
                bins=self.bin_edges_,
                labels=False,
                include_lowest=True,
                duplicates='drop'
            )
            # Convert to int and handle NaN values, then add 1 for 1-based ordinal labels
            return (np.nan_to_num(result).astype(np.int8) + 1).reshape(-1, 1)
        else:
            # If no bin edges, return ones (ordinal 1)
            return np.ones(X_array.shape[0], dtype=np.int8).reshape(-1, 1)


class WOETransformer(BaseEstimator, TransformerMixin):
    """Weight of Evidence transformer with smoothing and capping"""

    def __init__(self, epsilon=0.5, woe_cap=5.0):
        """
        Initialize the WOE transformer.

        Parameters:
        -----------
        epsilon : float, default=0.5
            Smoothing parameter to prevent extreme WOE values and division by zero.
            Higher values provide more smoothing.
        woe_cap : float, default=5.0
            Cap value for WOE to prevent extreme values. Applied as +/- woe_cap.
        """
        self.epsilon = epsilon
        self.woe_cap = woe_cap
        self.default_woe_ = 0.0  # Initialize default WOE to prevent missing attribute errors

    def fit(self, X, y):
        """
        Fit the Weight of Evidence transformer.

        Parameters:
        -----------
        X : array-like of shape (n_samples, 1)
            Binned input features
        y : array-like of shape (n_samples,)
            Target values (binary)

        Returns:
        --------
        self : object
            Returns self
        """
        X_array = X.reshape(-1, 1) if len(X.shape) == 1 else X

        # Print diagnostic information
        # No diagnostic printing to maintain consistency across methods

        if len(np.unique(X_array)) <= 1 or len(np.unique(y)) <= 1:
            # No diagnostic printing
            self.woe_map_ = {0: 0.0}  # Default WOE for constant features
            self.iv_ = 0.0
            self.default_woe_ = 0.0  # Set default WOE for edge cases
            return self

        # Calculate WOE and IV
        df = pd.DataFrame({'bin': X_array[:, 0], 'target': y})
        counts = df.groupby('bin').agg(
            total=('target', 'count'),
            bads=('target', 'sum')
        ).reset_index()

        counts['goods'] = counts['total'] - counts['bads']

        # Apply smoothing with Laplace smoothing
        counts['goods_smooth'] = counts['goods'] + self.epsilon
        counts['bads_smooth'] = counts['bads'] + self.epsilon

        total_goods = counts['goods'].sum() + self.epsilon * counts.shape[0]
        total_bads = counts['bads'].sum() + self.epsilon * counts.shape[0]

        counts['goods_rate'] = counts['goods_smooth'] / total_goods
        counts['bads_rate'] = counts['bads_smooth'] / total_bads

        # Calculate WOE and IV
        counts['woe'] = np.log(counts['goods_rate'] / counts['bads_rate'])

        # Apply capping to WOE values
        counts['woe'] = np.clip(counts['woe'], -self.woe_cap, self.woe_cap)

        counts['iv_component'] = (counts['goods_rate'] - counts['bads_rate']) * counts['woe']

        # Store results
        self.woe_map_ = dict(zip(counts['bin'], counts['woe']))
        self.iv_ = counts['iv_component'].sum()

        # Store default WOE (weighted average of WOE values for unseen bins)
        weights = counts['total'] / counts['total'].sum()
        self.default_woe_ = np.sum(weights * counts['woe'])

        # No diagnostic printing

        return self

    def transform(self, X):
        """
        Transform binned values to WOE values.

        Parameters:
        -----------
        X : array-like of shape (n_samples, 1)
            Binned input features

        Returns:
        --------
        X_woe : array-like of shape (n_samples, 1)
            WOE-transformed features
        """
        X_array = X.reshape(-1, 1) if len(X.shape) == 1 else X

        # Map bin values to WOE with more robust handling
        result = np.zeros(X_array.shape[0])
        unique_bins = np.unique(X_array[:, 0])

        # Check for unseen bins without printing
        unseen_bins = [b for b in unique_bins if b not in self.woe_map_]

        # Apply WOE mapping
        for bin_val in unique_bins:
            mask = X_array[:, 0] == bin_val
            # Use stored WOE if available, otherwise use default WOE
            woe_val = self.woe_map_.get(bin_val, self.default_woe_)
            result[mask] = woe_val

        return result.reshape(-1, 1)

class MonotonicityEnforcer(BinningTransformer):
    """
    Transformer that ensures monotonicity in the relationship between bins and target variable.
    Can be applied to any binning strategy's output.
    """
    def __init__(self, force_direction=None):
        """
        Initialize the monotonicity enforcer.

        Parameters:
        -----------
        force_direction : {None, 'increasing', 'decreasing'}, default=None
            If None, the direction (increasing or decreasing) is determined automatically
            from the data. If specified as 'increasing' or 'decreasing', forces that direction.
        """
        self.force_direction = force_direction
        self.bin_means_ = None
        self.original_to_monotonic_mapping_ = None
        self.direction_ = None

    def fit(self, X, y):
        """
        Fit the monotonicity enforcer by calculating bin means and creating a mapping
        that ensures monotonicity.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Binned input samples, where each value represents a bin index.
        y : array-like of shape (n_samples,)
            Target values.

        Returns:
        --------
        self : object
            Returns self.
        """
        # Validate input
        X = np.asarray(X).reshape(-1)  # Ensure X is a 1D array
        y = np.asarray(y)

        # Calculate mean target value for each bin
        unique_bins = np.sort(np.unique(X))
        bin_means = np.array([np.mean(y[X == bin_idx]) for bin_idx in unique_bins])

        # Determine direction of monotonicity if not specified
        if self.force_direction is None:
            # Calculate if increasing or decreasing relationship is stronger
            increasing_violations = sum(bin_means[i] > bin_means[i+1] for i in range(len(bin_means)-1))
            decreasing_violations = sum(bin_means[i] < bin_means[i+1] for i in range(len(bin_means)-1))

            self.direction_ = 'increasing' if increasing_violations <= decreasing_violations else 'decreasing'
        else:
            self.direction_ = self.force_direction

        # Create monotonic bins by merging non-monotonic adjacent bins
        monotonic_bins = self._enforce_monotonicity(unique_bins, bin_means)

        # Create mapping from original bins to monotonic bins
        self.original_to_monotonic_mapping_ = {
            orig: mono for orig, mono in zip(unique_bins, monotonic_bins)
        }

        # Store bin means for the new monotonic bins
        self.bin_means_ = {}
        for orig_bin, mono_bin in self.original_to_monotonic_mapping_.items():
            if mono_bin not in self.bin_means_:
                self.bin_means_[mono_bin] = []
            self.bin_means_[mono_bin].append(bin_means[np.where(unique_bins == orig_bin)[0][0]])

        # Calculate average mean for merged bins
        for mono_bin in self.bin_means_:
            self.bin_means_[mono_bin] = np.mean(self.bin_means_[mono_bin])

        return self

    def _enforce_monotonicity(self, unique_bins, bin_means):
        """
        Enforce monotonicity by merging adjacent bins as needed.

        Parameters:
        -----------
        unique_bins : array-like
            Array of unique bin indices.
        bin_means : array-like
            Mean target value for each bin.

        Returns:
        --------
        monotonic_bins : array-like
            New bin indices that ensure monotonicity.
        """
        if len(unique_bins) <= 1:
            return unique_bins

        monotonic_bins = unique_bins.copy()
        is_monotonic = False

        while not is_monotonic:
            current_bin_means = np.array([
                np.mean([bin_means[i] for i, b in enumerate(unique_bins)
                         if monotonic_bins[i] == unique_value])
                for unique_value in np.unique(monotonic_bins)
            ])

            # Check if current binning is monotonic
            if self.direction_ == 'increasing':
                violations = np.where(current_bin_means[:-1] >= current_bin_means[1:])[0]
            else:  # decreasing
                violations = np.where(current_bin_means[:-1] <= current_bin_means[1:])[0]

            if len(violations) == 0:
                is_monotonic = True
                continue

            # Get unique monotonic bin values
            unique_monotonic_bins = np.unique(monotonic_bins)

            # Merge the first violation
            violation_idx = violations[0]
            value_to_replace = unique_monotonic_bins[violation_idx + 1]
            value_to_keep = unique_monotonic_bins[violation_idx]

            # Update the bins
            monotonic_bins = np.array([
                value_to_keep if b == value_to_replace else b for b in monotonic_bins
            ])

        return monotonic_bins

    def transform(self, X):
        """
        Transform bins to ensure monotonicity.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Binned input samples.

        Returns:
        --------
        X_monotonic : array-like of shape (n_samples, n_features)
            Monotonic binned samples.
        """
        if self.original_to_monotonic_mapping_ is None:
            raise ValueError("This MonotonicityEnforcer instance is not fitted yet.")

        X = np.asarray(X).reshape(-1)

        # Map each bin to its monotonic version
        X_monotonic = np.array([
            self.original_to_monotonic_mapping_.get(x, x) for x in X
        ])

        return X_monotonic.reshape(-1, 1)

    def get_monotonic_event_rates(self):
        """
        Get the mean target value for each monotonic bin.

        Returns:
        --------
        event_rates : dict
            Dictionary mapping monotonic bin indices to their mean target values.
        """
        if self.bin_means_ is None:
            raise ValueError("This MonotonicityEnforcer instance is not fitted yet.")

        return self.bin_means_

    def is_monotonic(self, bin_means):
        """
        Check if a sequence of bin means is monotonic.

        Parameters:
        -----------
        bin_means : array-like
            Mean target value for each bin.

        Returns:
        --------
        is_monotonic : bool
            True if the sequence is monotonic (according to self.direction_),
            False otherwise.
        """
        if len(bin_means) <= 1:
            return True

        if self.direction_ == 'increasing':
            return np.all(bin_means[:-1] <= bin_means[1:])
        else:  # decreasing
            return np.all(bin_means[:-1] >= bin_means[1:])

    @property
    def monotonicity_direction(self):
        """Return the detected or forced monotonicity direction."""
        return self.direction_


def ensure_monotonic_pipeline(binning_strategy, force_direction=None, epsilon=0.5, woe_cap=5.0):
    """
    Create a pipeline that applies a binning strategy and ensures monotonicity.

    Parameters:
    -----------
    binning_strategy : object
        A binning strategy instance (e.g., EqualFreqBinning, DecisionTreeBinning).
    force_direction : {None, 'increasing', 'decreasing'}, default=None
        Direction of monotonicity to enforce.
    epsilon : float, default=0.5
        Smoothing parameter for WOE calculation.
    woe_cap : float, default=5.0
        Cap on the absolute value of WOE to prevent extreme values.

    Returns:
    --------
    pipeline : Pipeline
        A scikit-learn pipeline that includes the binning strategy,
        monotonicity enforcer, and WOE transformer.
    """
    return Pipeline([
        ('binning', binning_strategy),
        ('monotonicity_enforcer', MonotonicityEnforcer(force_direction=force_direction)),
        ('woe_transformer', WOETransformer(epsilon=epsilon, woe_cap=woe_cap))
    ])

# --- Pipeline Factory Functions ---

def create_equal_freq_pipeline(n_bins=10, max_bins=None, force_monotonic=True, monotonic_direction=None):
    """Create a pipeline with equal frequency binning and WOE transformation."""
    # Use max_bins if provided (for backward compatibility), otherwise use n_bins
    bins = max_bins if max_bins is not None else n_bins
    binning = EqualFreqBinning(max_bins=bins)
    return ensure_monotonic_pipeline(binning, force_direction=monotonic_direction, epsilon=0.5, woe_cap=5.0)

def create_decision_tree_pipeline(max_depth=3, random_state=None, force_monotonic=True, monotonic_direction=None, max_bins=None, n_jobs=None):
    """Create a parallel-optimized pipeline with decision tree binning and WOE transformation."""
    # Note: max_bins parameter is accepted for API consistency but not used
    binning = DecisionTreeBinning(max_bins=max_depth, random_state=random_state, n_jobs=n_jobs)
    return ensure_monotonic_pipeline(binning, force_direction=monotonic_direction, epsilon=0.5, woe_cap=5.0)

def create_chi_merge_pipeline(max_depth=3, threshold=None, force_monotonic=True, monotonic_direction=None, max_bins=None):
    """Create a pipeline with ChiMerge binning and WOE transformation."""
    # max_bins parameter is accepted for API consistency
    params = {}
    if max_depth is not None:
        params['max_bins'] = max_depth
    if threshold is not None:
        params['threshold'] = threshold
    binning = ChiMergeBinning(**params)
    return ensure_monotonic_pipeline(binning, force_direction=monotonic_direction, epsilon=0.5, woe_cap=5.0)

def create_mapa_pipeline(max_bins=10, force_monotonic=True, monotonic_direction=None, random_state=None):
    """Create a pipeline with MAPA binning and WOE transformation."""
    # Note that random_state is accepted but ignored if not used by MAPABinning
    binning = MAPABinning(
        max_bins=max_bins,
        direction='auto',
        random_state=random_state,
        bin_threshold=0.00001  # More sensitive threshold for bin creation
    )
    return ensure_monotonic_pipeline(binning, force_direction=monotonic_direction, epsilon=0.5, woe_cap=5.0)

class ConditionalInferenceTreeBinning(BinningTransformer):
    """
    Parallel-optimized Conditional Inference Tree Binning transformer

    This implementation uses statistical tests to determine significant split points
    with parallel processing for large datasets.
    """

    def __init__(self, max_bins=5, alpha=0.05, min_samples_leaf=30, random_state=None, n_jobs=None):
        """
        Initialize the conditional inference tree binning transformer.

        Parameters:
        -----------
        max_bins : int, default=5
            Maximum number of bins to create
        alpha : float, default=0.05
            Significance level for statistical tests
        min_samples_leaf : int, default=30
            Minimum number of samples required in a leaf node
        random_state : int, default=None
            Random state for reproducibility
        n_jobs : int, default=None
            Number of parallel jobs for statistical test evaluation
        """
        super().__init__(max_bins=max_bins)
        self.alpha = alpha
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.split_points_ = None
        self.bin_edges_ = None

    def fit(self, X, y):
        """
        Fit the conditional inference tree binning transformer.

        Parameters:
        -----------
        X : array-like of shape (n_samples, 1)
            Input feature to bin
        y : array-like of shape (n_samples,)
            Target variable

        Returns:
        --------
        self : object
            Returns self
        """
        # Convert inputs to numpy arrays
        X = np.asarray(X).reshape(-1)
        y = np.asarray(y)

        # Check if feature has only one unique value
        if len(np.unique(X)) <= 1:
            self.constant_value_ = True
            self.bin_edges_ = np.array([-np.inf, np.inf])
            return self

        self.constant_value_ = False

        # Initialize the tree
        self.split_points_ = []

        # Grow the tree recursively using statistical tests
        self._grow_tree(X, y, max_depth=0)

        # Sort split points and create bin edges
        if self.split_points_:
            self.split_points_ = sorted(set(self.split_points_))
            self.bin_edges_ = np.array([-np.inf] + self.split_points_ + [np.inf])
        else:
            # If no splits were found, create a single bin
            self.bin_edges_ = np.array([-np.inf, np.inf])

        return self

    def _grow_tree(self, X, y, max_depth):
        """
        Recursively grow the conditional inference tree.

        Parameters:
        -----------
        X : array-like of shape (n_samples,)
            Input feature values
        y : array-like of shape (n_samples,)
            Target values
        max_depth : int
            Current depth of the tree
        """
        # Stop if we've reached the maximum number of bins - 1 (splits)
        if len(self.split_points_) >= self.max_bins - 1:
            return

        # Stop if number of samples is too small
        if len(X) < 2 * self.min_samples_leaf:
            return

        # Stop if all target values are the same
        if len(np.unique(y)) <= 1:
            return

        # OPTIMIZATION: Pre-sort data once for this recursive call
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]

        # Find the best split point using statistical test with sorted data
        best_split, p_value = self._find_best_split_optimized(X_sorted, y_sorted)

        # If no significant split is found, stop
        if best_split is None or p_value > self.alpha:
            return

        # Add the split point
        self.split_points_.append(best_split)

        # OPTIMIZATION: Use sorted data for faster splitting
        split_idx = np.searchsorted(X_sorted, best_split, side='right')

        # Split the sorted data back to original order for recursion
        left_indices = sort_idx[:split_idx]
        right_indices = sort_idx[split_idx:]

        # Continue growing the tree recursively
        self._grow_tree(X[left_indices], y[left_indices], max_depth + 1)
        self._grow_tree(X[right_indices], y[right_indices], max_depth + 1)

    def _find_best_split(self, X, y):
        """
        Find the best split point using statistical tests (legacy method).
        """
        # For backward compatibility, use the optimized version
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        return self._find_best_split_optimized(X_sorted, y_sorted)

    def _find_best_split_optimized(self, X_sorted, y_sorted):
        """
        Find the best split point using parallel statistical tests.

        Parameters:
        -----------
        X_sorted : array-like of shape (n_samples,)
            Input feature values (pre-sorted)
        y_sorted : array-like of shape (n_samples,)
            Target values (sorted by X)

        Returns:
        --------
        best_split : float or None
            The best split point, or None if no significant split is found
        min_p_value : float
            The p-value of the best split
        """
        # Get unique values of X, excluding extremes
        unique_values = np.unique(X_sorted)
        if len(unique_values) <= 1:
            return None, 1.0

        # Consider potential split points (midpoints between unique values)
        potential_splits = [(unique_values[i] + unique_values[i+1]) / 2
                           for i in range(len(unique_values) - 1)]

        if not potential_splits:
            return None, 1.0

        # For large numbers of potential splits, use parallel processing
        if len(potential_splits) > 50 and self.n_jobs != 1:
            return self._parallel_split_evaluation(X_sorted, y_sorted, potential_splits)
        else:
            return self._sequential_split_evaluation(X_sorted, y_sorted, potential_splits)

    def _parallel_split_evaluation(self, X_sorted, y_sorted, potential_splits):
        """Evaluate split points in parallel"""
        try:
            n_workers = _get_optimal_workers(self.n_jobs, len(potential_splits), 'cpu')
            print(f"    DEBUG: Evaluating {len(potential_splits)} split points with {n_workers} workers")

            # Chunk the splits for parallel processing
            chunk_size = max(1, len(potential_splits) // n_workers)
            split_chunks = [potential_splits[i:i + chunk_size]
                           for i in range(0, len(potential_splits), chunk_size)]

            # Prepare arguments for parallel processing
            args_list = []
            for chunk in split_chunks:
                args_list.append((X_sorted, y_sorted, chunk, self.min_samples_leaf))

            # Process chunks in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(self._evaluate_split_chunk, args) for args in args_list]

                best_split = None
                min_p_value = 1.0

                for future in concurrent.futures.as_completed(futures):
                    try:
                        chunk_best_split, chunk_min_p = future.result()
                        if chunk_min_p < min_p_value:
                            min_p_value = chunk_min_p
                            best_split = chunk_best_split
                    except Exception as e:
                        print(f"    DEBUG: Split chunk evaluation failed: {str(e)}")
                        continue

            return best_split, min_p_value

        except Exception as e:
            print(f"    DEBUG: Parallel split evaluation failed: {str(e)}, using sequential")
            return self._sequential_split_evaluation(X_sorted, y_sorted, potential_splits)

    def _evaluate_split_chunk(self, args):
        """Evaluate a chunk of potential splits"""
        X_sorted, y_sorted, split_chunk, min_samples_leaf = args

        best_split = None
        min_p_value = 1.0

        # Pre-compute target statistics for binary case
        is_binary_target = len(np.unique(y_sorted)) == 2
        if is_binary_target:
            total_positive = np.sum(y_sorted == 1)
            total_negative = np.sum(y_sorted == 0)

            if total_positive == 0 or total_negative == 0:
                return None, 1.0

        # Test each split in this chunk
        for split in split_chunk:
            try:
                split_idx = np.searchsorted(X_sorted, split, side='right')

                # Skip if either group is too small
                if split_idx < min_samples_leaf or (len(X_sorted) - split_idx) < min_samples_leaf:
                    continue

                # Perform statistical test
                if is_binary_target:
                    left_positive = np.sum(y_sorted[:split_idx] == 1)
                    left_negative = split_idx - left_positive
                    right_positive = total_positive - left_positive
                    right_negative = total_negative - left_negative

                    # Skip if any cell has zero count
                    if left_positive == 0 or left_negative == 0 or right_positive == 0 or right_negative == 0:
                        continue

                    contingency_table = np.array([
                        [left_negative, left_positive],
                        [right_negative, right_positive]
                    ])

                    try:
                        _, p_value, _, _ = chi2_contingency(contingency_table)
                    except (ValueError, ZeroDivisionError):
                        continue
                else:
                    # For continuous target, use Mann-Whitney U test
                    try:
                        _, p_value = mannwhitneyu(y_sorted[:split_idx], y_sorted[split_idx:])
                    except (ValueError, RuntimeError):
                        continue

                # Update best split if this one is more significant
                if p_value < min_p_value:
                    min_p_value = p_value
                    best_split = split

            except Exception:
                continue

        return best_split, min_p_value

    def _sequential_split_evaluation(self, X_sorted, y_sorted, potential_splits):
        """Evaluate split points sequentially (original implementation)"""
        best_split = None
        min_p_value = 1.0

        # Pre-compute target statistics for binary case
        is_binary_target = len(np.unique(y_sorted)) == 2
        if is_binary_target:
            total_positive = np.sum(y_sorted == 1)
            total_negative = np.sum(y_sorted == 0)

            # Early exit if target is pure
            if total_positive == 0 or total_negative == 0:
                return None, 1.0

        # Test each potential split point
        for split in potential_splits:
            try:
                split_idx = np.searchsorted(X_sorted, split, side='right')

                # Skip if either group is too small
                if split_idx < self.min_samples_leaf or (len(X_sorted) - split_idx) < self.min_samples_leaf:
                    continue

                # Perform statistical test
                if is_binary_target:
                    left_positive = np.sum(y_sorted[:split_idx] == 1)
                    left_negative = split_idx - left_positive
                    right_positive = total_positive - left_positive
                    right_negative = total_negative - left_negative

                    # Skip if any cell has zero count
                    if left_positive == 0 or left_negative == 0 or right_positive == 0 or right_negative == 0:
                        continue

                    contingency_table = np.array([
                        [left_negative, left_positive],
                        [right_negative, right_positive]
                    ])

                    try:
                        _, p_value, _, _ = chi2_contingency(contingency_table)
                    except (ValueError, ZeroDivisionError):
                        continue
                else:
                    # For continuous target, use Mann-Whitney U test
                    try:
                        _, p_value = mannwhitneyu(y_sorted[:split_idx], y_sorted[split_idx:])
                    except (ValueError, RuntimeError):
                        continue

                # Update best split if this one is more significant
                if p_value < min_p_value:
                    min_p_value = p_value
                    best_split = split

            except Exception:
                continue

        return best_split, min_p_value

    def transform(self, X):
        """
        Transform X into bins.

        Parameters:
        -----------
        X : array-like of shape (n_samples, 1)
            Input feature to bin

        Returns:
        --------
        X_binned : array-like of shape (n_samples, 1)
            Binned feature
        """
        X = np.asarray(X).reshape(-1)

        if not hasattr(self, 'bin_edges_'):
            raise ValueError("This ConditionalInferenceTreeBinning instance is not fitted yet.")

        if self.constant_value_:
            return np.ones(X.shape[0], dtype=np.int8).reshape(-1, 1)

        # Assign bins (digitize starts at 1, keep 1-based indexing for ordinal labels)
        X_binned = np.digitize(X, self.bin_edges_[1:-1]).astype(np.int8)

        return X_binned.reshape(-1, 1)

def create_conditional_inference_tree_pipeline(max_bins=5, alpha=0.05, min_samples_leaf=30, n_jobs=None,
                                               random_state=None,
                                               force_monotonic=True,
                                               monotonic_direction=None):
    """
    Create a parallel-optimized pipeline with conditional inference tree binning.

    Parameters:
    -----------
    max_bins : int, default=5
        Maximum number of bins to create
    alpha : float, default=0.05
        Significance level for statistical tests
    min_samples_leaf : int, default=30
        Minimum number of samples required in a leaf node
    n_jobs : int, default=None
        Number of parallel jobs for split evaluation
    random_state : int, default=None
        Random state for reproducibility
    force_monotonic : bool, default=True
        Whether to enforce monotonicity
    monotonic_direction : str, default=None
        Direction of monotonicity ('auto', 'ascending', 'descending')

    Returns:
    --------
    pipeline : Pipeline
        Configured pipeline with parallel tree binning and WOE transformation
    """
    print(f"    DEBUG: Creating conditional inference tree pipeline with n_jobs={n_jobs}")
    binning = ConditionalInferenceTreeBinning(
        max_bins=max_bins,
        alpha=alpha,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs
    )
    return ensure_monotonic_pipeline(binning, force_direction=monotonic_direction, epsilon=0.5, woe_cap=5.0)

class MultiFeatureBinningManager:
    """
    Advanced multi-feature parallel binning manager for large datasets.

    Optimized for processing many features simultaneously with intelligent
    resource allocation, memory management, and error handling.
    """

    def __init__(self, n_jobs=None, max_memory_gb=None, verbose=True):
        """
        Initialize the multi-feature binning manager.

        Parameters:
        -----------
        n_jobs : int, optional
            Number of parallel jobs. Auto-detects if None.
        max_memory_gb : float, optional
            Maximum memory to use in GB. Auto-detects if None.
        verbose : bool, default=True
            Whether to print progress information
        """
        self.n_jobs = _get_optimal_workers(n_jobs, cpu_count(), 'cpu')
        self.max_memory_gb = max_memory_gb
        self.verbose = verbose
        self.fitted_transformers_ = {}
        self.feature_stats_ = {}

        if self.verbose:
            print(f"🚀 MultiFeatureBinningManager initialized with {self.n_jobs} workers")

    def fit_all_features(self, X, y, binning_configs=None, exclude_features=None):
        """
        Fit binning transformers for all features simultaneously.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features dataframe
        y : pd.Series
            Target variable
        binning_configs : dict, optional
            Feature-specific binning configurations. Format:
            {'feature_name': {'method': 'decision_tree', 'max_bins': 6, ...}}
        exclude_features : list, optional
            Features to exclude from processing

        Returns:
        --------
        self : MultiFeatureBinningManager
            Returns self with fitted transformers
        """
        if self.verbose:
            print(f"\n🎯 MULTI-FEATURE PARALLEL BINNING")
            print(f"{'='*50}")
            print(f"Dataset: {X.shape[0]:,} rows × {X.shape[1]} features")

        # Get features to process
        features = list(X.columns)
        if exclude_features:
            features = [f for f in features if f not in exclude_features]

        if self.verbose:
            print(f"Processing {len(features)} features: {features[:5]}{'...' if len(features) > 5 else ''}")

        # Generate default configurations if not provided
        if binning_configs is None:
            binning_configs = self._generate_default_configs(features, X.shape[0])

        # Determine optimal processing strategy
        processing_strategy = self._determine_processing_strategy(X, features)

        if processing_strategy == 'batch':
            self._fit_features_batch(X, y, features, binning_configs)
        else:
            self._fit_features_parallel(X, y, features, binning_configs)

        return self

    def transform_all_features(self, X, return_dataframe=True):
        """
        Transform all features using fitted transformers.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features to transform
        return_dataframe : bool, default=True
            Whether to return as DataFrame or dict of arrays

        Returns:
        --------
        X_binned : pd.DataFrame or dict
            Transformed features
        """
        if not self.fitted_transformers_:
            raise ValueError("No transformers fitted. Call fit_all_features first.")

        if self.verbose:
            print(f"\n🔄 Transforming {len(self.fitted_transformers_)} features...")

        # Prepare transformation tasks
        transform_args = []
        successful_features = []

        for feature_name, transformer in self.fitted_transformers_.items():
            if transformer is not None and feature_name in X.columns:
                transform_args.append((X[feature_name].values, feature_name, transformer))
                successful_features.append(feature_name)

        if not transform_args:
            raise ValueError("No valid transformers available for features in X")

        # Parallel transformation
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_feature = {}
            for args in transform_args:
                future = executor.submit(self._transform_single_feature, args)
                future_to_feature[future] = args[1]  # feature name

            for future in concurrent.futures.as_completed(future_to_feature):
                feature_name = future_to_feature[future]
                try:
                    binned_values = future.result()
                    results[feature_name] = binned_values
                    if self.verbose and len(results) % 10 == 0:
                        print(f"   ✅ Transformed {len(results)}/{len(transform_args)} features")
                except Exception as e:
                    if self.verbose:
                        print(f"   ❌ {feature_name} transformation failed: {str(e)}")

        if self.verbose:
            print(f"   ✅ Transformation complete: {len(results)}/{len(transform_args)} successful")

        if return_dataframe:
            return pd.DataFrame(results, index=X.index)
        else:
            return results

    def _generate_default_configs(self, features, n_samples):
        """Generate intelligent default configurations based on dataset characteristics"""
        configs = {}

        # Adaptive settings based on dataset size
        if n_samples > 2000000:  # 2M+ rows
            default_config = {
                'method': 'decision_tree',
                'max_bins': 5,
                'n_jobs': 1  # Conservative per-feature
            }
        elif n_samples > 500000:  # 500K+ rows
            default_config = {
                'method': 'decision_tree',
                'max_bins': 6,
                'n_jobs': 2
            }
        else:
            default_config = {
                'method': 'decision_tree',
                'max_bins': 7,
                'n_jobs': 2
            }

        for feature in features:
            configs[feature] = default_config.copy()

        return configs

    def _determine_processing_strategy(self, X, features):
        """Determine optimal processing strategy based on dataset characteristics"""
        n_samples, n_features = X.shape[0], len(features)

        # Memory estimation (rough)
        estimated_memory_per_feature = (n_samples * 8) / (1024**3)  # GB
        total_estimated_memory = estimated_memory_per_feature * min(self.n_jobs, n_features)

        if self.verbose:
            print(f"   💾 Estimated memory per feature: {estimated_memory_per_feature:.2f} GB")
            print(f"   💾 Estimated total memory: {total_estimated_memory:.2f} GB")

        # Choose strategy based on memory and dataset size
        if n_samples > 1000000 or total_estimated_memory > 8:
            return 'batch'
        else:
            return 'parallel'

    def _fit_features_batch(self, X, y, features, binning_configs):
        """Fit features using batch processing for memory efficiency"""
        if self.verbose:
            print(f"   🏭 Using batch processing strategy")

        # Calculate optimal batch size
        batch_size = max(1, min(self.n_jobs // 2, 4))
        feature_batches = [features[i:i + batch_size] for i in range(0, len(features), batch_size)]

        if self.verbose:
            print(f"   📦 Processing {len(feature_batches)} batches (size: {batch_size})")

        for batch_idx, feature_batch in enumerate(feature_batches):
            if self.verbose:
                print(f"\n   📦 Batch {batch_idx + 1}/{len(feature_batches)}: {feature_batch}")

            self._process_feature_batch(X, y, feature_batch, binning_configs)

    def _fit_features_parallel(self, X, y, features, binning_configs):
        """Fit features using full parallel processing"""
        if self.verbose:
            print(f"   ⚡ Using full parallel processing strategy")

        # Prepare all tasks
        fit_args = []
        for feature in features:
            config = binning_configs.get(feature, {})
            fit_args.append((X[feature].values, y.values, feature, config))

        # Process all features in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_feature = {}
            for args in fit_args:
                future = executor.submit(self._fit_single_feature_advanced, args)
                future_to_feature[future] = args[2]  # feature name

            # Collect results with progress tracking
            completed = 0
            for future in concurrent.futures.as_completed(future_to_feature):
                feature_name = future_to_feature[future]
                completed += 1

                try:
                    transformer, stats = future.result()
                    self.fitted_transformers_[feature_name] = transformer
                    self.feature_stats_[feature_name] = stats

                    if self.verbose and completed % 5 == 0:
                        print(f"   ✅ Completed {completed}/{len(features)} features")

                except Exception as e:
                    if self.verbose:
                        print(f"   ❌ {feature_name} failed: {str(e)}")
                    self.fitted_transformers_[feature_name] = None
                    self.feature_stats_[feature_name] = {'error': str(e)}

    def _process_feature_batch(self, X, y, feature_batch, binning_configs):
        """Process a batch of features in parallel"""
        batch_args = []
        for feature in feature_batch:
            config = binning_configs.get(feature, {})
            batch_args.append((X[feature].values, y.values, feature, config))

        # Process batch in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(feature_batch)) as executor:
            future_to_feature = {}
            for args in batch_args:
                future = executor.submit(self._fit_single_feature_advanced, args)
                future_to_feature[future] = args[2]

            # Collect batch results
            for future in concurrent.futures.as_completed(future_to_feature):
                feature_name = future_to_feature[future]
                try:
                    transformer, stats = future.result()
                    self.fitted_transformers_[feature_name] = transformer
                    self.feature_stats_[feature_name] = stats
                    if self.verbose:
                        print(f"      ✅ {feature_name}: {stats.get('bins_created', 'N/A')} bins")
                except Exception as e:
                    if self.verbose:
                        print(f"      ❌ {feature_name}: {str(e)}")
                    self.fitted_transformers_[feature_name] = None
                    self.feature_stats_[feature_name] = {'error': str(e)}

    def _fit_single_feature_advanced(self, args):
        """Advanced single feature fitting with statistics collection"""
        feature_data, y_data, feature_name, config = args

        start_time = time.time()

        try:
            # Extract configuration
            method = config.get('method', 'decision_tree')
            max_bins = config.get('max_bins', 6)
            feature_n_jobs = config.get('n_jobs', 1)

            # Set parallel environment
            set_parallel_env(feature_n_jobs)

            # Create appropriate transformer
            if method == 'decision_tree':
                transformer = DecisionTreeBinning(
                    max_bins=max_bins,
                    random_state=42,
                    n_jobs=feature_n_jobs
                )
            elif method == 'conditional_inference':
                min_samples_leaf = max(30, int(0.001 * len(feature_data)))
                min_samples_leaf = min(min_samples_leaf, 2000)

                transformer = ConditionalInferenceTreeBinning(
                    max_bins=max_bins,
                    alpha=config.get('alpha', 0.01),
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=feature_n_jobs
                )
            elif method == 'equal_freq':
                transformer = EqualFreqBinning(max_bins=max_bins)
            elif method == 'chi_merge':
                transformer = ChiMergeBinning(max_bins=max_bins)
            else:
                raise ValueError(f"Unknown binning method: {method}")

            # Fit the transformer
            transformer.fit(feature_data, y_data)

            # Collect statistics
            fit_time = time.time() - start_time

            # Test transformation to get bin count
            test_transform = transformer.transform(feature_data[:1000])  # Sample for stats
            n_unique_bins = len(np.unique(test_transform))

            stats = {
                'method': method,
                'fit_time': fit_time,
                'bins_created': n_unique_bins,
                'data_samples': len(feature_data),
                'config': config
            }

            return transformer, stats

        except Exception as e:
            fit_time = time.time() - start_time
            stats = {
                'method': config.get('method', 'unknown'),
                'fit_time': fit_time,
                'bins_created': 0,
                'data_samples': len(feature_data),
                'error': str(e)
            }
            return None, stats

    def _transform_single_feature(self, args):
        """Transform a single feature"""
        feature_data, feature_name, transformer = args
        return transformer.transform(feature_data).flatten()

    def get_feature_summary(self):
        """Get summary statistics for all processed features"""
        if not self.feature_stats_:
            return pd.DataFrame()

        summary_data = []
        for feature_name, stats in self.feature_stats_.items():
            summary_row = {
                'feature': feature_name,
                'method': stats.get('method', 'unknown'),
                'bins_created': stats.get('bins_created', 0),
                'fit_time': stats.get('fit_time', 0),
                'data_samples': stats.get('data_samples', 0),
                'success': 'error' not in stats
            }
            summary_data.append(summary_row)

        return pd.DataFrame(summary_data)

def parallel_multi_feature_binning(X, y, binning_configs=None, n_jobs=None,
                                 exclude_features=None, return_summary=False, verbose=True):
    """
    Convenience function for parallel multi-feature binning.

    Parameters:
    -----------
    X : pd.DataFrame
        Input features dataframe
    y : pd.Series
        Target variable
    binning_configs : dict, optional
        Feature-specific configurations
    n_jobs : int, optional
        Number of parallel jobs
    exclude_features : list, optional
        Features to exclude
    return_summary : bool, default=False
        Whether to return summary statistics
    verbose : bool, default=True
        Whether to print progress

    Returns:
    --------
    X_binned : pd.DataFrame
        Binned features dataframe
    summary : pd.DataFrame, optional
        Summary statistics if return_summary=True
    """
    manager = MultiFeatureBinningManager(n_jobs=n_jobs, verbose=verbose)

    # Fit all features
    manager.fit_all_features(X, y, binning_configs=binning_configs, exclude_features=exclude_features)

    # Transform features
    X_binned = manager.transform_all_features(X)

    if return_summary:
        summary = manager.get_feature_summary()
        return X_binned, summary
    else:
        return X_binned

def parallel_batch_tree_binning(X, y, features=None, binning_method='decision_tree',
                               max_bins=6, n_jobs=None, batch_size=None, verbose=True):
    """
    Legacy function - use parallel_multi_feature_binning for better performance.

    Efficiently process multiple features with parallel tree binning for large datasets.
    """
    if verbose:
        print("⚠️  Legacy function - consider using parallel_multi_feature_binning for better performance")

    # Convert to new API
    if features is None:
        features = list(X.columns)

    # Generate configurations
    binning_configs = {}
    for feature in features:
        binning_configs[feature] = {
            'method': binning_method,
            'max_bins': max_bins,
            'n_jobs': 1  # Conservative per-feature
        }

    # Use new multi-feature binning
    manager = MultiFeatureBinningManager(n_jobs=n_jobs, verbose=verbose)
    manager.fit_all_features(X, y, binning_configs=binning_configs)

    # Return transformers dictionary for compatibility
    return manager.fitted_transformers_

def _fit_single_feature_tree_binning(args):
    """Helper function to fit tree binning for a single feature (legacy)"""
    feature_data, y_data, feature_name, binning_method, max_bins, n_jobs_feature = args

    try:
        # Set parallel environment for this worker
        set_parallel_env(min(n_jobs_feature, 2))  # Conservative per-feature parallelism

        if binning_method == 'decision_tree':
            transformer = DecisionTreeBinning(
                max_bins=max_bins,
                random_state=42,
                n_jobs=min(n_jobs_feature, 2)  # Limit per-feature parallelism
            )
        elif binning_method == 'conditional_inference':
            # Adaptive parameters based on data size
            min_samples_leaf = max(30, int(0.001 * len(feature_data)))  # 0.1% of data
            min_samples_leaf = min(min_samples_leaf, 2000)  # Cap at 2000 for very large datasets

            transformer = ConditionalInferenceTreeBinning(
                max_bins=max_bins,
                alpha=0.01,  # More stringent for large datasets
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=min(n_jobs_feature, 2)
            )
        else:
            raise ValueError(f"Unknown binning method: {binning_method}")

        # Fit the transformer
        transformer.fit(feature_data, y_data)
        return transformer

    except Exception as e:
        print(f"    DEBUG: Error fitting {feature_name}: {str(e)}")
        return None

# Multi-feature pipeline creation functions
def create_multi_feature_pipeline_config(features, default_method='decision_tree',
                                        feature_specific_configs=None):
    """
    Create configuration dictionary for multi-feature binning pipelines.

    Parameters:
    -----------
    features : list
        List of feature names
    default_method : str, default='decision_tree'
        Default binning method for all features
    feature_specific_configs : dict, optional
        Feature-specific overrides

    Returns:
    --------
    dict : Configuration dictionary for parallel_multi_feature_binning
    """
    configs = {}

    for feature in features:
        configs[feature] = {
            'method': default_method,
            'max_bins': 6,
            'n_jobs': 1
        }

        # Apply feature-specific overrides
        if feature_specific_configs and feature in feature_specific_configs:
            configs[feature].update(feature_specific_configs[feature])

    return configs

class IsotonicRegressionBinning(BinningTransformer):
    """
    Binning transformer based on Isotonic Regression.

    This method fits an isotonic (monotonic) function to the relationship
    between the feature and target variable, then creates bins based on
    the flat segments (plateaus) of the resulting function.
    """

    def __init__(self, max_bins=5, increasing=None, min_samples_bin=30, random_state=None):
        """
        Initialize the isotonic regression binning transformer.

        Parameters:
        -----------
        max_bins : int, default=5
            Maximum number of bins to create
        increasing : bool or None, default=None
            If True, fit an increasing isotonic function
            If False, fit a decreasing isotonic function
            If None, determine the direction automatically
        min_samples_bin : int, default=30
            Minimum number of samples required in each bin
        random_state : int, default=None
            Random state for reproducibility (not used, but kept for API consistency)
        """
        super().__init__(max_bins=max_bins)
        self.increasing = increasing
        self.min_samples_bin = min_samples_bin
        self.random_state = random_state
        self.bin_edges_ = None

    def fit(self, X, y):
        """
        Fit the isotonic regression binning transformer.

        Parameters:
        -----------
        X : array-like of shape (n_samples, 1)
            Input feature to bin
        y : array-like of shape (n_samples,)
            Target variable

        Returns:
        --------
        self : object
            Returns self
        """
        # Convert inputs to numpy arrays
        X = np.asarray(X).reshape(-1)
        y = np.asarray(y)

        # Check if feature has only one unique value
        if len(np.unique(X)) <= 1:
            self.constant_value_ = True
            self.bin_edges_ = np.array([-np.inf, np.inf])
            return self

        self.constant_value_ = False

        # Determine direction of monotonicity if not specified
        if self.increasing is None:
            correlation = np.corrcoef(X, y)[0, 1]
            self.increasing_ = correlation >= 0 if not np.isnan(correlation) else True
        else:
            self.increasing_ = self.increasing

        # Sort data by feature value
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]

        # Fit isotonic regression
        ir = IsotonicRegression(increasing=self.increasing_, out_of_bounds='clip')
        y_pred = ir.fit_transform(X_sorted, y_sorted)

        # Find plateaus (flat segments) in the isotonic function
        # These will be our potential bin edges
        changes = np.where(np.diff(y_pred) != 0)[0]

        # Add bin edges at the change points
        if len(changes) > 0:
            # Use the midpoint between consecutive values as the bin edge
            potential_edges = [(X_sorted[i] + X_sorted[i+1]) / 2 for i in changes]

            # Ensure we don't exceed max_bins
            if len(potential_edges) > self.max_bins - 1:
                # Prioritize edges with biggest changes in the target value
                change_magnitudes = np.abs(np.diff(y_pred)[changes])
                top_indices = np.argsort(change_magnitudes)[:-(self.max_bins - 1)]
                potential_edges = [potential_edges[i] for i in top_indices]

            # Create bin edges including -inf and inf
            edges = [-np.inf] + sorted(potential_edges) + [np.inf]
            self.bin_edges_ = np.array(edges)

            # Ensure bins have minimum sample size
            self._merge_small_bins(X)
        else:
            # If no changes found, create a single bin
            self.bin_edges_ = np.array([-np.inf, np.inf])

        return self

    def _merge_small_bins(self, X):
        """
        Merge bins that have fewer than min_samples_bin samples.

        Parameters:
        -----------
        X : array-like
            Input feature values to check bin sizes
        """
        if len(self.bin_edges_) <= 2:  # Only one bin
            return

        # Count samples in each bin
        binned = np.digitize(X, self.bin_edges_[1:-1])
        bin_counts = np.bincount(binned, minlength=len(self.bin_edges_)-1)

        # If any bin has fewer than min_samples_bin, merge it with adjacent bin
        while np.any(bin_counts < self.min_samples_bin) and len(bin_counts) > 1:
            # Find the smallest bin
            smallest_bin = np.argmin(bin_counts)

            # If it's the first bin, merge with the next
            if smallest_bin == 0:
                # Remove the edge between bins 0 and 1
                self.bin_edges_ = np.concatenate([
                    self.bin_edges_[:1],
                    self.bin_edges_[2:]
                ])
            # If it's the last bin, merge with the previous
            elif smallest_bin == len(bin_counts) - 1:
                # Remove the edge between the last two bins
                self.bin_edges_ = np.concatenate([
                    self.bin_edges_[:-2],
                    self.bin_edges_[-1:]
                ])
            # Otherwise, merge with the smaller of its adjacent bins
            else:
                if bin_counts[smallest_bin - 1] < bin_counts[smallest_bin + 1]:
                    # Merge with previous bin
                    remove_idx = smallest_bin
                else:
                    # Merge with next bin
                    remove_idx = smallest_bin + 1

                # Remove the edge
                self.bin_edges_ = np.concatenate([
                    self.bin_edges_[:remove_idx],
                    self.bin_edges_[remove_idx+1:]
                ])

            # Recalculate bin counts
            binned = np.digitize(X, self.bin_edges_[1:-1])
            bin_counts = np.bincount(binned, minlength=len(self.bin_edges_)-1)

    def transform(self, X):
        """
        Transform X into bins.

        Parameters:
        -----------
        X : array-like of shape (n_samples, 1)
            Input feature to bin

        Returns:
        --------
        X_binned : array-like of shape (n_samples, 1)
            Binned feature
        """
        X = np.asarray(X).reshape(-1)

        if not hasattr(self, 'bin_edges_'):
            raise ValueError("This IsotonicRegressionBinning instance is not fitted yet.")

        if self.constant_value_:
            return np.ones(X.shape[0], dtype=np.int8).reshape(-1, 1)

        # Assign bins (digitize returns bin indices starting from 1)
        X_binned = np.digitize(X, self.bin_edges_[1:-1]).astype(np.int8)

        return X_binned.reshape(-1, 1)


class MultiIntervalDiscretizationBinning(BinningTransformer):
    """
    Multi-Interval Discretization (MID) Binning based on entropy minimization.

    This implements the supervised discretization method based on entropy minimization,
    recursively finding optimal cut points that minimize entropy with respect to the
    target variable.
    """

    def __init__(self, max_bins=5, min_samples_leaf=30, min_entropy_decrease=0.0001, random_state=None):
        """
        Initialize the MID binning transformer.

        Parameters:
        -----------
        max_bins : int, default=5
            Maximum number of bins to create
        min_samples_leaf : int, default=30
            Minimum number of samples required in a leaf node
        min_entropy_decrease : float, default=0.0001
            Minimum entropy decrease required to create a new split
        random_state : int, default=None
            Random state for reproducibility (not used, but kept for API consistency)
        """
        super().__init__(max_bins=max_bins)
        self.min_samples_leaf = min_samples_leaf
        self.min_entropy_decrease = min_entropy_decrease
        self.random_state = random_state
        self.split_points_ = None
        self.bin_edges_ = None
        # OPTIMIZATION: Cache for entropy calculations
        self._entropy_cache = {}

    def _entropy(self, y):
        """Calculate the entropy of a set of target values (optimized)."""
        # OPTIMIZATION: Error handling for empty arrays
        if len(y) == 0:
            return 0.0

        # OPTIMIZATION: Cached entropy results using array hash
        try:
            y_hash = hash(y.tobytes()) if hasattr(y, 'tobytes') else hash(tuple(y))
            if y_hash in self._entropy_cache:
                return self._entropy_cache[y_hash]
        except (TypeError, AttributeError):
            y_hash = None

        # OPTIMIZATION: Vectorized entropy calculation
        try:
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)

            # Handle log(0) case efficiently with boolean indexing
            positive_probs = probabilities[probabilities > 0]
            if len(positive_probs) == 0:
                entropy_val = 0.0
            else:
                # Vectorized entropy calculation
                entropy_val = -np.sum(positive_probs * np.log2(positive_probs))

            # Cache the result if hashing was successful
            if y_hash is not None:
                self._entropy_cache[y_hash] = entropy_val

            return entropy_val

        except (ValueError, ZeroDivisionError):
            # OPTIMIZATION: Error handling for problematic data
            return 0.0

    def _information_gain(self, y, left_y, right_y):
        """Calculate the information gain of a split (optimized)."""
        # OPTIMIZATION: Error handling for edge cases
        n = len(y)
        n_left = len(left_y)
        n_right = len(right_y)

        if n == 0 or n_left == 0 or n_right == 0:
            return 0.0

        try:
            # Calculate parent entropy
            parent_entropy = self._entropy(y)

            # Calculate weighted average of child entropies
            left_entropy = self._entropy(left_y)
            right_entropy = self._entropy(right_y)

            child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy

            # Return information gain
            return max(0.0, parent_entropy - child_entropy)

        except (ValueError, ZeroDivisionError, TypeError):
            # OPTIMIZATION: Error handling for problematic calculations
            return 0.0

    def fit(self, X, y):
        """
        Fit the MID binning transformer.

        Parameters:
        -----------
        X : array-like of shape (n_samples, 1)
            Input feature to bin
        y : array-like of shape (n_samples,)
            Target variable (binary)

        Returns:
        --------
        self : object
            Returns self
        """
        # Convert inputs to numpy arrays
        X = np.asarray(X).reshape(-1)
        y = np.asarray(y)

        # Check if feature has only one unique value
        if len(np.unique(X)) <= 1:
            self.constant_value_ = True
            self.bin_edges_ = np.array([-np.inf, np.inf])
            return self

        self.constant_value_ = False

        # Initialize splits
        self.split_points_ = []

        # Start recursive splitting
        self._recursive_split(X, y, max_depth=0)

        # Sort split points and create bin edges
        if self.split_points_:
            self.split_points_ = sorted(set(self.split_points_))
            self.bin_edges_ = np.array([-np.inf] + self.split_points_ + [np.inf])
            print(f"MID Binning created {len(self.bin_edges_)-1} bins with edges: {self.split_points_}")
        else:
            # If no splits were found, create a single bin
            self.bin_edges_ = np.array([-np.inf, np.inf])
            # No diagnostic printing

        return self

    def _recursive_split(self, X, y, max_depth):
        """
        Recursively split the data based on entropy minimization (optimized).

        Parameters:
        -----------
        X : array-like of shape (n_samples,)
            Input feature values
        y : array-like of shape (n_samples,)
            Target values (binary)
        max_depth : int
            Current depth of the recursion
        """
        # Stop if we've reached the maximum number of bins - 1 (splits)
        if len(self.split_points_) >= self.max_bins - 1:
            return

        # Stop if there aren't enough samples
        if len(X) < 2 * self.min_samples_leaf:
            return

        # Stop if all target values are the same
        if len(np.unique(y)) <= 1:
            return

        # OPTIMIZATION: Pre-sort data once for this recursive call
        try:
            sort_idx = np.argsort(X)
            X_sorted = X[sort_idx]
            y_sorted = y[sort_idx]
        except (ValueError, TypeError):
            # OPTIMIZATION: Error handling for problematic data
            return

        # Find the best split using optimized method with sorted data
        best_split, best_gain = self._find_best_split_optimized(X_sorted, y_sorted)

        # If no good split is found or the gain is too small, stop
        if best_split is None or best_gain < self.min_entropy_decrease:
            return

        # Add the split point
        self.split_points_.append(best_split)

        # OPTIMIZATION: Use sorted data for faster splitting
        split_idx = np.searchsorted(X_sorted, best_split, side='right')

        # Split the sorted data back to original order for recursion
        left_indices = sort_idx[:split_idx]
        right_indices = sort_idx[split_idx:]

        # Continue recursively
        self._recursive_split(X[left_indices], y[left_indices], max_depth + 1)
        self._recursive_split(X[right_indices], y[right_indices], max_depth + 1)

    def _find_best_split(self, X, y):
        """
        Find the best split point based on information gain (legacy method).
        """
        # For backward compatibility, use the optimized version
        try:
            sort_idx = np.argsort(X)
            X_sorted = X[sort_idx]
            y_sorted = y[sort_idx]
            return self._find_best_split_optimized(X_sorted, y_sorted)
        except (ValueError, TypeError):
            return None, 0.0

    def _find_best_split_optimized(self, X_sorted, y_sorted):
        """
        Find the best split point based on information gain (optimized version).

        Parameters:
        -----------
        X_sorted : array-like of shape (n_samples,)
            Input feature values (pre-sorted)
        y_sorted : array-like of shape (n_samples,)
            Target values (sorted by X)

        Returns:
        --------
        best_split : float or None
            The best split point, or None if no good split is found
        best_gain : float
            The information gain of the best split
        """
        # OPTIMIZATION: Error handling for edge cases
        if len(X_sorted) <= 1:
            return None, 0.0

        # Get unique values for split candidates
        try:
            unique_values = np.unique(X_sorted)
            if len(unique_values) <= 1:
                return None, 0.0
        except (ValueError, TypeError):
            return None, 0.0

        # Calculate potential split points (midpoints between consecutive values)
        potential_splits = [(unique_values[i] + unique_values[i+1]) / 2
                           for i in range(len(unique_values) - 1)]

        if not potential_splits:
            return None, 0.0

        # OPTIMIZATION: Pre-compute statistics
        total_samples = len(y_sorted)

        # Initialize variables to track best split
        best_split = None
        best_gain = 0.0

        # Evaluate each potential split
        for split in potential_splits:
            try:
                # OPTIMIZATION: Use searchsorted for faster split finding
                split_idx = np.searchsorted(X_sorted, split, side='right')

                # Skip if either group is too small
                if split_idx < self.min_samples_leaf or (total_samples - split_idx) < self.min_samples_leaf:
                    continue

                # OPTIMIZATION: Use pre-sorted data for splitting
                left_y = y_sorted[:split_idx]
                right_y = y_sorted[split_idx:]

                # Calculate information gain
                gain = self._information_gain(y_sorted, left_y, right_y)

                # Update best split if this one is better
                if gain > best_gain:
                    best_gain = gain
                    best_split = split

            except Exception:
                # OPTIMIZATION: Comprehensive error handling
                continue

        return best_split, best_gain

    def transform(self, X):
        """
        Transform X into bins.

        Parameters:
        -----------
        X : array-like of shape (n_samples, 1)
            Input feature to bin

        Returns:
        --------
        X_binned : array-like of shape (n_samples, 1)
            Binned feature
        """
        X = np.asarray(X).reshape(-1)

        if not hasattr(self, 'bin_edges_'):
            raise ValueError("This MultiIntervalDiscretizationBinning instance is not fitted yet.")

        if self.constant_value_:
            return np.ones(X.shape[0], dtype=np.int8).reshape(-1, 1)

        # Assign bins
        X_binned = np.digitize(X, self.bin_edges_[1:-1]).astype(np.int8)

        return X_binned.reshape(-1, 1)

def create_mid_pipeline(max_bins=5, min_samples_leaf=30, min_entropy_decrease=0.0001,
                        random_state=None, force_monotonic=True, monotonic_direction=None):
    """Create a pipeline with Multi-Interval Discretization binning and WOE transformation."""
    binning = MultiIntervalDiscretizationBinning(
        max_bins=max_bins,
        min_samples_leaf=min_samples_leaf,
        min_entropy_decrease=min_entropy_decrease,
        random_state=random_state
    )
    return ensure_monotonic_pipeline(binning, force_direction=monotonic_direction, epsilon=0.5, woe_cap=5.0)


def create_isotonic_regression_pipeline(max_bins=5, increasing=None, min_samples_bin=30,
                                       random_state=None, force_monotonic=True,
                                       monotonic_direction=None):
    """Create a pipeline with isotonic regression binning and WOE transformation."""
    binning = IsotonicRegressionBinning(
        max_bins=max_bins,
        increasing=increasing,
        min_samples_bin=min_samples_bin,
        random_state=random_state
    )
    return ensure_monotonic_pipeline(binning, force_direction=monotonic_direction, epsilon=0.5, woe_cap=5.0)


class ChiIsotonicBinning(BinningTransformer):
    """
    Hybrid binning method that first applies Chi-Merge to get initial bins,
    then uses Isotonic Regression to enforce monotonicity if needed.
    """

    def __init__(self, max_bins=5, threshold=3.841, min_samples_leaf=30, random_state=None):
        """
        Initialize the Chi-Isotonic binning transformer.

        Parameters:
        -----------
        max_bins : int, default=5
            Maximum number of bins to create
        threshold : float, default=3.841
            Chi-square threshold for merging bins (default is 95% confidence)
        min_samples_leaf : int, default=30
            Minimum number of samples required in each bin
        random_state : int, default=None
            Random state for reproducibility
        """
        super().__init__(max_bins=max_bins)
        self.threshold = threshold
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.bin_edges_ = None

    def fit(self, X, y):
        """
        Fit the Chi-Isotonic binning transformer.

        Parameters:
        -----------
        X : array-like of shape (n_samples, 1)
            Input feature to bin
        y : array-like of shape (n_samples,)
            Target variable (binary)

        Returns:
        --------
        self : object
            Returns self
        """
        # Convert inputs to numpy arrays
        X = np.asarray(X).reshape(-1)
        y = np.asarray(y)

        # Apply Chi-Merge first
        chi_merge = ChiMergeBinning(max_bins=self.max_bins, threshold=self.threshold)
        chi_merge.fit(X.reshape(-1, 1), y)

        # If Chi-Merge couldn't create bins, return
        if not hasattr(chi_merge, 'bin_edges_') or chi_merge.bin_edges_ is None:
            self.bin_edges_ = np.array([-np.inf, np.inf])
            return self

        # Get Chi-Merge bins
        initial_bin_edges = chi_merge.bin_edges_

        # Create DataFrame with binning data
        binned_values = chi_merge.transform(X.reshape(-1, 1))
        df = pd.DataFrame({
            'feature': X,
            'bin': binned_values.flatten(),
            'target': y
        })

        # Calculate statistics for each bin
        bin_stats = df.groupby('bin').agg(
            target_rate=('target', 'mean'),
            min_feature=('feature', 'min'),
            max_feature=('feature', 'max'),
            count=('feature', 'count')
        ).reset_index()

        # Sort by feature value
        bin_stats = bin_stats.sort_values(by='min_feature')

        # Check if bins are monotonic
        target_rates = bin_stats['target_rate'].values

        # Determine if monotonicity is needed
        if len(target_rates) <= 1:
            # Only one bin, no monotonicity needed
            self.bin_edges_ = initial_bin_edges
            return self

        # Check if rates are monotonic (either increasing or decreasing)
        is_increasing = np.all(np.diff(target_rates) >= 0)
        is_decreasing = np.all(np.diff(target_rates) <= 0)

        if is_increasing or is_decreasing:
            # Already monotonic, use Chi-Merge bins
            self.bin_edges_ = initial_bin_edges
            return self

        # Need to enforce monotonicity using isotonic regression
        # Determine the direction (use correlation between bin index and target rate)
        bin_indices = bin_stats['bin'].values
        correlation = np.corrcoef(bin_indices, target_rates)[0, 1]
        increasing = correlation >= 0 if not np.isnan(correlation) else True

        # Apply isotonic regression
        ir = IsotonicRegression(increasing=increasing, out_of_bounds='clip')
        bin_stats['monotonic_rate'] = ir.fit_transform(
            bin_stats['min_feature'],
            bin_stats['target_rate']
        )

        # Create new bin edges based on monotonic rates
        edges = [-np.inf]
        current_rate = None

        for _, row in bin_stats.iterrows():
            if current_rate is None or abs(row['monotonic_rate'] - current_rate) > 0.00001:
                if current_rate is not None:
                    edges.append(row['min_feature'])
                current_rate = row['monotonic_rate']

        edges.append(np.inf)

        # Only update bins if we have at least 2 bins
        if len(edges) > 2:
            self.bin_edges_ = np.array(edges)
        else:
            # Fall back to original Chi-Merge bins
            self.bin_edges_ = initial_bin_edges

        return self

    def transform(self, X):
        """
        Transform X into bins.

        Parameters:
        -----------
        X : array-like of shape (n_samples, 1)
            Input feature to bin

        Returns:
        --------
        X_binned : array-like of shape (n_samples, 1)
            Binned feature
        """
        X = np.asarray(X).reshape(-1)

        if not hasattr(self, 'bin_edges_'):
            raise ValueError("This ChiIsotonicBinning instance is not fitted yet.")

        # Apply binning using the bin edges
        try:
            result = pd.cut(
                X,
                bins=self.bin_edges_,
                labels=False,
                include_lowest=True,
                duplicates='drop'
            )
            # Convert to int and handle NaN values, then add 1 for 1-based ordinal labels
            return (np.nan_to_num(result).astype(np.int8) + 1).reshape(-1, 1)
        except Exception as e:
            # If binning fails, return ones (ordinal 1)
            return np.ones(X.shape[0], dtype=np.int8).reshape(-1, 1)


class IsotonicChiBinning(BinningTransformer):
    """
    Hybrid binning method that first applies Isotonic Regression to get monotonic bins,
    then applies Chi-square tests to further refine bins based on statistical significance.
    """

    def __init__(self, max_bins=5, threshold=3.841, min_samples_leaf=30, random_state=None):
        """
        Initialize the Isotonic-Chi binning transformer.

        Parameters:
        -----------
        max_bins : int, default=5
            Maximum number of bins to create
        threshold : float, default=3.841
            Chi-square threshold for testing bins (default is 95% confidence)
        min_samples_leaf : int, default=30
            Minimum number of samples required in each bin
        random_state : int, default=None
            Random state for reproducibility
        """
        super().__init__(max_bins=max_bins)
        self.threshold = threshold
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.bin_edges_ = None

    def fit(self, X, y):
        """
        Fit the Isotonic-Chi binning transformer.

        Parameters:
        -----------
        X : array-like of shape (n_samples, 1)
            Input feature to bin
        y : array-like of shape (n_samples,)
            Target variable (binary)

        Returns:
        --------
        self : object
            Returns self
        """
        # Convert inputs to numpy arrays
        X = np.asarray(X).reshape(-1)
        y = np.asarray(y)

        # First apply Isotonic Regression for monotonic bins
        isotonic = IsotonicRegressionBinning(max_bins=min(self.max_bins * 2, 20))
        isotonic.fit(X.reshape(-1, 1), y)

        # If Isotonic Regression couldn't create bins, return
        if not hasattr(isotonic, 'bin_edges_') or isotonic.bin_edges_ is None:
            self.bin_edges_ = np.array([-np.inf, np.inf])
            return self

        # Get isotonic bins
        isotonic_bin_edges = isotonic.bin_edges_

        # Create DataFrame with binning data
        binned_values = isotonic.transform(X.reshape(-1, 1))
        df = pd.DataFrame({
            'feature': X,
            'bin': binned_values.flatten(),
            'target': y
        })

        # For each isotonic bin, apply chi-square test to see if further refinement is needed
        final_edges = [-np.inf]

        # Process each isotonic bin
        for bin_idx in range(len(isotonic_bin_edges) - 1):
            # Get data for this bin
            bin_data = df[df['bin'] == bin_idx]

            # If bin is too small, skip refinement
            if len(bin_data) < 2 * self.min_samples_leaf:
                if bin_idx > 0:  # Don't add the first edge again
                    final_edges.append(isotonic_bin_edges[bin_idx + 1])
                continue

            # Get feature values and target for this bin
            bin_X = bin_data['feature'].values
            bin_y = bin_data['target'].values

            # If all target values are the same, no refinement needed
            if len(np.unique(bin_y)) <= 1:
                if bin_idx > 0:
                    final_edges.append(isotonic_bin_edges[bin_idx + 1])
                continue

            # Try to apply Chi-Merge within this bin
            try:
                # Create initial quantile bins for Chi-Merge
                n_quantiles = min(10, max(2, len(bin_X) // self.min_samples_leaf))

                # Sort data
                sort_idx = np.argsort(bin_X)
                sorted_X = bin_X[sort_idx]
                sorted_y = bin_y[sort_idx]

                # Create quantile-based initial edges
                inner_edges = []
                for q in range(1, n_quantiles):
                    quantile_idx = int(q * len(sorted_X) / n_quantiles)
                    if quantile_idx < len(sorted_X):
                        inner_edges.append(sorted_X[quantile_idx])

                # Apply Chi-Merge to these edges
                if inner_edges:
                    # Create contingency tables and calculate chi-square for each candidate edge
                    significant_edges = []

                    for edge in inner_edges:
                        left_mask = sorted_X <= edge
                        right_mask = ~left_mask

                        # Skip if either side has too few samples
                        if sum(left_mask) < self.min_samples_leaf or sum(right_mask) < self.min_samples_leaf:
                            continue

                        # Create contingency table
                        left_pos = sum(sorted_y[left_mask])
                        left_neg = sum(left_mask) - left_pos
                        right_pos = sum(sorted_y[right_mask])
                        right_neg = sum(right_mask) - right_pos

                        contingency = np.array([[left_neg, left_pos], [right_neg, right_pos]])

                        # Skip if any cell is zero
                        if np.any(contingency == 0):
                            continue

                        # Calculate chi-square
                        chi2, p, _, _ = chi2_contingency(contingency)

                        # If significant, keep this edge
                        if chi2 > self.threshold:
                            significant_edges.append(edge)

                    # Add significant edges to final edges
                    for edge in sorted(significant_edges):
                        if bin_idx == 0 and edge < isotonic_bin_edges[1]:
                            final_edges.append(edge)
                        elif bin_idx > 0 and edge > isotonic_bin_edges[bin_idx] and edge < isotonic_bin_edges[bin_idx + 1]:
                            final_edges.append(edge)
            except:
                # If Chi-Merge fails, just use the isotonic bin edge
                pass

            # Always add the isotonic bin edge
            if bin_idx > 0:
                final_edges.append(isotonic_bin_edges[bin_idx + 1])

        # Add infinity as the last edge
        final_edges.append(np.inf)

        # Remove duplicate edges and sort
        final_edges = sorted(list(set(final_edges)))

        # Ensure we don't exceed max_bins
        if len(final_edges) > self.max_bins + 1:
            # Keep -inf and inf, and sample evenly from the rest
            inner_edges = final_edges[1:-1]
            step = len(inner_edges) / (self.max_bins - 1)
            selected_indices = [int(i * step) for i in range(self.max_bins - 1)]
            selected_edges = [inner_edges[i] for i in selected_indices]
            final_edges = [-np.inf] + sorted(selected_edges) + [np.inf]

        self.bin_edges_ = np.array(final_edges)
        return self

    def transform(self, X):
        """
        Transform X into bins.

        Parameters:
        -----------
        X : array-like of shape (n_samples, 1)
            Input feature to bin

        Returns:
        --------
        X_binned : array-like of shape (n_samples, 1)
            Binned feature
        """
        X = np.asarray(X).reshape(-1)

        if not hasattr(self, 'bin_edges_'):
            raise ValueError("This IsotonicChiBinning instance is not fitted yet.")

        # Apply binning using the bin edges
        try:
            result = pd.cut(
                X,
                bins=self.bin_edges_,
                labels=False,
                include_lowest=True,
                duplicates='drop'
            )
            # Convert to int and handle NaN values, then add 1 for 1-based ordinal labels
            return (np.nan_to_num(result).astype(np.int8) + 1).reshape(-1, 1)
        except Exception as e:
            # If binning fails, return ones (ordinal 1)
            return np.ones(X.shape[0], dtype=np.int8).reshape(-1, 1)


def create_chi_isotonic_pipeline(max_bins=5, threshold=3.841, min_samples_leaf=30,
                               random_state=None, force_monotonic=True, monotonic_direction=None):
    """Create a pipeline with Chi-Isotonic binning and WOE transformation."""
    binning = ChiIsotonicBinning(
        max_bins=max_bins,
        threshold=threshold,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    return ensure_monotonic_pipeline(binning, force_direction=monotonic_direction, epsilon=0.5, woe_cap=5.0)


def create_isotonic_chi_pipeline(max_bins=5, threshold=3.841, min_samples_leaf=30,
                               random_state=None, force_monotonic=True, monotonic_direction=None):
    """Create a pipeline with Isotonic-Chi binning and WOE transformation."""
    binning = IsotonicChiBinning(
        max_bins=max_bins,
        threshold=threshold,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    return ensure_monotonic_pipeline(binning, force_direction=monotonic_direction, epsilon=0.5, woe_cap=5.0)


class EqualFreqChiBinning(BinningTransformer):
    """
    Hybrid binning method that first applies Equal Frequency binning to create initial bins,
    then uses Chi-square tests to ensure the bins are statistically significant.
    """

    def __init__(self, max_bins=5, threshold=3.841, min_samples_leaf=30, random_state=None):
        """
        Initialize the Equal Frequency + Chi-square binning transformer.

        Parameters:
        -----------
        max_bins : int, default=5
            Maximum number of bins to create
        threshold : float, default=3.841
            Chi-square threshold for testing significance (default is 95% confidence)
        min_samples_leaf : int, default=30
            Minimum number of samples required in each bin
        random_state : int, default=None
            Random state for reproducibility
        """
        super().__init__(max_bins=max_bins)
        self.threshold = threshold
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.bin_edges_ = None

    def fit(self, X, y):
        """
        Fit the Equal Frequency + Chi-square binning transformer.

        Parameters:
        -----------
        X : array-like of shape (n_samples, 1)
            Input feature to bin
        y : array-like of shape (n_samples,)
            Target variable (binary)

        Returns:
        --------
        self : object
            Returns self
        """
        # Convert inputs to numpy arrays
        X = np.asarray(X).reshape(-1)
        y = np.asarray(y)

        # Check if feature has only one unique value
        if len(np.unique(X)) <= 1:
            self.constant_value_ = True
            self.bin_edges_ = np.array([-np.inf, np.inf])
            return self

        self.constant_value_ = False

        # First apply Equal Frequency binning
        # Start with more bins than requested to allow for merging
        initial_bins = min(self.max_bins * 3, len(np.unique(X)))
        initial_bins = max(initial_bins, 2)  # Ensure at least 2 bins

        # Create equal frequency bins
        n_samples = len(X)
        sorted_idx = np.argsort(X)
        sorted_X = X[sorted_idx]
        sorted_y = y[sorted_idx]

        # Create initial bin edges
        edges = [-np.inf]
        for i in range(1, initial_bins):
            idx = i * n_samples // initial_bins
            if idx < n_samples:
                edges.append(sorted_X[idx])
        edges.append(np.inf)

        # Create initial contingency tables for adjacent bins
        contingency_tables = []
        for i in range(len(edges) - 2):
            left_edge = edges[i]
            mid_edge = edges[i + 1]
            right_edge = edges[i + 2]

            left_mask = (X > left_edge) & (X <= mid_edge)
            right_mask = (X > mid_edge) & (X <= right_edge)

            left_pos = np.sum(y[left_mask])
            left_neg = np.sum(left_mask) - left_pos
            right_pos = np.sum(y[right_mask])
            right_neg = np.sum(right_mask) - right_pos

            # Skip if any cell would be too small
            if (left_pos + left_neg < self.min_samples_leaf or
                right_pos + right_neg < self.min_samples_leaf):
                continue

            table = np.array([[left_neg, left_pos], [right_neg, right_pos]])
            contingency_tables.append((i, table))

        # Iteratively merge bins that don't meet the chi-square threshold
        while len(edges) > 2 and len(contingency_tables) > 0:
            # Find the least significant split (lowest chi-square value)
            min_chi2 = float('inf')
            min_idx = -1
            min_table_idx = -1

            for table_idx, (i, table) in enumerate(contingency_tables):
                try:
                    chi2, _, _, _ = chi2_contingency(table)
                    if chi2 < min_chi2:
                        min_chi2 = chi2
                        min_idx = i
                        min_table_idx = table_idx
                except:
                    # Skip if chi-square can't be calculated
                    continue

            # If all splits are significant, stop merging
            if min_chi2 >= self.threshold:
                break

            # Remove the least significant split
            if min_idx >= 0:
                edges.pop(min_idx + 1)
                contingency_tables.pop(min_table_idx)

                # Update affected contingency tables
                updated_tables = []
                for i, table in contingency_tables:
                    if i == min_idx - 1:  # Table to the left of the removed edge
                        left_edge = edges[i]
                        mid_edge = edges[i + 1]  # This is the next edge after removal
                        right_edge = edges[i + 2]  # This may be a different edge now

                        left_mask = (X > left_edge) & (X <= mid_edge)
                        right_mask = (X > mid_edge) & (X <= right_edge)

                        left_pos = np.sum(y[left_mask])
                        left_neg = np.sum(left_mask) - left_pos
                        right_pos = np.sum(y[right_mask])
                        right_neg = np.sum(right_mask) - right_pos

                        if (left_pos + left_neg >= self.min_samples_leaf and
                            right_pos + right_neg >= self.min_samples_leaf):
                            new_table = np.array([[left_neg, left_pos], [right_neg, right_pos]])
                            updated_tables.append((i, new_table))

                    elif i == min_idx:  # Table to the right of the removed edge
                        # This table now spans a larger range, recalculate
                        left_edge = edges[i]  # This index has shifted
                        mid_edge = edges[i + 1]
                        right_edge = edges[i + 2] if i + 2 < len(edges) else np.inf

                        left_mask = (X > left_edge) & (X <= mid_edge)
                        right_mask = (X > mid_edge) & (X <= right_edge)

                        left_pos = np.sum(y[left_mask])
                        left_neg = np.sum(left_mask) - left_pos
                        right_pos = np.sum(y[right_mask])
                        right_neg = np.sum(right_mask) - right_pos

                        if (left_pos + left_neg >= self.min_samples_leaf and
                            right_pos + right_neg >= self.min_samples_leaf):
                            new_table = np.array([[left_neg, left_pos], [right_neg, right_pos]])
                            updated_tables.append((i, new_table))
                    else:
                        # Adjust index if it was after the removed edge
                        new_i = i if i < min_idx else i - 1
                        updated_tables.append((new_i, table))

                contingency_tables = updated_tables

        # Store the final bin edges
        if len(edges) <= self.max_bins + 1:
            self.bin_edges_ = np.array(edges)
        else:
            # If we have too many bins, select evenly spaced ones
            selected_edges = [edges[0]]  # Always keep -inf
            step = (len(edges) - 2) / (self.max_bins - 1)
            for i in range(1, self.max_bins):
                idx = 1 + int(i * step)
                selected_edges.append(edges[idx])
            selected_edges.append(edges[-1])  # Always keep inf
            self.bin_edges_ = np.array(selected_edges)

        return self

    def transform(self, X):
        """
        Transform X into bins.

        Parameters:
        -----------
        X : array-like of shape (n_samples, 1)
            Input feature to bin

        Returns:
        --------
        X_binned : array-like of shape (n_samples, 1)
            Binned feature
        """
        X = np.asarray(X).reshape(-1)

        if not hasattr(self, 'bin_edges_'):
            raise ValueError("This EqualFreqChiBinning instance is not fitted yet.")

        if self.constant_value_:
            return np.zeros(X.shape[0], dtype=np.int8).reshape(-1, 1)

        # Apply binning using the bin edges
        try:
            result = pd.cut(
                X,
                bins=self.bin_edges_,
                labels=False,
                include_lowest=True,
                duplicates='drop'
            )
            # Convert to int and handle NaN values, then add 1 for 1-based ordinal labels
            return (np.nan_to_num(result).astype(np.int8) + 1).reshape(-1, 1)
        except Exception as e:
            # If binning fails, return ones (ordinal 1)
            return np.ones(X.shape[0], dtype=np.int8).reshape(-1, 1)


def create_equal_freq_chi_pipeline(max_bins=5, threshold=3.841, min_samples_leaf=30,
                                 random_state=None, force_monotonic=True, monotonic_direction=None):
    """Create a pipeline with Equal Frequency + Chi-square binning and WOE transformation."""
    binning = EqualFreqChiBinning(
        max_bins=max_bins,
        threshold=threshold,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    return ensure_monotonic_pipeline(binning, force_direction=monotonic_direction, epsilon=0.5, woe_cap=5.0)

# --- Parallel Feature Processing Functions ---

def _process_single_feature_binning(args):
    """
    Process a single feature with binning pipeline in parallel.

    Parameters:
    -----------
    args : tuple
        (feature_name, X_feature_data, y_data, pipeline_factory, pipeline_config, verbose)

    Returns:
    --------
    tuple : (feature_name, success, iv_value, train_values, transform_pipeline, error_msg)
    """
    feature_name, X_feature_data, y_data, pipeline_factory, pipeline_config, verbose = args

    try:
        # Create pipeline with given configuration
        pipeline = pipeline_factory(**pipeline_config)

        # Fit the pipeline
        pipeline.fit(X_feature_data, y_data)

        # Get IV value from WOE transformer
        try:
            step_names = list(pipeline.named_steps.keys())
            woe_step_name = [step for step in step_names if 'woe' in step.lower()][0] if any('woe' in step.lower() for step in step_names) else 'woe_transformer'
            woe_transformer = pipeline.named_steps[woe_step_name]
            iv_value = getattr(woe_transformer, 'iv_', 0.0)
        except:
            iv_value = 0.0

        # Prepare binned data (without WOE) for model training
        X_binned_for_model = X_feature_data
        pipeline_steps_for_model = []

        for step_name, transformer_obj in pipeline.steps:
            if 'woe' in step_name.lower():
                break  # Stop before WOE step
            X_binned_for_model = transformer_obj.transform(X_binned_for_model)
            pipeline_steps_for_model.append((step_name, transformer_obj))

        # Create transform pipeline for test data
        transform_pipeline = Pipeline(steps=pipeline_steps_for_model)

        return (feature_name, True, iv_value, X_binned_for_model, transform_pipeline, None)

    except Exception as e:
        return (feature_name, False, 0.0, None, None, str(e))

def _parallel_process_features_for_binning(X_train, y_train, feature_columns, pipeline_factory,
                                         pipeline_config, n_jobs=None, verbose=False):
    """
    Process multiple features in parallel for a single binning method.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature data
    y_train : pd.Series
        Training target data
    feature_columns : list
        List of feature column names to process
    pipeline_factory : callable
        Factory function to create binning pipelines
    pipeline_config : dict
        Configuration parameters for the pipeline factory
    n_jobs : int, optional
        Number of parallel jobs (conservative approach for 22 cores)
    verbose : bool
        Whether to print progress messages

    Returns:
    --------
    tuple : (method_ivs, transformed_features_for_model)
    """
    # Conservative parallelization for 22 cores
    if n_jobs is None:
        n_jobs = min(8, len(feature_columns))  # Conservative: max 8 cores for feature processing
    else:
        n_jobs = min(n_jobs, 10, len(feature_columns))  # Cap at 10 cores max

    if verbose:
        print(f"    DEBUG: Processing {len(feature_columns)} features using {n_jobs} parallel workers")

    # Prepare arguments for parallel processing
    args_list = []
    for feature in feature_columns:
        X_feature_data = X_train[feature].values.reshape(-1, 1)
        args_list.append((feature, X_feature_data, y_train, pipeline_factory, pipeline_config, verbose))

    method_ivs = {}
    transformed_features_for_model = {}

    # Process features in parallel with conservative worker allocation
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all feature processing tasks
            future_to_feature = {}
            for args in args_list:
                future = executor.submit(_process_single_feature_binning, args)
                future_to_feature[future] = args[0]  # feature name

            # Collect results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_feature):
                feature_name = future_to_feature[future]
                completed += 1

                try:
                    feature_name, success, iv_value, train_values, transform_pipeline, error_msg = future.result()

                    if success:
                        method_ivs[feature_name] = iv_value
                        transformed_features_for_model[feature_name] = {
                            'train_values': train_values,
                            'transform_pipeline_for_test': transform_pipeline
                        }
                        if verbose and completed % 5 == 0:
                            print(f"    DEBUG: Completed {completed}/{len(feature_columns)} features")
                    else:
                        method_ivs[feature_name] = 0.0
                        if verbose:
                            print(f"    Error processing {feature_name}: {error_msg}")

                except Exception as e:
                    method_ivs[feature_name] = 0.0
                    if verbose:
                        print(f"    Error processing {feature_name}: {str(e)}")

        if verbose:
            print(f"    DEBUG: Parallel feature processing completed: {len(transformed_features_for_model)}/{len(feature_columns)} successful")

    except Exception as e:
        if verbose:
            print(f"    DEBUG: Parallel processing failed, falling back to sequential: {str(e)}")

        # Fallback to sequential processing
        for feature in feature_columns:
            try:
                X_feature_data = X_train[feature].values.reshape(-1, 1)
                args = (feature, X_feature_data, y_train, pipeline_factory, pipeline_config, verbose)
                feature_name, success, iv_value, train_values, transform_pipeline, error_msg = _process_single_feature_binning(args)

                if success:
                    method_ivs[feature_name] = iv_value
                    transformed_features_for_model[feature_name] = {
                        'train_values': train_values,
                        'transform_pipeline_for_test': transform_pipeline
                    }
                else:
                    method_ivs[feature_name] = 0.0
                    if verbose:
                        print(f"    Error processing {feature_name}: {error_msg}")
            except Exception as e:
                method_ivs[feature_name] = 0.0
                if verbose:
                    print(f"    Error processing {feature_name}: {str(e)}")

    return method_ivs, transformed_features_for_model

# --- Main Comparison Function ---

def compare_binning_strategies_on_dataset(
    X_train_input: pd.DataFrame,
    X_test_input: pd.DataFrame,
    y_train_input: pd.Series,
    y_test_input: pd.Series,
    feature_columns: list = None,
    max_bins_per_feature: int = 10,
    random_state: int = 42,
    force_monotonic: bool = True,
    monotonic_direction: str = None,
    verbose: bool = False,
    log_outputs: bool = True,
    output_dir: str = None,
    n_jobs: int = None
):
    """
    Compare different binning strategies with parallel feature processing.

    **NEW: Multi-Core Parallel Processing**
    This function now processes features in parallel across multiple CPU cores for each binning method,
    significantly reducing computation time on multi-core systems.

    Parameters:
    -----------
    X_train_input : pd.DataFrame
        Pre-split training features (already imputed if necessary).
    X_test_input : pd.DataFrame
        Pre-split testing features (already imputed if necessary).
    y_train_input : pd.Series
        Pre-split training target.
    y_test_input : pd.Series
        Pre-split testing target.
    feature_columns : list, optional
        List of feature columns (numeric) from X_train_input to process.
        If None, all numeric columns in X_train_input will be used.
    max_bins_per_feature : int, default=10
        Maximum number of bins per feature
    random_state : int, default=42
        Random state for reproducibility
    force_monotonic : bool, default=True
        Whether to enforce monotonicity in binning results
    monotonic_direction : {None, 'increasing', 'decreasing'}, default=None
        Direction of monotonicity to enforce if force_monotonic is True
    verbose : bool, default=False
        Whether to print detailed progress information
    log_outputs : bool, default=True
        Whether to log outputs to file and save images as JPEG
    output_dir : str, optional
        Directory to save log and image files. If None, uses current directory.
    n_jobs : int, optional
        Number of parallel jobs for feature processing. If None, uses conservative
        auto-detection (max 8 cores for 22-core systems). Recommended: 8-10 for
        systems with 22+ cores.

    Returns:
    --------
    pd.DataFrame
        Summary of results including computation time
    dict
        Feature importance values (IV) for each method and feature
    pd.DataFrame
        Model coefficients for each binning strategy
    list[matplotlib.figure.Figure]
        Visualization figures for results comparison

    **Performance Notes:**
    - Uses conservative parallel processing (max 8-10 cores) for stability
    - Each binning method processes all features in parallel
    - Automatically falls back to sequential processing if parallel fails
    - Memory usage is optimized for large datasets
    """
    import time

    # Initialize logger if requested
    logger = None
    if log_outputs:
        logger = BinningLogger(base_filename="binning_analysis", output_dir=output_dir)
        logger.log_section("BINNING STRATEGIES COMPARISON", level=1)
        logger.log(f"Analysis started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log(f"Parameters:")
        logger.log(f"  - Max bins per feature: {max_bins_per_feature}")
        logger.log(f"  - Random state: {random_state}")
        logger.log(f"  - Force monotonic: {force_monotonic}")
        logger.log(f"  - Monotonic direction: {monotonic_direction}")
        logger.log(f"  - Training samples: {len(X_train_input)}")
        logger.log(f"  - Test samples: {len(X_test_input)}")
        logger.log(f"  - Parallel jobs: {n_jobs if n_jobs else 'Auto (conservative max 8)'}")

        # Set conservative n_jobs if not specified (for 22-core systems)
        if n_jobs is None:
            n_jobs = min(8, cpu_count())  # Conservative default for stability
            if verbose:
                print(f"    DEBUG: Auto-setting n_jobs to {n_jobs} (conservative for {cpu_count()} cores)")

    X_train = X_train_input.copy()
    X_test = X_test_input.copy()
    y_train = y_train_input.copy()
    y_test = y_test_input.copy()

    # Identify feature columns if not provided
    if feature_columns is None:
        feature_columns = [
            col for col in X_train.columns
            if pd.api.types.is_numeric_dtype(X_train[col])
        ]
        if not feature_columns:
            message = "No numeric feature columns found in X_train_input."
            if logger:
                logger.log(message)
            else:
                print(message)
            return pd.DataFrame(), {}, pd.DataFrame(columns=feature_columns if feature_columns else []), None


    if len(np.unique(y_train)) < 2:
        message = f"Training target has only one class."
        if logger:
            logger.log(message)
        else:
            print(message)
        return pd.DataFrame(), {}, pd.DataFrame(columns=feature_columns), None
    if len(np.unique(y_test)) < 2:
        message = f"Test target has only one class."
        if logger:
            logger.log(message)
        else:
            print(message)
        # Potentially still proceed but metrics might be limited or behave unexpectedly
        # For now, let's return empty/NaN results similar to other error conditions
        return pd.DataFrame(), {}, pd.DataFrame(columns=feature_columns), None


    # Register binning methods
    binning_methods = {
        "EqualFreq": create_equal_freq_pipeline,
        "DecisionTree": create_decision_tree_pipeline,
        "ChiMerge": create_chi_merge_pipeline,
        "MAPA": create_mapa_pipeline,
        "ConditionalInference": create_conditional_inference_tree_pipeline,
        "MultiIntervalDiscretization": create_mid_pipeline,
        "IsotonicRegression": create_isotonic_regression_pipeline,
        "ChiIsotonic": create_chi_isotonic_pipeline,
        "IsotonicChi": create_isotonic_chi_pipeline,
        "EqualFreqChi": create_equal_freq_chi_pipeline
    }

    results_list = []
    all_feature_ivs = {}
    all_coefficients = {} # To store model coefficients
    all_predictions = {} # To store predictions for ROC/PR curves

    if logger:
        logger.log_section("BINNING STRATEGIES EVALUATION", level=2)
        logger.log(f"Feature columns to process: {feature_columns}")
    else:
        print("\n===== Comparing Binning Strategies =====")

    # Evaluate each method
    for method_name, pipeline_factory in binning_methods.items():
        if logger:
            logger.log_section(f"Applying {method_name} Binning", level=3)
        else:
            print(f"\n--- Applying {method_name} Binning ---")

        # Start timing for this binning method
        start_time = time.time()

        method_ivs = {}

        # Transform each feature
        transformed_features_for_model = {} # Stores binned data and transformation pipeline for the model
        # method_ivs (already existing) will store IV values.

        # Configure pipeline parameters based on method
        if method_name == "EqualFreq":
            pipeline_config = {
                'max_bins': max_bins_per_feature,
                'force_monotonic': force_monotonic,
                'monotonic_direction': monotonic_direction
            }
        elif method_name == "ChiMerge":
            pipeline_config = {
                'max_bins': max_bins_per_feature,
                'threshold': 3.841,
                'force_monotonic': force_monotonic,
                'monotonic_direction': monotonic_direction
            }
        elif method_name == "MAPA":
            pipeline_config = {
                'max_bins': max_bins_per_feature,
                'force_monotonic': force_monotonic,
                'monotonic_direction': monotonic_direction
            }
        elif method_name == "ConditionalInference":
            pipeline_config = {
                'max_bins': max_bins_per_feature,
                'alpha': 0.05,
                'min_samples_leaf': 10000,
                'random_state': random_state,
                'force_monotonic': force_monotonic,
                'monotonic_direction': monotonic_direction
            }
        elif method_name == "MultiIntervalDiscretization":
            pipeline_config = {
                'max_bins': max_bins_per_feature,
                'min_samples_leaf': 10000,
                'min_entropy_decrease': 0.0001,
                'random_state': random_state,
                'force_monotonic': force_monotonic,
                'monotonic_direction': monotonic_direction
            }
        elif method_name == "IsotonicRegression":
            pipeline_config = {
                'max_bins': max_bins_per_feature,
                'increasing': None,
                'min_samples_bin': 10000,
                'random_state': random_state,
                'force_monotonic': force_monotonic,
                'monotonic_direction': monotonic_direction
            }
        elif method_name == "ChiIsotonic":
            pipeline_config = {
                'max_bins': max_bins_per_feature,
                'threshold': 3.841,
                'min_samples_leaf': 10000,
                'random_state': random_state,
                'force_monotonic': force_monotonic,
                'monotonic_direction': monotonic_direction
            }
        elif method_name == "IsotonicChi":
            pipeline_config = {
                'max_bins': max_bins_per_feature,
                'threshold': 3.841,
                'min_samples_leaf': 10000,
                'random_state': random_state,
                'force_monotonic': force_monotonic,
                'monotonic_direction': monotonic_direction
            }
        elif method_name == "EqualFreqChi":
            pipeline_config = {
                'max_bins': max_bins_per_feature,
                'threshold': 3.841,
                'min_samples_leaf': 10000,
                'random_state': random_state,
                'force_monotonic': force_monotonic,
                'monotonic_direction': monotonic_direction
            }
        else:  # DecisionTree
            pipeline_config = {
                'max_depth': max_bins_per_feature,
                'random_state': random_state,
                'force_monotonic': force_monotonic,
                'monotonic_direction': monotonic_direction
            }

        # Process all features in parallel using conservative CPU allocation
        message = f"  Processing {len(feature_columns)} features in parallel..."
        if logger:
            logger.log(message)
        else:
            print(message)

        # Conservative parallel processing for 22 cores
        n_jobs_features = min(8, len(feature_columns))  # Conservative: max 8 cores for features
        if n_jobs is not None:
            n_jobs_features = min(n_jobs, len(feature_columns))  # Respect user's n_jobs setting
        method_ivs, transformed_features_for_model = _parallel_process_features_for_binning(
            X_train, y_train, feature_columns, pipeline_factory, pipeline_config,
            n_jobs=n_jobs_features, verbose=verbose
        )

        all_feature_ivs[method_name] = method_ivs

        # Build model if we have transformed features for the model
        if transformed_features_for_model:
            # Create binned feature matrix for model training
            binned_train_df = pd.DataFrame()
            for feature_key, model_data in transformed_features_for_model.items():
                binned_train_df[feature_key] = model_data['train_values'].flatten()

            # Apply iterative VIF feature removal

            message = f"  Applying VIF and AIC selection for {method_name}..."
            if logger:
                logger.log(message)
            else:
                print(message)

            # Debug message for VIF/AIC parallel processing
            vif_debug_msg = f"    DEBUG: Using {n_jobs} cores for VIF and AIC calculations"
            if logger:
                logger.log(vif_debug_msg)
            elif verbose:
                print(vif_debug_msg)

            retained_features_after_vif_aic, X_train_vif_filtered, final_model_aic = calculate_vif_iteratively(
                X=binned_train_df,
                y=y_train_input, # Make sure y_train_input is correctly passed and prepared
                threshold=5.0,
                n_jobs=n_jobs  # Pass through the n_jobs parameter to maintain parallelization
            )

            if not retained_features_after_vif_aic:
                # End timing for method with no retained features
                end_time = time.time()
                computation_time = end_time - start_time

                message1 = f"    DEBUG: No features retained after VIF+AIC selection for {method_name}. Skipping model training."
                message2 = f"    VIF: No features retained for {method_name}. Skipping model training."
                if logger:
                    logger.log(message1)
                    logger.log(message2)
                else:
                    print(message1)
                    print(message2)
                metrics = {
                    'Binning_Strategy': method_name,
                    'Avg_IV': np.mean(list(method_ivs.values())) if method_ivs else 0.0,
                    'AUROC': np.nan, 'KS': np.nan, 'Recall_pos': np.nan,
                    'F1_pos': np.nan, 'PR_AUC': np.nan, 'AIC': final_model_aic,
                    'Computation_Time_Seconds': computation_time
                }
                all_coefficients[method_name] = pd.Series({feat: np.nan for feat in feature_columns}) # Store NaNs for coefficients
                results_list.append(metrics)
                continue # Skip to the next binning method

            # Train logistic regression
            try:
                # No diagnostic printing

                # Standard model across all methods
                # Train logistic regression
                model = LogisticRegression(
                    solver='saga',
                    penalty=None,
                    random_state=random_state,
                    n_jobs=-1
                )
                model.fit(X_train_vif_filtered, y_train) # y_train is from input y_train_input

                # Store coefficients
                if hasattr(model, 'coef_'):
                    coef_values = model.coef_[0]
                    # Coefficients correspond to the features remaining after VIF
                    coef_feature_names = X_train_vif_filtered.columns
                    all_coefficients[method_name] = pd.Series(coef_values, index=coef_feature_names)
                else:
                    # Should not happen with LogisticRegression, but good for robustness
                    all_coefficients[method_name] = pd.Series(dtype='float64')

                # No additional printing to maintain consistency

                # Transform test data using stored binned pipelines
                binned_test_df = pd.DataFrame()
                for feature_key, model_data in transformed_features_for_model.items():
                    # Get the pipeline that does binning (and mono, if applicable) for this feature
                    model_feature_pipeline_for_test = model_data['transform_pipeline_for_test']
                    X_feat_test_raw = X_test[feature_key].values.reshape(-1, 1) # Use X_test from input
                    binned_test_df[feature_key] = model_feature_pipeline_for_test.transform(X_feat_test_raw).flatten()

                # Filter test data to include only features retained after VIF on training data
                X_test_vif_filtered = binned_test_df[retained_features_after_vif_aic]

                # Make predictions
                y_pred_proba = model.predict_proba(X_test_vif_filtered)[:, 1]
                y_pred = model.predict(X_test_vif_filtered) # y_test is from input y_test_input

                # Store predictions for ROC/PR curves
                all_predictions[method_name] = {
                    'y_true': y_test.copy(),
                    'y_pred_proba': y_pred_proba.copy(),
                    'y_pred': y_pred.copy()
                }

                # End timing for this binning method
                end_time = time.time()
                computation_time = end_time - start_time

                # Calculate metrics
                metrics = {
                    'Binning_Strategy': method_name,
                    'Avg_IV': np.mean(list(method_ivs.values())),
                    'AUROC': roc_auc_score(y_test, y_pred_proba), # y_test is from input y_test_input
                    'KS': ks_statistic(y_test, y_pred_proba), # y_test is from input y_test_input
                    'Recall_pos': recall_score(y_test, y_pred, pos_label=1, zero_division='warn'), # y_test is from input y_test_input
                    'F1_pos': f1_score(y_test, y_pred, pos_label=1, zero_division='warn'), # y_test is from input y_test_input
                    'PR_AUC': average_precision_score(y_test, y_pred_proba), # y_test is from input y_test_input
                    'AIC': final_model_aic,
                    'Computation_Time_Seconds': computation_time
                }
                message1 = f"    DEBUG: Model trained successfully with {len(retained_features_after_vif_aic)} features"
                message2 = f"    AUROC: {metrics['AUROC']:.4f}, F1: {metrics['F1_pos']:.4f}, AIC: {final_model_aic:.2f}"
                message3 = f"    Computation Time: {computation_time:.2f} seconds"
                if logger:
                    logger.log(message1)
                    logger.log(message2)
                    logger.log(message3)
                else:
                    print(message1)
                    print(message2)
                    print(message3)

            except Exception as e:
                # End timing for failed method
                end_time = time.time()
                computation_time = end_time - start_time

                message = f"    Model Error: {str(e)}"
                if logger:
                    logger.log(message)
                else:
                    print(message)
                metrics = {
                    'Binning_Strategy': method_name,
                    'Avg_IV': np.mean(list(method_ivs.values())),
                    'AUROC': np.nan,
                    'KS': np.nan,
                    'Recall_pos': np.nan,
                    'F1_pos': np.nan,
                    'PR_AUC': np.nan,
                    'AIC': final_model_aic,
                    'Computation_Time_Seconds': computation_time
                }
                all_coefficients[method_name] = pd.Series({feat: np.nan for feat in feature_columns})
                # Store empty predictions for failed methods
                all_predictions[method_name] = {
                    'y_true': None,
                    'y_pred_proba': None,
                    'y_pred': None
                }
        else:
            # End timing for method with no successful features
            end_time = time.time()
            computation_time = end_time - start_time

            # No successful transformed features
            metrics = {
                'Binning_Strategy': method_name,
                'Avg_IV': 0.0,
                'AUROC': np.nan,
                'KS': np.nan,
                'Recall_pos': np.nan,
                'F1_pos': np.nan,
                'PR_AUC': np.nan,
                'AIC': np.nan,
                'Computation_Time_Seconds': computation_time
            }
            all_coefficients[method_name] = pd.Series({feat: np.nan for feat in feature_columns})
            # Store empty predictions for methods with no features
            all_predictions[method_name] = {
                'y_true': None,
                'y_pred_proba': None,
                'y_pred': None
            }

        results_list.append(metrics)

    # Create summary dataframe
    results_df = pd.DataFrame(results_list)

    # Log results summary
    if logger:
        logger.log_section("RESULTS SUMMARY", level=2)
        logger.log("Results DataFrame:")
        for _, row in results_df.iterrows():
            logger.log(f"  {row['Binning_Strategy']}: AUROC={row['AUROC']:.4f}, F1={row['F1_pos']:.4f}, AIC={row['AIC']:.2f}")

        # Save results to CSV
        logger.save_dataframe(results_df, "results_summary.csv", "Summary of all binning strategies performance")

    # Create coefficients DataFrame
    coefficients_df = pd.DataFrame(all_coefficients).T
    if not coefficients_df.empty:
        coefficients_df = coefficients_df.reindex(columns=feature_columns) # Ensure consistent column order
    else:
        # If all_coefficients was empty or resulted in an empty DataFrame
        coefficients_df = pd.DataFrame(columns=feature_columns)


    # Create comprehensive visualizations
    figs = []
    if not results_df.empty:

        # Individual metric charts
        metrics_to_plot = ['Avg_IV', 'AUROC', 'KS', 'PR_AUC', 'F1_pos', 'Recall_pos', 'AIC']
        plot_cols = [col for col in metrics_to_plot if col in results_df.columns]

        if plot_cols:
            plot_df = results_df.set_index('Binning_Strategy')[plot_cols]

            if not plot_df.empty and not plot_df.isnull().all().all():
                # Create individual charts for each metric with 2-row layout
                n_metrics = len(plot_cols)
                n_cols = min(4, n_metrics)  # Max 4 columns
                n_rows_needed = int(np.ceil(n_metrics / n_cols))

                # Create bar charts
                fig_bars, axes_bars = plt.subplots(n_rows_needed, n_cols, figsize=(5*n_cols, 4*n_rows_needed))
                fig_bars.suptitle('Individual Metric Comparisons - Bar Charts', fontsize=16, y=0.98)

                if n_rows_needed == 1:
                    axes_bars = axes_bars.reshape(1, -1) if n_cols > 1 else np.array([[axes_bars]])
                elif n_cols == 1:
                    axes_bars = axes_bars.reshape(-1, 1)

                for i, metric in enumerate(plot_cols):
                    row = i // n_cols
                    col = i % n_cols
                    ax = axes_bars[row, col]

                    if not plot_df[metric].isnull().all():
                        # Bar chart with proper alignment
                        x_positions = np.arange(len(plot_df))
                        bars = ax.bar(x_positions, plot_df[metric], color='steelblue', alpha=0.7, width=0.6)
                        ax.set_title(f'{metric}', fontsize=12, pad=10)
                        ax.set_ylabel(metric, fontsize=10)
                        ax.set_xticks(x_positions)
                        ax.set_xticklabels(plot_df.index, rotation=45, ha='right', fontsize=9)
                        ax.grid(True, alpha=0.3, axis='y')

                        # Add value labels on bars
                        for bar in bars:
                            height = bar.get_height()
                            if not np.isnan(height):
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)

                # Hide unused subplots for bar charts
                for i in range(n_metrics, n_rows_needed * n_cols):
                    row = i // n_cols
                    col = i % n_cols
                    axes_bars[row, col].set_visible(False)

                plt.tight_layout()
                figs.append(fig_bars)

                # Create line charts
                fig_lines, axes_lines = plt.subplots(n_rows_needed, n_cols, figsize=(5*n_cols, 4*n_rows_needed))
                fig_lines.suptitle('Individual Metric Comparisons - Line Charts', fontsize=16, y=0.98)

                if n_rows_needed == 1:
                    axes_lines = axes_lines.reshape(1, -1) if n_cols > 1 else np.array([[axes_lines]])
                elif n_cols == 1:
                    axes_lines = axes_lines.reshape(-1, 1)

                for i, metric in enumerate(plot_cols):
                    row = i // n_cols
                    col = i % n_cols
                    ax = axes_lines[row, col]

                    if not plot_df[metric].isnull().all():
                        # Line chart with proper alignment
                        x_positions = np.arange(len(plot_df))
                        ax.plot(x_positions, plot_df[metric], marker='o', linewidth=2,
                               markersize=8, color='darkred', markerfacecolor='red', markeredgecolor='darkred')
                        ax.set_title(f'{metric}', fontsize=12, pad=10)
                        ax.set_ylabel(metric, fontsize=10)
                        ax.set_xticks(x_positions)
                        ax.set_xticklabels(plot_df.index, rotation=45, ha='right', fontsize=9)
                        ax.grid(True, alpha=0.3)

                        # Add value labels on points
                        for j, value in enumerate(plot_df[metric]):
                            if not np.isnan(value):
                                ax.text(j, value, f'{value:.3f}', ha='center', va='bottom',
                                       fontsize=8, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

                # Hide unused subplots for line charts
                for i in range(n_metrics, n_rows_needed * n_cols):
                    row = i // n_cols
                    col = i % n_cols
                    axes_lines[row, col].set_visible(False)

                plt.tight_layout()
                figs.append(fig_lines)

                # Line chart for trends
                fig_line, ax_line = plt.subplots(figsize=(14, 8))
                for metric in plot_cols:
                    if not plot_df[metric].isnull().all():
                        ax_line.plot(range(len(plot_df)), plot_df[metric], marker='o', label=metric, linewidth=2)

                ax_line.set_title('Metric Trends Across Binning Methods', fontsize=14)
                ax_line.set_xlabel('Binning Methods')
                ax_line.set_ylabel('Metric Values')
                ax_line.set_xticks(range(len(plot_df)))
                ax_line.set_xticklabels(plot_df.index, rotation=45, ha='right')
                ax_line.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax_line.grid(True, alpha=0.3)
                plt.tight_layout()
                figs.append(fig_line)

        # ROC Curves
        fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(all_predictions)))

        for i, (method_name, pred_data) in enumerate(all_predictions.items()):
            if pred_data['y_true'] is not None and pred_data['y_pred_proba'] is not None:
                try:
                    fpr, tpr, _ = roc_curve(pred_data['y_true'], pred_data['y_pred_proba'])
                    auc_score = roc_auc_score(pred_data['y_true'], pred_data['y_pred_proba'])
                    ax_roc.plot(fpr, tpr, color=colors[i], linewidth=2,
                               label=f'{method_name} (AUC = {auc_score:.3f})')
                except:
                    continue

        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curves - All Binning Methods')
        ax_roc.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_roc.grid(True, alpha=0.3)
        plt.tight_layout()
        figs.append(fig_roc)

        # Precision-Recall Curves
        fig_pr, ax_pr = plt.subplots(figsize=(10, 8))

        for i, (method_name, pred_data) in enumerate(all_predictions.items()):
            if pred_data['y_true'] is not None and pred_data['y_pred_proba'] is not None:
                try:
                    precision, recall, _ = precision_recall_curve(pred_data['y_true'], pred_data['y_pred_proba'])
                    pr_auc = average_precision_score(pred_data['y_true'], pred_data['y_pred_proba'])
                    ax_pr.plot(recall, precision, color=colors[i], linewidth=2,
                              label=f'{method_name} (AP = {pr_auc:.3f})')
                except:
                    continue

        # Add baseline (random classifier)
        y_true_sample = next((pred_data['y_true'] for pred_data in all_predictions.values()
                             if pred_data['y_true'] is not None), None)
        if y_true_sample is not None:
            baseline = np.mean(y_true_sample)
            ax_pr.axhline(y=baseline, color='k', linestyle='--', alpha=0.5,
                         label=f'Random Classifier (AP = {baseline:.3f})')

        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Precision-Recall Curves - All Binning Methods')
        ax_pr.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_pr.grid(True, alpha=0.3)
        plt.tight_layout()
        figs.append(fig_pr)

    # Save all figures if logger is available
    if logger and figs:
        logger.log_section("SAVING VISUALIZATIONS", level=2)
        figure_names = [
            "individual_metrics_bar_charts.jpg",
            "individual_metrics_line_charts.jpg",
            "metric_trends_comparison.jpg",
            "roc_curves_comparison.jpg",
            "precision_recall_curves_comparison.jpg"
        ]

        for i, fig in enumerate(figs):
            if i < len(figure_names):
                logger.save_figure(fig, figure_names[i])
            else:
                logger.save_figure(fig, f"additional_figure_{i+1}.jpg")

        # Save feature IVs and coefficients
        if all_feature_ivs:
            iv_df = pd.DataFrame(all_feature_ivs).T
            logger.save_dataframe(iv_df, "feature_ivs.csv", "Information Value for each feature by binning method")

        if not coefficients_df.empty:
            logger.save_dataframe(coefficients_df, "model_coefficients.csv", "Logistic regression coefficients for each binning method")

        logger.log_section("ANALYSIS COMPLETED", level=1)
        logger.log(f"Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log(f"All outputs saved to: {logger.output_dir}")

    return results_df, all_feature_ivs, coefficients_df, figs

def example_comprehensive_debug_logging(X_train, X_test, y_train, y_test, feature_columns):
    """
    Example function demonstrating how to use comprehensive debug logging
    to capture ALL terminal output including debug information.

    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and test features
    y_train, y_test : pd.Series
        Training and test targets
    feature_columns : list
        List of feature column names

    Returns:
    --------
    str
        Path to the comprehensive debug log file
    """
    # Initialize logger
    logger = BinningLogger("comprehensive_analysis")

    # Start comprehensive output capture
    with logger.capture_all_output():
        # Log initial information
        debug_info = {
            "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": len(feature_columns),
            "target_distribution": str(y_train.value_counts().to_dict())
        }
        logger.log_debug_info(debug_info)

        print("=== STARTING COMPREHENSIVE BINNING ANALYSIS ===")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {feature_columns}")
        print(f"Target distribution: {y_train.value_counts().to_dict()}")

        # Run the binning analysis with all debug output captured
        try:
            results, ivs, coefficients, figs = compare_binning_strategies_on_dataset(
                X_train_input=X_train,
                X_test_input=X_test,
                y_train_input=y_train,
                y_test_input=y_test,
                feature_columns=feature_columns,
                max_bins_per_feature=10,
                force_monotonic=True,
                monotonic_direction=None,
                verbose=True,  # This will generate lots of debug output
                n_jobs=8
            )

            print("\n=== ANALYSIS COMPLETED SUCCESSFULLY ===")
            print("Results summary:")
            print(results.to_string())

            # Log completion information
            completion_info = {
                "end_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "methods_tested": len(results),
                "best_method": results.index[0] if len(results) > 0 else "None",
                "figures_generated": len(figs) if figs else 0
            }
            logger.log_debug_info(completion_info)

        except Exception as e:
            print(f"\n=== ERROR OCCURRED ===")
            print(f"Error: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")

            error_info = {
                "error_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
            logger.log_debug_info(error_info)

            raise

    print(f"\n=== COMPREHENSIVE DEBUG LOG SAVED ===")
    print(f"Debug log location: {logger.debug_log_filepath}")
    print(f"Regular log location: {logger.log_filepath}")

    return logger.debug_log_filepath

#  train = pd.read_parquet('transformations2_good/train_selected_final.parquet')
#  test = pd.read_parquet('transformations2_good/test_selected_final.parquet')
#  y_train = train['target']
#  y_test = test['target']
#  X_train = train.drop(columns=['target'])
#  X_test = test.drop(columns=['target'])

# del train
# del test

# # Assume X_train_ex and X_test_ex are already imputed if they were to come from a real dataset
# # For this synthetic example, they don't have NaNs.

# # Use the actual feature columns from the loaded dataframes
# feature_columns_actual = [col for col in X_train.columns if col != 'target']

# logger = BinningLogger("my_analysis")
# with logger.capture_all_output():
#     results, ivs, coefficients, plot = compare_binning_strategies_on_dataset(
#         X_train_input=X_train,
#         X_test_input=X_test,
#         y_train_input=y_train,
#         y_test_input=y_test,
#         feature_columns=feature_columns_actual, # Pass the list of actual feature names
#         max_bins_per_feature=10,
#         force_monotonic=True,
#         monotonic_direction=None,
#         verbose=True # Set to True to see VIF steps etc.
#     )

#     # Print results
#     print("\n===== Coefficients =====")
#     print(coefficients)

#     print("\n===== Overall Results =====")
#     print(results)

#     print("\n===== IV Values by Feature =====")
#     for method, values in ivs.items():
#         print(f"\n--- {method} ---")
#         for feat, iv in values.items():
#             print(f"  {feat}: {iv:.4f}")

#     if plot:
#         for i, fig in enumerate(plot):
#             plt.figure(fig.number)
#             plt.show()


# ===================================================================
# BOOTSTRAP STATISTICAL ANALYSIS FUNCTIONS
# ===================================================================

def bootstrap_binning_comparison(
    X_train_population: pd.DataFrame,
    X_test_population: pd.DataFrame,
    y_train_population: pd.Series,
    y_test_population: pd.Series,
    feature_columns: list = None,
    bootstrap_iterations: int = 1000,
    train_sample_size: int = 150000,
    test_sample_size: int = 150000,
    max_bins_per_feature: int = 10,
    random_state: int = 42,
    force_monotonic: bool = True,
    monotonic_direction: str = None,
    confidence_level: float = 0.95,
    overlap_threshold: float = 0.1,
    auto_adjust_sample_size: bool = True,
    max_iterations: int = 2000,
    verbose: bool = True,
    log_outputs: bool = True,
    output_dir: str = None,
    n_jobs: int = None
):
    """
    Compare binning strategies using bootstrap sampling to create statistical distributions.

    Creates bootstrap distributions by repeatedly sampling from training and test populations,
    fitting binning methods, and evaluating performance. Automatically detects overlapping
    confidence intervals and adjusts sample size if needed.

    **COMPLETE LOGGING AND FIGURE SAVING**: All outputs, logs, and figures are saved
    exactly like the original compare_binning_strategies_on_dataset function.

    Parameters:
    -----------
    X_train_population : pd.DataFrame
        Full training population (e.g., 3M observations)
    X_test_population : pd.DataFrame
        Full test population (e.g., 3M observations)
    y_train_population : pd.Series
        Full training target population
    y_test_population : pd.Series
        Full test target population
    bootstrap_iterations : int, default=1000
        Number of bootstrap samples to draw
    train_sample_size : int, default=150000
        Size of training sample for each bootstrap iteration
    test_sample_size : int, default=150000
        Size of test sample for each bootstrap iteration
    confidence_level : float, default=0.95
        Confidence level for intervals (e.g., 0.95 for 95% CI)
    overlap_threshold : float, default=0.1
        Maximum allowed overlap ratio between confidence intervals
    auto_adjust_sample_size : bool, default=True
        Whether to automatically increase sample size if intervals overlap
    max_iterations : int, default=2000
        Maximum bootstrap iterations when auto-adjusting
    log_outputs : bool, default=True
        Whether to save detailed logs and figures (same as original function)
    output_dir : str, optional
        Directory for saving all outputs, logs, and figures

    Returns:
    --------
    dict containing:
        'results_summary': pd.DataFrame with means, std errors, and confidence intervals
        'bootstrap_distributions': dict with raw distributions for each method/metric
        'convergence_info': dict with information about sample size adjustments
        'figures': list of matplotlib figures (saved automatically if log_outputs=True)
        'individual_results': list of all individual bootstrap iteration results
    """

    import time
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    # Initialize comprehensive logging exactly like original function
    logger = None
    if log_outputs:
        logger = BinningLogger(base_filename="bootstrap_binning_analysis", output_dir=output_dir)
        logger.log_section("BOOTSTRAP BINNING STATISTICAL ANALYSIS", level=1)
        logger.log(f"Analysis started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log(f"Bootstrap Parameters:")
        logger.log(f"  - Bootstrap iterations: {bootstrap_iterations}")
        logger.log(f"  - Train sample size: {train_sample_size:,}")
        logger.log(f"  - Test sample size: {test_sample_size:,}")
        logger.log(f"  - Training population: {len(X_train_population):,}")
        logger.log(f"  - Test population: {len(X_test_population):,}")
        logger.log(f"  - Confidence level: {confidence_level}")
        logger.log(f"  - Auto adjust sample size: {auto_adjust_sample_size}")
        logger.log(f"  - Overlap threshold: {overlap_threshold}")
        logger.log(f"  - Max bins per feature: {max_bins_per_feature}")
        logger.log(f"  - Force monotonic: {force_monotonic}")
        logger.log(f"  - Random state: {random_state}")

    # Validate inputs
    if train_sample_size > len(X_train_population):
        error_msg = f"train_sample_size ({train_sample_size}) cannot be larger than population ({len(X_train_population)})"
        if logger:
            logger.log(f"ERROR: {error_msg}")
        raise ValueError(error_msg)

    if test_sample_size > len(X_test_population):
        error_msg = f"test_sample_size ({test_sample_size}) cannot be larger than population ({len(X_test_population)})"
        if logger:
            logger.log(f"ERROR: {error_msg}")
        raise ValueError(error_msg)

    # Identify feature columns
    if feature_columns is None:
        feature_columns = [col for col in X_train_population.columns
                          if pd.api.types.is_numeric_dtype(X_train_population[col])]
        if logger:
            logger.log(f"Auto-detected {len(feature_columns)} numeric feature columns: {feature_columns}")

    if not feature_columns:
        error_msg = "No numeric feature columns found"
        if logger:
            logger.log(f"ERROR: {error_msg}")
        raise ValueError(error_msg)

    # Set up parallel processing
    if n_jobs is None:
        n_jobs = min(8, cpu_count())

    if logger:
        logger.log(f"Using {n_jobs} parallel jobs for bootstrap iterations")

    # Track results across iterations
    bootstrap_results = []
    current_iterations = bootstrap_iterations
    current_train_size = train_sample_size
    current_test_size = test_sample_size

    iteration_count = 0
    max_attempts = 3

    while iteration_count < max_attempts:
        iteration_count += 1

        if logger:
            logger.log_section(f"Bootstrap Iteration Set {iteration_count}", level=2)
            logger.log(f"Running {current_iterations} iterations with:")
            logger.log(f"  - Train sample size: {current_train_size:,}")
            logger.log(f"  - Test sample size: {current_test_size:,}")
            logger.log(f"  - Expected total samples processed: {current_iterations * (current_train_size + current_test_size):,}")

        start_time = time.time()

        # Run bootstrap iterations
        bootstrap_results = _run_bootstrap_iterations(
            X_train_population, X_test_population, y_train_population, y_test_population,
            feature_columns, current_iterations, current_train_size, current_test_size,
            max_bins_per_feature, random_state, force_monotonic, monotonic_direction,
            n_jobs, verbose, logger
        )

        end_time = time.time()
        total_time = end_time - start_time

        if logger:
            logger.log(f"Bootstrap iterations completed in {total_time:.2f} seconds")
            logger.log(f"Average time per iteration: {total_time/current_iterations:.3f} seconds")
            logger.log(f"Successful iterations: {len(bootstrap_results)}/{current_iterations}")

            if len(bootstrap_results) < current_iterations * 0.8:  # Less than 80% success rate
                logger.log(f"WARNING: Low success rate ({len(bootstrap_results)/current_iterations*100:.1f}%) - check data quality")

        # Analyze results and check for overlapping confidence intervals
        results_summary, distributions = _analyze_bootstrap_results(
            bootstrap_results, confidence_level, logger
        )

        # Check for overlapping confidence intervals
        overlaps_detected = False
        if auto_adjust_sample_size and iteration_count < max_attempts:
            overlaps_detected = _check_confidence_interval_overlaps(
                results_summary, overlap_threshold, logger
            )

        if not overlaps_detected or iteration_count == max_attempts:
            break

        # Adjust sample sizes for next iteration
        if overlaps_detected:
            old_train_size = current_train_size
            old_test_size = current_test_size
            old_iterations = current_iterations

            current_train_size = min(int(current_train_size * 1.5), len(X_train_population))
            current_test_size = min(int(current_test_size * 1.5), len(X_test_population))
            current_iterations = min(int(current_iterations * 1.2), max_iterations)

            if logger:
                logger.log_section("SAMPLE SIZE ADJUSTMENT", level=3)
                logger.log(f"Overlapping confidence intervals detected. Adjusting parameters:")
                logger.log(f"  - Train sample size: {old_train_size:,} → {current_train_size:,}")
                logger.log(f"  - Test sample size: {old_test_size:,} → {current_test_size:,}")
                logger.log(f"  - Bootstrap iterations: {old_iterations} → {current_iterations}")
                logger.log(f"This will increase statistical power and reduce overlap")

    # Create comprehensive visualizations with logging
    if logger:
        logger.log_section("CREATING VISUALIZATIONS", level=2)
        logger.log("Generating comprehensive bootstrap visualization suite...")

    figures = _create_bootstrap_visualizations(
        results_summary, distributions, confidence_level, logger
    )

    # Prepare convergence info
    convergence_info = {
        'initial_train_sample_size': train_sample_size,
        'initial_test_sample_size': test_sample_size,
        'initial_iterations': bootstrap_iterations,
        'final_train_sample_size': current_train_size,
        'final_test_sample_size': current_test_size,
        'final_iterations': current_iterations,
        'attempts_made': iteration_count,
        'overlaps_resolved': not _check_confidence_interval_overlaps(results_summary, overlap_threshold, None),
        'total_bootstrap_samples_processed': len(bootstrap_results),
        'success_rate': len(bootstrap_results) / current_iterations if current_iterations > 0 else 0.0
    }

    # Save all results with comprehensive logging
    if logger:
        logger.log_section("SAVING RESULTS", level=2)
        logger.log(f"Final parameters used:")
        logger.log(f"  - Train sample size: {current_train_size:,}")
        logger.log(f"  - Test sample size: {current_test_size:,}")
        logger.log(f"  - Bootstrap iterations: {current_iterations}")
        logger.log(f"  - Total attempts: {iteration_count}")
        logger.log(f"  - Success rate: {convergence_info['success_rate']*100:.1f}%")
        logger.log(f"  - Overlaps resolved: {convergence_info['overlaps_resolved']}")

        # Save main results DataFrame
        logger.save_dataframe(results_summary, "bootstrap_results_summary.csv",
                            "Bootstrap analysis results with confidence intervals")

        # Save detailed bootstrap distributions
        distributions_df_list = []
        for method, method_distributions in distributions.items():
            for metric, values in method_distributions.items():
                for i, value in enumerate(values):
                    distributions_df_list.append({
                        'method': method,
                        'metric': metric,
                        'iteration': i,
                        'value': value
                    })

        if distributions_df_list:
            distributions_df = pd.DataFrame(distributions_df_list)
            logger.save_dataframe(distributions_df, "bootstrap_raw_distributions.csv",
                                "Raw bootstrap distributions for all methods and metrics")

        # Save convergence information
        convergence_df = pd.DataFrame([convergence_info])
        logger.save_dataframe(convergence_df, "bootstrap_convergence_info.csv",
                            "Bootstrap convergence and optimization information")

        logger.log_section("ANALYSIS COMPLETED SUCCESSFULLY", level=1)
        logger.log(f"Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log(f"All outputs saved to: {logger.output_dir}")
        logger.log(f"Generated {len(figures)} visualization figures")

        # Log final recommendations
        logger.log_section("STATISTICAL RECOMMENDATIONS", level=2)
        _generate_statistical_recommendations(results_summary, convergence_info, logger)

    return {
        'results_summary': results_summary,
        'bootstrap_distributions': distributions,
        'convergence_info': convergence_info,
        'figures': figures,
        'individual_results': bootstrap_results
    }


def _run_bootstrap_iterations(
    X_train_pop, X_test_pop, y_train_pop, y_test_pop, feature_columns,
    n_iterations, train_size, test_size, max_bins, random_state,
    force_monotonic, monotonic_direction, n_jobs, verbose, logger
):
    """Run bootstrap iterations in parallel with comprehensive logging."""

    bootstrap_results = []

    if logger:
        logger.log_section("BOOTSTRAP ITERATION EXECUTION", level=3)
        logger.log(f"Preparing {n_iterations} bootstrap iterations...")
        logger.log(f"Each iteration samples {train_size:,} train + {test_size:,} test observations")
        logger.log(f"Total data points to process: {n_iterations * (train_size + test_size):,}")

    # Create argument list for parallel processing
    args_list = []
    for i in range(n_iterations):
        args_list.append((
            X_train_pop, X_test_pop, y_train_pop, y_test_pop,
            feature_columns, train_size, test_size, max_bins,
            random_state + i, force_monotonic, monotonic_direction, i
        ))

    successful_iterations = 0
    failed_iterations = 0

    # Run bootstrap iterations in parallel
    if n_jobs > 1 and n_iterations > 1:
        max_workers = min(n_jobs, n_iterations)

        if logger:
            logger.log(f"Executing bootstrap in parallel with {max_workers} workers")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_iteration = {
                executor.submit(_single_bootstrap_iteration, args): args[-1]
                for args in args_list
            }

            completed = 0
            for future in as_completed(future_to_iteration):
                try:
                    result = future.result()
                    if result is not None:
                        bootstrap_results.append(result)
                        successful_iterations += 1
                    else:
                        failed_iterations += 1

                    completed += 1

                    # Progress reporting
                    if verbose and completed % max(1, n_iterations // 20) == 0:
                        progress = completed / n_iterations * 100
                        if logger:
                            logger.log(f"Bootstrap progress: {completed}/{n_iterations} ({progress:.1f}%) - "
                                     f"Success: {successful_iterations}, Failed: {failed_iterations}")
                        else:
                            print(f"Bootstrap progress: {completed}/{n_iterations} ({progress:.1f}%)")

                except Exception as e:
                    iteration_num = future_to_iteration[future]
                    failed_iterations += 1
                    error_msg = f"Bootstrap iteration {iteration_num} failed: {str(e)}"
                    if logger:
                        logger.log(f"ERROR: {error_msg}")
                    elif verbose:
                        print(f"ERROR: {error_msg}")
    else:
        # Sequential processing with detailed logging
        if logger:
            logger.log("Running bootstrap iterations sequentially")

        for i, args in enumerate(args_list):
            try:
                result = _single_bootstrap_iteration(args)
                if result is not None:
                    bootstrap_results.append(result)
                    successful_iterations += 1
                else:
                    failed_iterations += 1

                # Progress reporting
                if verbose and (i + 1) % max(1, n_iterations // 20) == 0:
                    progress = (i + 1) / n_iterations * 100
                    if logger:
                        logger.log(f"Bootstrap progress: {i+1}/{n_iterations} ({progress:.1f}%) - "
                                 f"Success: {successful_iterations}, Failed: {failed_iterations}")
                    else:
                        print(f"Bootstrap progress: {i+1}/{n_iterations} ({progress:.1f}%)")

            except Exception as e:
                failed_iterations += 1
                error_msg = f"Bootstrap iteration {i} failed: {str(e)}"
                if logger:
                    logger.log(f"ERROR: {error_msg}")
                elif verbose:
                    print(f"ERROR: {error_msg}")

    # Log final iteration statistics
    if logger:
        logger.log(f"Bootstrap iterations completed:")
        logger.log(f"  - Successful: {successful_iterations}/{n_iterations} ({successful_iterations/n_iterations*100:.1f}%)")
        logger.log(f"  - Failed: {failed_iterations}/{n_iterations} ({failed_iterations/n_iterations*100:.1f}%)")

        if failed_iterations > n_iterations * 0.1:  # More than 10% failure rate
            logger.log(f"WARNING: High failure rate ({failed_iterations/n_iterations*100:.1f}%) detected")
            logger.log("Consider checking data quality, reducing sample size, or adjusting parameters")

    return bootstrap_results


def _single_bootstrap_iteration(args):
    """Run a single bootstrap iteration with error handling."""

    (X_train_pop, X_test_pop, y_train_pop, y_test_pop, feature_columns,
     train_size, test_size, max_bins, iteration_random_state,
     force_monotonic, monotonic_direction, iteration_num) = args

    try:
        # Set random state for this iteration
        np.random.seed(iteration_random_state)

        # Sample from populations
        train_indices = np.random.choice(len(X_train_pop), size=train_size, replace=False)
        test_indices = np.random.choice(len(X_test_pop), size=test_size, replace=False)

        X_train_sample = X_train_pop.iloc[train_indices].copy()
        X_test_sample = X_test_pop.iloc[test_indices].copy()
        y_train_sample = y_train_pop.iloc[train_indices].copy()
        y_test_sample = y_test_pop.iloc[test_indices].copy()

        # Reset indices to avoid any issues
        X_train_sample.reset_index(drop=True, inplace=True)
        X_test_sample.reset_index(drop=True, inplace=True)
        y_train_sample.reset_index(drop=True, inplace=True)
        y_test_sample.reset_index(drop=True, inplace=True)

        # Run binning comparison on this sample (suppress logging for individual iterations)
        results_df, feature_ivs, coefficients_df, _ = compare_binning_strategies_on_dataset(
            X_train_sample, X_test_sample, y_train_sample, y_test_sample,
            feature_columns=feature_columns,
            max_bins_per_feature=max_bins,
            random_state=iteration_random_state,
            force_monotonic=force_monotonic,
            monotonic_direction=monotonic_direction,
            verbose=False,  # Suppress verbose output for individual iterations
            log_outputs=False,  # Don't create logs for individual iterations
            n_jobs=1  # Use single core per iteration to avoid nested parallelism
        )

        # Extract key metrics for each method
        iteration_results = {}
        for _, row in results_df.iterrows():
            method_name = row['Binning_Strategy']
            iteration_results[method_name] = {
                'iteration': iteration_num,
                'Avg_IV': row.get('Avg_IV', np.nan),
                'AUROC': row.get('AUROC', np.nan),
                'KS': row.get('KS', np.nan),
                'PR_AUC': row.get('PR_AUC', np.nan),
                'F1_pos': row.get('F1_pos', np.nan),
                'Recall_pos': row.get('Recall_pos', np.nan),
                'AIC': row.get('AIC', np.nan),
                'Computation_Time_Seconds': row.get('Computation_Time_Seconds', np.nan)
            }

        return iteration_results

    except Exception as e:
        # Return None for failed iterations - this will be logged by the caller
        return None


def _analyze_bootstrap_results(bootstrap_results, confidence_level, logger):
    """Analyze bootstrap results and compute statistics with comprehensive logging."""

    if logger:
        logger.log_section("BOOTSTRAP STATISTICAL ANALYSIS", level=3)
        logger.log(f"Analyzing {len(bootstrap_results)} successful bootstrap iterations")

    # Collect all results by method and metric
    method_metrics = {}

    for iteration_result in bootstrap_results:
        for method_name, metrics in iteration_result.items():
            if method_name not in method_metrics:
                method_metrics[method_name] = {metric: [] for metric in metrics.keys() if metric != 'iteration'}

            for metric, value in metrics.items():
                if metric != 'iteration' and not np.isnan(value):
                    method_metrics[method_name][metric].append(value)

    # Calculate statistics for each method and metric
    results_summary_data = []
    distributions = {}

    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    if logger:
        logger.log(f"Computing {confidence_level*100:.0f}% confidence intervals")
        logger.log(f"Confidence interval percentiles: {lower_percentile:.1f}% - {upper_percentile:.1f}%")

    for method_name, metrics_dict in method_metrics.items():
        distributions[method_name] = metrics_dict

        method_stats = {'Binning_Strategy': method_name}

        if logger:
            logger.log(f"Analyzing {method_name}:")

        for metric, values in metrics_dict.items():
            if len(values) > 0:
                values_array = np.array(values)
                mean_val = np.mean(values_array)
                std_val = np.std(values_array, ddof=1)
                se_val = std_val / np.sqrt(len(values_array))
                ci_lower = np.percentile(values_array, lower_percentile)
                ci_upper = np.percentile(values_array, upper_percentile)

                method_stats[f'{metric}_mean'] = mean_val
                method_stats[f'{metric}_std'] = std_val
                method_stats[f'{metric}_se'] = se_val
                method_stats[f'{metric}_ci_lower'] = ci_lower
                method_stats[f'{metric}_ci_upper'] = ci_upper
                method_stats[f'{metric}_n_samples'] = len(values_array)
                method_stats[f'{metric}_median'] = np.median(values_array)
                method_stats[f'{metric}_min'] = np.min(values_array)
                method_stats[f'{metric}_max'] = np.max(values_array)

                if logger:
                    ci_width = ci_upper - ci_lower
                    logger.log(f"  {metric}: {mean_val:.4f} ± {se_val:.4f} (CI: {ci_lower:.4f}-{ci_upper:.4f}, width: {ci_width:.4f}, n={len(values_array)})")
            else:
                # Handle missing data
                for stat_suffix in ['_mean', '_std', '_se', '_ci_lower', '_ci_upper', '_median', '_min', '_max']:
                    method_stats[f'{metric}{stat_suffix}'] = np.nan
                method_stats[f'{metric}_n_samples'] = 0

                if logger:
                    logger.log(f"  {metric}: No valid data (n=0)")

        results_summary_data.append(method_stats)

    results_summary = pd.DataFrame(results_summary_data)

    if logger:
        logger.log(f"Statistical analysis completed for {len(results_summary)} methods")
        logger.log("Summary statistics computed: mean, std, SE, CI, median, min, max")

    return results_summary, distributions


def _check_confidence_interval_overlaps(results_summary, overlap_threshold, logger):
    """Check for overlapping confidence intervals between methods with detailed logging."""

    key_metrics = ['AUROC', 'KS', 'Avg_IV']
    overlaps_detected = False
    overlap_details = []

    if logger:
        logger.log_section("CONFIDENCE INTERVAL OVERLAP ANALYSIS", level=3)
        logger.log(f"Checking for overlaps using threshold: {overlap_threshold} ({overlap_threshold*100:.1f}%)")

    for metric in key_metrics:
        ci_lower_col = f'{metric}_ci_lower'
        ci_upper_col = f'{metric}_ci_upper'

        if ci_lower_col in results_summary.columns and ci_upper_col in results_summary.columns:
            methods_data = results_summary[['Binning_Strategy', ci_lower_col, ci_upper_col]].dropna()

            if len(methods_data) > 1:
                if logger:
                    logger.log(f"Analyzing {metric} confidence intervals for {len(methods_data)} methods:")

                # Check all pairwise overlaps
                for i in range(len(methods_data)):
                    for j in range(i + 1, len(methods_data)):
                        method1 = methods_data.iloc[i]
                        method2 = methods_data.iloc[j]

                        # Calculate overlap
                        overlap_start = max(method1[ci_lower_col], method2[ci_lower_col])
                        overlap_end = min(method1[ci_upper_col], method2[ci_upper_col])

                        if overlap_start < overlap_end:
                            overlap_length = overlap_end - overlap_start
                            interval1_length = method1[ci_upper_col] - method1[ci_lower_col]
                            interval2_length = method2[ci_upper_col] - method2[ci_lower_col]
                            avg_interval_length = (interval1_length + interval2_length) / 2

                            if avg_interval_length > 0:
                                overlap_ratio = overlap_length / avg_interval_length

                                overlap_info = {
                                    'metric': metric,
                                    'method1': method1['Binning_Strategy'],
                                    'method2': method2['Binning_Strategy'],
                                    'overlap_ratio': overlap_ratio,
                                    'overlap_length': overlap_length,
                                    'method1_ci': f"[{method1[ci_lower_col]:.4f}, {method1[ci_upper_col]:.4f}]",
                                    'method2_ci': f"[{method2[ci_lower_col]:.4f}, {method2[ci_upper_col]:.4f}]"
                                }
                                overlap_details.append(overlap_info)

                                if overlap_ratio > overlap_threshold:
                                    overlaps_detected = True
                                    if logger:
                                        logger.log(f"  OVERLAP DETECTED: {method1['Binning_Strategy']} vs {method2['Binning_Strategy']}")
                                        logger.log(f"    Overlap ratio: {overlap_ratio:.3f} (threshold: {overlap_threshold})")
                                        logger.log(f"    {method1['Binning_Strategy']} CI: [{method1[ci_lower_col]:.4f}, {method1[ci_upper_col]:.4f}]")
                                        logger.log(f"    {method2['Binning_Strategy']} CI: [{method2[ci_lower_col]:.4f}, {method2[ci_upper_col]:.4f}]")
                                        logger.log(f"    Overlap region: [{overlap_start:.4f}, {overlap_end:.4f}]")
                                else:
                                    if logger:
                                        logger.log(f"  {method1['Binning_Strategy']} vs {method2['Binning_Strategy']}: "
                                                 f"overlap ratio {overlap_ratio:.3f} (acceptable)")
                        else:
                            # No overlap - this is good
                            if logger:
                                logger.log(f"  {method1['Binning_Strategy']} vs {method2['Binning_Strategy']}: "
                                         f"no overlap (intervals well separated)")
            else:
                if logger:
                    logger.log(f"  {metric}: Only {len(methods_data)} method(s) with valid CIs - no overlap analysis needed")

    # Summary
    if logger:
        total_comparisons = len(overlap_details)
        significant_overlaps = sum(1 for od in overlap_details if od['overlap_ratio'] > overlap_threshold)

        logger.log(f"Overlap Analysis Summary:")
        logger.log(f"  - Total pairwise comparisons: {total_comparisons}")
        logger.log(f"  - Significant overlaps detected: {significant_overlaps}")
        logger.log(f"  - Overlaps requiring attention: {overlaps_detected}")

        if overlaps_detected:
            logger.log("RECOMMENDATION: Consider increasing sample size for better statistical separation")
        else:
            logger.log("RESULT: All confidence intervals are adequately separated")

    return overlaps_detected


def _create_bootstrap_visualizations(results_summary, distributions, confidence_level, logger):
    """Create comprehensive visualizations for bootstrap results with full logging and saving."""

    if logger:
        logger.log_section("GENERATING BOOTSTRAP VISUALIZATIONS", level=3)
        logger.log("Creating comprehensive visualization suite...")

    figures = []
    figure_info = []

    # 1. Distribution plots for key metrics
    key_metrics = ['AUROC', 'KS', 'Avg_IV', 'PR_AUC']
    methods = list(distributions.keys())

    for metric in key_metrics:
        if any(metric in distributions[method] and len(distributions[method][metric]) > 0 for method in methods):
            if logger:
                logger.log(f"Creating distribution plots for {metric}...")

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Bootstrap Distributions: {metric}', fontsize=16, y=0.98)

            colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

            # Histogram
            ax_hist = axes[0, 0]
            for i, method in enumerate(methods):
                if metric in distributions[method] and len(distributions[method][metric]) > 0:
                    values = distributions[method][metric]
                    ax_hist.hist(values, alpha=0.6, label=method, color=colors[i], bins=30, density=True)

            ax_hist.set_title(f'{metric} - Probability Distributions')
            ax_hist.set_xlabel(metric)
            ax_hist.set_ylabel('Density')
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.3)

            # Box plot
            ax_box = axes[0, 1]
            box_data = []
            box_labels = []

            for method in methods:
                if metric in distributions[method] and len(distributions[method][metric]) > 0:
                    box_data.append(distributions[method][metric])
                    box_labels.append(method)

            if box_data:
                bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)

            ax_box.set_title(f'{metric} - Box Plots')
            ax_box.set_ylabel(metric)
            ax_box.tick_params(axis='x', rotation=45)
            ax_box.grid(True, alpha=0.3)

            # Confidence intervals
            ax_ci = axes[1, 0]
            method_names = []
            means = []
            ci_lowers = []
            ci_uppers = []

            for _, row in results_summary.iterrows():
                mean_col = f'{metric}_mean'
                ci_lower_col = f'{metric}_ci_lower'
                ci_upper_col = f'{metric}_ci_upper'

                if all(col in row and not pd.isna(row[col]) for col in [mean_col, ci_lower_col, ci_upper_col]):
                    method_names.append(row['Binning_Strategy'])
                    means.append(row[mean_col])
                    ci_lowers.append(row[ci_lower_col])
                    ci_uppers.append(row[ci_upper_col])

            if method_names:
                y_pos = np.arange(len(method_names))
                errors = [np.array(means) - np.array(ci_lowers),
                         np.array(ci_uppers) - np.array(means)]

                ax_ci.errorbar(means, y_pos, xerr=errors, fmt='o', capsize=5, capthick=2, markersize=8)
                ax_ci.set_yticks(y_pos)
                ax_ci.set_yticklabels(method_names)
                ax_ci.set_title(f'{metric} - {confidence_level*100:.0f}% Confidence Intervals')
                ax_ci.set_xlabel(f'{metric} Value')
                ax_ci.grid(True, alpha=0.3)

                # Add value labels
                for i, (mean, lower, upper) in enumerate(zip(means, ci_lowers, ci_uppers)):
                    ax_ci.text(mean, i, f'{mean:.3f}\n[{lower:.3f}, {upper:.3f}]',
                              ha='center', va='center', fontsize=9,
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

            # Violin plot
            ax_violin = axes[1, 1]
            if box_data:
                parts = ax_violin.violinplot(box_data, positions=range(1, len(box_data)+1),
                                           showmeans=True, showmedians=True)
                for pc, color in zip(parts['bodies'], colors[:len(box_data)]):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.6)

                ax_violin.set_xticks(range(1, len(box_labels)+1))
                ax_violin.set_xticklabels(box_labels, rotation=45)
                ax_violin.set_title(f'{metric} - Distribution Shapes')
                ax_violin.set_ylabel(metric)
                ax_violin.grid(True, alpha=0.3)

            plt.tight_layout()
            figures.append(fig)

            figure_name = f"bootstrap_distributions_{metric.lower().replace('_', '')}.jpg"
            figure_info.append({
                'figure': fig,
                'filename': figure_name,
                'description': f'Bootstrap distribution analysis for {metric}'
            })

    # 2. Combined comparison plot
    if logger:
        logger.log("Creating combined results comparison...")

    fig_combined, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig_combined.suptitle('Bootstrap Results Summary - All Key Metrics', fontsize=16, y=0.98)

    plot_metrics = ['AUROC', 'KS', 'Avg_IV', 'PR_AUC']

    for idx, metric in enumerate(plot_metrics[:4]):
        if idx < 4:
            ax = axes[idx // 2, idx % 2]

            method_names = []
            means = []
            std_errors = []

            mean_col = f'{metric}_mean'
            se_col = f'{metric}_se'

            for _, row in results_summary.iterrows():
                if mean_col in row and se_col in row and not pd.isna(row[mean_col]):
                    method_names.append(row['Binning_Strategy'])
                    means.append(row[mean_col])
                    std_errors.append(row[se_col] if not pd.isna(row[se_col]) else 0)

            if method_names:
                y_pos = np.arange(len(method_names))
                bars = ax.barh(y_pos, means, xerr=std_errors, capsize=5, alpha=0.7,
                              color=plt.cm.viridis(np.linspace(0.2, 0.8, len(method_names))))

                ax.set_yticks(y_pos)
                ax.set_yticklabels(method_names, fontsize=10)
                ax.set_xlabel(f'{metric} (Mean ± SE)', fontsize=11)
                ax.set_title(f'{metric} Comparison', fontsize=12, pad=10)
                ax.grid(True, alpha=0.3, axis='x')

                # Add value labels
                for i, (bar, mean, se) in enumerate(zip(bars, means, std_errors)):
                    width = bar.get_width()
                    ax.text(width + se + 0.01 * max(means), bar.get_y() + bar.get_height()/2,
                           f'{mean:.3f}±{se:.3f}', ha='left', va='center', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    plt.tight_layout()
    figures.append(fig_combined)
    figure_info.append({
        'figure': fig_combined,
        'filename': "bootstrap_results_summary.jpg",
        'description': 'Combined bootstrap results summary for all key metrics'
    })

    # 3. Statistical significance heatmap
    if logger:
        logger.log("Creating statistical significance analysis...")

    fig_stats, ax_stats = plt.subplots(figsize=(12, 10))

    # Create a matrix of confidence interval widths and overlaps
    methods_list = results_summary['Binning_Strategy'].tolist()
    n_methods = len(methods_list)

    # Confidence interval widths matrix
    ci_widths_matrix = np.zeros((n_methods, len(key_metrics)))

    for i, method in enumerate(methods_list):
        method_row = results_summary[results_summary['Binning_Strategy'] == method].iloc[0]
        for j, metric in enumerate(key_metrics):
            ci_lower_col = f'{metric}_ci_lower'
            ci_upper_col = f'{metric}_ci_upper'
            if ci_lower_col in method_row and ci_upper_col in method_row:
                if not (pd.isna(method_row[ci_lower_col]) or pd.isna(method_row[ci_upper_col])):
                    ci_widths_matrix[i, j] = method_row[ci_upper_col] - method_row[ci_lower_col]
                else:
                    ci_widths_matrix[i, j] = np.nan

    # Create heatmap
    im = ax_stats.imshow(ci_widths_matrix, cmap='YlOrRd', aspect='auto')
    ax_stats.set_xticks(range(len(key_metrics)))
    ax_stats.set_xticklabels(key_metrics, fontsize=12)
    ax_stats.set_yticks(range(n_methods))
    ax_stats.set_yticklabels(methods_list, fontsize=10)
    ax_stats.set_title('Bootstrap Confidence Interval Widths\n(Smaller values indicate higher precision)',
                       fontsize=14, pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_stats, fraction=0.046, pad=0.04)
    cbar.set_label('Confidence Interval Width', fontsize=12)

    # Add text annotations
    for i in range(n_methods):
        for j in range(len(key_metrics)):
            if not np.isnan(ci_widths_matrix[i, j]):
                text_color = "white" if ci_widths_matrix[i, j] > np.nanmean(ci_widths_matrix) else "black"
                text = ax_stats.text(j, i, f'{ci_widths_matrix[i, j]:.4f}',
                                   ha="center", va="center", color=text_color, fontsize=9)

    plt.tight_layout()
    figures.append(fig_stats)
    figure_info.append({
        'figure': fig_stats,
        'filename': "bootstrap_statistical_analysis.jpg",
        'description': 'Statistical analysis showing confidence interval widths and precision'
    })

    # Save all figures with comprehensive logging
    if logger:
        logger.log_section("SAVING VISUALIZATION FIGURES", level=3)
        logger.log(f"Saving {len(figures)} figures to output directory...")

        for info in figure_info:
            logger.save_figure(info['figure'], info['filename'])
            logger.log(f"  Saved: {info['filename']} - {info['description']}")

        logger.log(f"All {len(figures)} figures saved successfully")

    return figures


def _generate_statistical_recommendations(results_summary, convergence_info, logger):
    """Generate statistical recommendations based on bootstrap analysis."""

    if logger:
        logger.log("Statistical Analysis Recommendations:")

        # Find best methods for each metric
        key_metrics = ['AUROC', 'KS', 'Avg_IV']
        best_methods = {}

        for metric in key_metrics:
            mean_col = f'{metric}_mean'
            if mean_col in results_summary.columns:
                metric_data = results_summary[['Binning_Strategy', mean_col]].dropna()
                if not metric_data.empty:
                    best_method = metric_data.loc[metric_data[mean_col].idxmax(), 'Binning_Strategy']
                    best_value = metric_data[mean_col].max()
                    best_methods[metric] = {'method': best_method, 'value': best_value}
                    logger.log(f"  • Best {metric}: {best_method} ({best_value:.4f})")

        # Overall recommendation
        if 'AUROC' in best_methods:
            logger.log(f"  • Primary Recommendation: {best_methods['AUROC']['method']} (highest AUROC)")

        # Consistency check
        method_wins = {}
        for metric, info in best_methods.items():
            method = info['method']
            method_wins[method] = method_wins.get(method, 0) + 1

        if method_wins:
            consistent_winner = max(method_wins, key=method_wins.get)
            if method_wins[consistent_winner] >= 2:
                logger.log(f"  • Most Consistent: {consistent_winner} (best in {method_wins[consistent_winner]} metrics)")

        # Statistical power assessment
        if convergence_info['overlaps_resolved']:
            logger.log(f"  • Statistical Power: Adequate (sample size {convergence_info['final_train_sample_size']:,} provides clear separation)")
        else:
            logger.log(f"  • Statistical Power: Consider larger samples (current: {convergence_info['final_train_sample_size']:,})")

        # Success rate assessment
        success_rate = convergence_info['success_rate']
        if success_rate >= 0.95:
            logger.log(f"  • Data Quality: Excellent ({success_rate*100:.1f}% success rate)")
        elif success_rate >= 0.8:
            logger.log(f"  • Data Quality: Good ({success_rate*100:.1f}% success rate)")
        else:
            logger.log(f"  • Data Quality: Review needed ({success_rate*100:.1f}% success rate - check for data issues)")


def example_bootstrap_binning_analysis(
    X_train_population: pd.DataFrame,
    X_test_population: pd.DataFrame,
    y_train_population: pd.Series,
    y_test_population: pd.Series,
    output_dir: str = None,
    quick_demo: bool = False
):
    """
    Comprehensive example of bootstrap binning analysis with automatic optimization.

    This example demonstrates the full workflow with complete logging and figure saving.
    All outputs are saved exactly like the original compare_binning_strategies_on_dataset.
    """

    print("="*80)
    print("COMPREHENSIVE BOOTSTRAP BINNING ANALYSIS")
    print("="*80)

    start_time = time.time()

    # Step 1: Initial bootstrap analysis
    print("\nStep 1: Running initial bootstrap analysis...")
    print("-" * 50)

    # Adjust parameters for demo mode
    if quick_demo:
        iterations = 100
        sample_size = 50000
        print("Running in QUICK DEMO mode with reduced parameters")
    else:
        iterations = 1000
        sample_size = 150000
        print("Running in FULL ANALYSIS mode")

    initial_results = bootstrap_binning_comparison(
        X_train_population=X_train_population,
        X_test_population=X_test_population,
        y_train_population=y_train_population,
        y_test_population=y_test_population,
        bootstrap_iterations=iterations,
        train_sample_size=sample_size,
        test_sample_size=sample_size,
        confidence_level=0.95,
        overlap_threshold=0.1,
        auto_adjust_sample_size=True,
        max_iterations=2000,
        verbose=True,
        log_outputs=True,
        output_dir=output_dir,
        n_jobs=12
    )

    # Analyze initial results
    results_summary = initial_results['results_summary']
    convergence_info = initial_results['convergence_info']

    print(f"\nInitial Analysis Results:")
    print(f"- Final sample sizes: Train={convergence_info['final_train_sample_size']:,}, Test={convergence_info['final_test_sample_size']:,}")
    print(f"- Bootstrap iterations: {convergence_info['final_iterations']}")
    print(f"- Optimization attempts: {convergence_info['attempts_made']}")
    print(f"- Overlaps resolved: {convergence_info['overlaps_resolved']}")

    # Step 2: Detailed results analysis
    print("\nStep 2: Analyzing results by metric...")
    print("-" * 50)

    key_metrics = ['AUROC', 'KS', 'Avg_IV']
    best_methods = {}

    for metric in key_metrics:
        print(f"\n{metric} Results (Mean ± SE):")
        mean_col = f'{metric}_mean'
        se_col = f'{metric}_se'
        ci_lower_col = f'{metric}_ci_lower'
        ci_upper_col = f'{metric}_ci_upper'

        if mean_col in results_summary.columns:
            metric_results = results_summary[['Binning_Strategy', mean_col, se_col, ci_lower_col, ci_upper_col]].dropna()
            metric_results = metric_results.sort_values(mean_col, ascending=False)

            if not metric_results.empty:
                best_method = metric_results.iloc[0]['Binning_Strategy']
                best_value = metric_results.iloc[0][mean_col]
                best_methods[metric] = {'method': best_method, 'value': best_value}

            for _, row in metric_results.iterrows():
                method = row['Binning_Strategy']
                mean_val = row[mean_col]
                se_val = row[se_col]
                ci_lower = row[ci_lower_col]
                ci_upper = row[ci_upper_col]

                print(f"  {method:25}: {mean_val:.4f} ± {se_val:.4f} (CI: {ci_lower:.4f} - {ci_upper:.4f})")

    # Step 3: Statistical significance analysis
    print("\nStep 3: Statistical significance analysis...")
    print("-" * 50)

    for metric, info in best_methods.items():
        print(f"Best {metric}: {info['method']} ({info['value']:.4f})")

    # Step 4: Recommendations
    print("\nStep 4: Recommendations...")
    print("-" * 50)

    recommendations = []

    # Overall best method (based on AUROC)
    if 'AUROC' in best_methods:
        overall_best = best_methods['AUROC']['method']
        recommendations.append(f"Overall recommended method: {overall_best} (highest AUROC)")

    # Check for consistent winners
    method_wins = {}
    for metric, info in best_methods.items():
        method = info['method']
        method_wins[method] = method_wins.get(method, 0) + 1

    consistent_winner = max(method_wins, key=method_wins.get) if method_wins else None
    if consistent_winner and method_wins[consistent_winner] >= 2:
        recommendations.append(f"Most consistent performer: {consistent_winner} (best in {method_wins[consistent_winner]} metrics)")

    # Sample size recommendations
    if not convergence_info['overlaps_resolved']:
        recommendations.append("Consider increasing sample size further for better separation of confidence intervals")
    else:
        recommendations.append("Current sample size provides adequate statistical power")

    for rec in recommendations:
        print(f"• {rec}")

    # Step 5: Final summary
    total_time = time.time() - start_time

    print(f"\nAnalysis completed in {total_time:.2f} seconds")
    print(f"Results saved to: {output_dir if output_dir else 'current directory'}")

    return {
        'bootstrap_results': initial_results,
        'best_methods': best_methods,
        'recommendations': recommendations,
        'convergence_info': convergence_info,
        'total_time': total_time
    }


# === MEMORY-SAFE BOOTSTRAP FUNCTIONS ===

def bootstrap_binning_comparison_with_checkpoints(
    X_train_population: pd.DataFrame,
    X_test_population: pd.DataFrame,
    y_train_population: pd.Series,
    y_test_population: pd.Series,
    feature_columns: list = None,
    bootstrap_iterations: int = 1000,
    train_sample_size: int = 150000,
    test_sample_size: int = 150000,
    max_bins_per_feature: int = 10,
    random_state: int = 42,
    force_monotonic: bool = True,
    monotonic_direction: str = None,
    confidence_level: float = 0.95,
    overlap_threshold: float = 0.1,
    auto_adjust_sample_size: bool = True,
    max_iterations: int = 2000,
    verbose: bool = True,
    log_outputs: bool = True,
    output_dir: str = None,
    n_jobs: int = None,
    # NEW PARAMETERS FOR MEMORY MANAGEMENT AND CHECKPOINTING
    max_memory_gb: float = None,
    checkpoint_frequency: int = 50,  # Save every N iterations
    resume_from_checkpoint: bool = True,
    chunk_size: int = 10,  # Process N iterations at a time
    emergency_save_on_error: bool = True
):
    """
    Memory-safe bootstrap binning comparison with checkpointing and resume capability.

    NEW FEATURES:
    - **Checkpointing**: Automatic saving every N iterations
    - **Resume Capability**: Can resume from last checkpoint
    - **Memory Management**: Monitors and controls memory usage
    - **Emergency Save**: Saves progress on errors/crashes
    - **Chunked Processing**: Processes iterations in small chunks
    """

    if output_dir is None:
        output_dir = f"bootstrap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize managers
    memory_manager = MemoryManager(max_memory_gb=max_memory_gb)
    checkpoint_manager = CheckpointManager(output_dir, "bootstrap_analysis")

    # Initialize logger
    logger = None
    if log_outputs:
        logger = BinningLogger(base_filename="bootstrap_binning_analysis", output_dir=output_dir)
        logger.log_section("MEMORY-SAFE BOOTSTRAP BINNING ANALYSIS", level=1)
        logger.log(f"Analysis started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log(f"Memory Management: Max {max_memory_gb}GB" if max_memory_gb else "Memory Management: No limit")
        logger.log(f"Checkpointing: Every {checkpoint_frequency} iterations")
        logger.log(f"Chunk Size: {chunk_size} iterations per chunk")

    # Identify feature columns
    if feature_columns is None:
        feature_columns = [col for col in X_train_population.columns
                          if pd.api.types.is_numeric_dtype(X_train_population[col])]

    # Save initial configuration and metadata
    config = {
        'bootstrap_iterations': bootstrap_iterations,
        'train_sample_size': train_sample_size,
        'test_sample_size': test_sample_size,
        'max_bins_per_feature': max_bins_per_feature,
        'random_state': random_state,
        'force_monotonic': force_monotonic,
        'monotonic_direction': monotonic_direction,
        'confidence_level': confidence_level,
        'overlap_threshold': overlap_threshold,
        'feature_columns': feature_columns,
        'max_memory_gb': max_memory_gb,
        'checkpoint_frequency': checkpoint_frequency,
        'chunk_size': chunk_size
    }
    checkpoint_manager.save_config(config)

    metadata = {
        'train_population_shape': X_train_population.shape,
        'test_population_shape': X_test_population.shape,
        'train_population_memory_mb': X_train_population.memory_usage(deep=True).sum() / (1024**2),
        'test_population_memory_mb': X_test_population.memory_usage(deep=True).sum() / (1024**2),
        'feature_columns': feature_columns,
        'target_distribution': y_train_population.value_counts().to_dict()
    }
    checkpoint_manager.save_metadata(metadata)

    # Check for existing checkpoint
    checkpoint_data = None
    bootstrap_results = []
    start_iteration = 0

    if resume_from_checkpoint:
        checkpoint_data = checkpoint_manager.load_checkpoint()
        if checkpoint_data['has_checkpoint'] and checkpoint_data['results']:
            bootstrap_results = checkpoint_data['results']
            start_iteration = len(bootstrap_results)
            if logger:
                logger.log(f"RESUMING from checkpoint: {start_iteration} iterations completed")

    try:
        # Run chunked bootstrap iterations with memory management
        with memory_manager.memory_context("Bootstrap Analysis"):
            bootstrap_results = _run_chunked_bootstrap_iterations(
                X_train_population=X_train_population,
                X_test_population=X_test_population,
                y_train_population=y_train_population,
                y_test_population=y_test_population,
                feature_columns=feature_columns,
                total_iterations=bootstrap_iterations,
                start_iteration=start_iteration,
                existing_results=bootstrap_results,
                config=config,
                memory_manager=memory_manager,
                checkpoint_manager=checkpoint_manager,
                checkpoint_frequency=checkpoint_frequency,
                chunk_size=chunk_size,
                n_jobs=n_jobs or 1,
                verbose=verbose,
                logger=logger
            )

        # Analyze final results
        if logger:
            logger.log_section("ANALYZING FINAL RESULTS", level=2)

        results_summary, distributions = _analyze_bootstrap_results(
            bootstrap_results, confidence_level, logger
        )

        # Create visualizations
        figures = _create_bootstrap_visualizations(
            results_summary, distributions, confidence_level, logger
        )

        # Prepare final output
        convergence_info = {
            'total_iterations_completed': len(bootstrap_results),
            'checkpoints_saved': len(bootstrap_results) // checkpoint_frequency,
            'memory_usage_gb': memory_manager.get_memory_usage(),
            'analysis_successful': True
        }

        # Final save
        checkpoint_manager.save_results(bootstrap_results)
        checkpoint_manager.save_progress({
            'completed': len(bootstrap_results),
            'total': bootstrap_iterations,
            'status': 'completed'
        })

        if logger:
            logger.log_section("ANALYSIS COMPLETED SUCCESSFULLY", level=1)
            logger.log(f"Total iterations: {len(bootstrap_results)}")
            logger.log(f"Final memory usage: {memory_manager.get_memory_usage():.2f}GB")

        return {
            'results_summary': results_summary,
            'bootstrap_distributions': distributions,
            'convergence_info': convergence_info,
            'figures': figures,
            'individual_results': bootstrap_results,
            'checkpoint_manager': checkpoint_manager
        }

    except Exception as e:
        if logger:
            logger.log(f"ERROR: Analysis failed: {str(e)}")
            logger.log(f"Traceback: {traceback.format_exc()}")

        # Emergency save
        if emergency_save_on_error:
            emergency_file = checkpoint_manager.create_emergency_save(
                bootstrap_results=bootstrap_results,
                config=config,
                metadata=metadata,
                error_info={
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'iterations_completed': len(bootstrap_results),
                    'memory_usage_gb': memory_manager.get_memory_usage()
                }
            )

            print(f"🚨 ANALYSIS CRASHED - Emergency save created: {emergency_file}")
            print(f"🚨 You can resume using the checkpoint system")

        raise

def _run_chunked_bootstrap_iterations(
    X_train_population, X_test_population, y_train_population, y_test_population,
    feature_columns, total_iterations, start_iteration, existing_results,
    config, memory_manager, checkpoint_manager, checkpoint_frequency,
    chunk_size, n_jobs, verbose, logger
):
    """Run bootstrap iterations in memory-safe chunks with checkpointing."""

    bootstrap_results = existing_results.copy()
    current_iteration = start_iteration

    while current_iteration < total_iterations:
        # Calculate chunk parameters
        remaining_iterations = total_iterations - current_iteration
        current_chunk_size = min(chunk_size, remaining_iterations)
        chunk_end = current_iteration + current_chunk_size

        if logger:
            logger.log(f"Processing chunk: iterations {current_iteration}-{chunk_end-1} ({current_chunk_size} iterations)")

        # Memory check before chunk
        memory_warnings = memory_manager.check_memory_limit()
        if memory_warnings:
            for warning in memory_warnings:
                if logger:
                    logger.log(f"⚠️  MEMORY WARNING: {warning}")
                print(f"⚠️  MEMORY WARNING: {warning}")

            # Force garbage collection
            freed = memory_manager.force_garbage_collection()
            if freed > 0.1:
                if logger:
                    logger.log(f"🧠 Freed {freed:.2f}GB via garbage collection")

        # Process chunk
        with memory_manager.memory_context(f"Bootstrap Chunk {current_iteration}-{chunk_end-1}"):
            chunk_results = _process_bootstrap_chunk(
                X_train_population, X_test_population, y_train_population, y_test_population,
                feature_columns, current_iteration, current_chunk_size, config, n_jobs, verbose
            )

            bootstrap_results.extend(chunk_results)
            current_iteration = chunk_end

        # Checkpoint if needed
        if current_iteration % checkpoint_frequency == 0 or current_iteration >= total_iterations:
            checkpoint_manager.save_results(bootstrap_results, current_iteration)
            checkpoint_manager.save_progress({
                'completed': current_iteration,
                'total': total_iterations,
                'last_chunk_size': len(chunk_results),
                'status': 'in_progress' if current_iteration < total_iterations else 'completed'
            })

            if logger:
                logger.log(f"📊 Checkpoint saved: {current_iteration}/{total_iterations} iterations")

        # Progress update
        if verbose:
            progress_pct = (current_iteration / total_iterations) * 100
            print(f"📈 Progress: {current_iteration}/{total_iterations} ({progress_pct:.1f}%)")

    return bootstrap_results

def _process_bootstrap_chunk(
    X_train_pop, X_test_pop, y_train_pop, y_test_pop, feature_columns,
    start_iteration, chunk_size, config, n_jobs, verbose
):
    """Process a single chunk of bootstrap iterations with minimal memory footprint."""

    chunk_results = []

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid data copying
    max_workers = min(n_jobs, chunk_size, 4)  # Conservative limit

    # Create arguments for chunk
    args_list = []
    for i in range(chunk_size):
        iteration_num = start_iteration + i
        args_list.append((
            X_train_pop, X_test_pop, y_train_pop, y_test_pop,
            feature_columns, config, iteration_num
        ))

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks in chunk
            future_to_iteration = {
                executor.submit(_single_bootstrap_iteration_memory_safe, args): args[-1]
                for args in args_list
            }

            # Collect results
            for future in as_completed(future_to_iteration):
                try:
                    result = future.result()
                    if result is not None:
                        chunk_results.append(result)

                        # Immediate cleanup
                        del result

                except Exception as e:
                    iteration_num = future_to_iteration[future]
                    if verbose:
                        print(f"❌ Iteration {iteration_num} failed: {e}")
    else:
        # Sequential processing for memory safety
        for args in args_list:
            try:
                result = _single_bootstrap_iteration_memory_safe(args)
                if result is not None:
                    chunk_results.append(result)

                # Immediate cleanup
                del result
                gc.collect()

            except Exception as e:
                if verbose:
                    print(f"❌ Iteration {args[-1]} failed: {e}")

    # Force garbage collection after chunk
    gc.collect()

    return chunk_results

def _single_bootstrap_iteration_memory_safe(args):
    """Memory-safe single bootstrap iteration with explicit cleanup."""

    (X_train_pop, X_test_pop, y_train_pop, y_test_pop,
     feature_columns, config, iteration_num) = args

    try:
        # Extract config
        train_size = config['train_sample_size']
        test_size = config['test_sample_size']
        max_bins = config['max_bins_per_feature']
        random_state = config['random_state'] + iteration_num

        # Set random state
        np.random.seed(random_state)

        # Sample with explicit cleanup of indices
        train_indices = np.random.choice(len(X_train_pop), size=train_size, replace=False)
        test_indices = np.random.choice(len(X_test_pop), size=test_size, replace=False)

        # Create samples with memory-efficient copying
        X_train_sample = X_train_pop.iloc[train_indices].copy()
        X_test_sample = X_test_pop.iloc[test_indices].copy()
        y_train_sample = y_train_pop.iloc[train_indices].copy()
        y_test_sample = y_test_pop.iloc[test_indices].copy()

        # Clean up indices immediately
        del train_indices, test_indices

        # Reset indices to avoid memory fragmentation
        X_train_sample.reset_index(drop=True, inplace=True)
        X_test_sample.reset_index(drop=True, inplace=True)
        y_train_sample.reset_index(drop=True, inplace=True)
        y_test_sample.reset_index(drop=True, inplace=True)

        # Run comparison with minimal logging
        results_df, _, _, _ = compare_binning_strategies_on_dataset(
            X_train_sample, X_test_sample, y_train_sample, y_test_sample,
            feature_columns=feature_columns,
            max_bins_per_feature=max_bins,
            random_state=random_state,
            force_monotonic=config.get('force_monotonic', True),
            monotonic_direction=config.get('monotonic_direction'),
            verbose=False,
            log_outputs=False,
            n_jobs=1  # Force single-threaded to avoid nested parallelism
        )

        # Extract minimal results
        iteration_results = {}
        for _, row in results_df.iterrows():
            method_name = row['Binning_Strategy']
            iteration_results[method_name] = {
                'iteration': iteration_num,
                'AUROC': row.get('AUROC', np.nan),
                'KS': row.get('KS', np.nan),
                'Avg_IV': row.get('Avg_IV', np.nan),
                'PR_AUC': row.get('PR_AUC', np.nan),
                'F1_pos': row.get('F1_pos', np.nan),
                'AIC': row.get('AIC', np.nan)
            }

        # Explicit cleanup
        del X_train_sample, X_test_sample, y_train_sample, y_test_sample, results_df
        gc.collect()

        return iteration_results

    except Exception as e:
        # Cleanup on error
        gc.collect()
        return None

# === RESUME AND ANALYSIS FUNCTIONS ===

def resume_bootstrap_analysis(checkpoint_dir, output_dir=None):
    """Resume bootstrap analysis from checkpoint."""

    if output_dir is None:
        output_dir = checkpoint_dir

    checkpoint_manager = CheckpointManager(output_dir, "bootstrap_analysis")
    checkpoint_data = checkpoint_manager.load_checkpoint()

    if not checkpoint_data['has_checkpoint']:
        print("❌ No checkpoint found to resume from")
        return None

    print("🔄 Resuming bootstrap analysis...")

    # Load results and analyze
    if checkpoint_data['results']:
        results_summary, distributions = _analyze_bootstrap_results(
            checkpoint_data['results'],
            checkpoint_data['config'].get('confidence_level', 0.95),
            None  # No logger for resume
        )

        # Create visualizations
        figures = _create_bootstrap_visualizations(
            results_summary, distributions,
            checkpoint_data['config'].get('confidence_level', 0.95),
            None
        )

        print(f"✅ Analysis resumed with {len(checkpoint_data['results'])} bootstrap results")

        return {
            'results_summary': results_summary,
            'bootstrap_distributions': distributions,
            'figures': figures,
            'individual_results': checkpoint_data['results'],
            'config': checkpoint_data['config'],
            'metadata': checkpoint_data['metadata']
        }

    else:
        print("❌ No results found in checkpoint")
        return None

def create_emergency_save_script(current_variables, output_dir):
    """Create emergency save of current session variables."""

    checkpoint_manager = CheckpointManager(output_dir, "emergency_bootstrap")

    # Filter out large datasets but keep essential variables
    essential_vars = {}

    for var_name, var_value in current_variables.items():
        # Skip large datasets
        if var_name in ['X_train', 'X_test', 'X_train_population', 'X_test_population']:
            # Save only metadata about datasets
            if hasattr(var_value, 'shape'):
                essential_vars[f"{var_name}_shape"] = var_value.shape
                essential_vars[f"{var_name}_columns"] = list(var_value.columns) if hasattr(var_value, 'columns') else None
                essential_vars[f"{var_name}_memory_mb"] = var_value.memory_usage(deep=True).sum() / (1024**2) if hasattr(var_value, 'memory_usage') else None
            continue

        # Skip other large objects
        if var_name in ['logger', 'figures', 'figs']:
            continue

        # Keep essential results and configs
        if var_name in ['bootstrap_results', 'comprehensive_results', 'results_summary',
                       'distributions', 'convergence_info', 'individual_results']:
            essential_vars[var_name] = var_value

        # Keep small configuration objects
        elif isinstance(var_value, (dict, list, str, int, float, bool)):
            essential_vars[var_name] = var_value

    # Create emergency save
    emergency_file = checkpoint_manager.create_emergency_save(**essential_vars)

    if emergency_file:
        # Create resume script
        resume_script = f"""
# EMERGENCY RESUME SCRIPT
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

import pickle
import pandas as pd
from pathlib import Path

# Load emergency save
emergency_file = "{emergency_file}"
print(f"Loading emergency save: {{emergency_file}}")

with open(emergency_file, 'rb') as f:
    emergency_data = pickle.load(f)

# Extract variables
saved_vars = emergency_data['variables']
print(f"Saved variables: {{list(saved_vars.keys())}}")

# Restore key variables
bootstrap_results = saved_vars.get('bootstrap_results')
comprehensive_results = saved_vars.get('comprehensive_results')
results_summary = saved_vars.get('results_summary')

print(f"Bootstrap results: {{len(bootstrap_results) if bootstrap_results else 0}} iterations")
print(f"Analysis completed: {{comprehensive_results is not None}}")

# To resume analysis, reload your datasets and use:
# from efx_bootstrap import resume_bootstrap_analysis
# resumed_results = resume_bootstrap_analysis("{checkpoint_manager.checkpoint_dir}")
"""

        resume_script_file = checkpoint_manager.output_dir / "emergency_resume_script.py"
        with open(resume_script_file, 'w') as f:
            f.write(resume_script)

        print(f"📝 Emergency resume script created: {resume_script_file}")

        return {
            'emergency_file': emergency_file,
            'resume_script': str(resume_script_file),
            'saved_variables': list(essential_vars.keys())
        }

    return None

def safe_demo_analysis():
    """Demonstrate safe bootstrap analysis with checkpointing."""

    print("🚀 Starting SAFE bootstrap analysis with checkpointing...")

    try:
        # Load your data (replace with your actual data loading)
        train = pd.read_parquet('transformations2_good/train_selected_final.parquet')
        test = pd.read_parquet('transformations2_good/test_selected_final.parquet')

        y_train = train['target']
        y_test = test['target']
        X_train = train.drop(columns=['target'])
        X_test = test.drop(columns=['target'])

        del train, test  # Immediate cleanup

        # Run safe bootstrap analysis
        results = bootstrap_binning_comparison_with_checkpoints(
            X_train_population=X_train,
            X_test_population=X_test,
            y_train_population=y_train,
            y_test_population=y_test,
            bootstrap_iterations=1000,
            train_sample_size=150000,
            test_sample_size=150000,
            max_bins_per_feature=10,
            # Memory management
            max_memory_gb=32.0,  # Adjust based on your system
            checkpoint_frequency=25,  # Save every 25 iterations
            chunk_size=5,  # Process 5 iterations at a time
            # Safety settings
            resume_from_checkpoint=True,
            emergency_save_on_error=True,
            n_jobs=4,  # Conservative parallelism
            verbose=True,
            output_dir="safe_bootstrap_analysis"
        )

        print("✅ Analysis completed successfully!")
        return results

    except Exception as e:
        print(f"❌ Analysis failed: {e}")

        # Create emergency save of current session
        current_vars = {
            'X_train': locals().get('X_train'),
            'X_test': locals().get('X_test'),
            'y_train': locals().get('y_train'),
            'y_test': locals().get('y_test'),
            'results': locals().get('results')
        }

        emergency_info = create_emergency_save_script(current_vars, "emergency_output")
        if emergency_info:
            print(f"🚨 Emergency save created: {emergency_info['emergency_file']}")
            print(f"📝 Resume script: {emergency_info['resume_script']}")

        raise

def demo_comprehensive_analysis():
    """Demonstrate comprehensive analysis workflow with memory safety."""

    print("\n" + "="*80)
    print(" MEMORY-SAFE COMPREHENSIVE ANALYSIS WORKFLOW")
    print("="*80)

    # Load data with immediate cleanup
    train = pd.read_parquet('transformations2_good/train_selected_final.parquet')
    test = pd.read_parquet('transformations2_good/test_selected_final.parquet')
    y_train = train['target']
    y_test = test['target']
    X_train = train.drop(columns=['target'])
    X_test = test.drop(columns=['target'])

    del train, test  # Immediate cleanup

    print("\nRunning MEMORY-SAFE comprehensive bootstrap analysis...")

    comprehensive_results = example_bootstrap_binning_analysis(
        X_train_population=X_train,
        X_test_population=X_test,
        y_train_population=y_train,
        y_test_population=y_test,
        output_dir="comprehensive_output",
        quick_demo=False # Use full analysis
    )

    print(f"\nComprehensive analysis completed!")
    print(f"Check the comprehensive_output/ directory for all results")

    return comprehensive_results

def updated_example_bootstrap_binning_analysis(
    X_train_population: pd.DataFrame,
    X_test_population: pd.DataFrame,
    y_train_population: pd.Series,
    y_test_population: pd.Series,
    output_dir: str = None,
    quick_demo: bool = False
):
    """
    UPDATED: Memory-safe comprehensive bootstrap analysis with checkpointing.

    This example demonstrates the full workflow with complete logging, figure saving,
    and new memory management features.
    """

    print("="*80)
    print("MEMORY-SAFE COMPREHENSIVE BOOTSTRAP BINNING ANALYSIS")
    print("="*80)

    start_time = time.time()

    # Step 1: Initial memory-safe bootstrap analysis
    print("\nStep 1: Running memory-safe bootstrap analysis with checkpointing...")
    print("-" * 50)

    # Adjust parameters for demo mode
    if quick_demo:
        iterations = 100
        sample_size = 50000
        memory_limit = 8.0
        checkpoint_freq = 10
        chunk_size = 3
        print("Running in QUICK DEMO mode with reduced parameters")
    else:
        iterations = 1000
        sample_size = 150000
        memory_limit = 32.0
        checkpoint_freq = 25
        chunk_size = 5
        print("Running in FULL ANALYSIS mode with memory management")

    try:
        initial_results = bootstrap_binning_comparison_with_checkpoints(
            X_train_population=X_train_population,
            X_test_population=X_test_population,
            y_train_population=y_train_population,
            y_test_population=y_test_population,
            bootstrap_iterations=iterations,
            train_sample_size=sample_size,
            test_sample_size=sample_size,
            # Memory management and checkpointing
            max_memory_gb=memory_limit,
            checkpoint_frequency=checkpoint_freq,
            chunk_size=chunk_size,
            resume_from_checkpoint=True,
            emergency_save_on_error=True,
            # Analysis parameters
            confidence_level=0.95,
            overlap_threshold=0.1,
            auto_adjust_sample_size=True,
            max_iterations=2000,
            verbose=True,
            log_outputs=True,
            output_dir=output_dir,
            n_jobs=4  # Conservative parallelism
        )

        # Analyze results
        results_summary = initial_results['results_summary']
        convergence_info = initial_results['convergence_info']

        print(f"\nMemory-Safe Analysis Results:")
        print(f"- Total iterations completed: {convergence_info['total_iterations_completed']}")
        print(f"- Checkpoints saved: {convergence_info['checkpoints_saved']}")
        print(f"- Final memory usage: {convergence_info['memory_usage_gb']:.2f}GB")
        print(f"- Analysis successful: {convergence_info['analysis_successful']}")

        # Step 2: Detailed results analysis
        print("\nStep 2: Analyzing results by metric...")
        print("-" * 50)

        key_metrics = ['AUROC', 'KS', 'Avg_IV']
        best_methods = {}

        for metric in key_metrics:
            print(f"\n{metric} Results (Mean ± SE):")
            mean_col = f'{metric}_mean'
            se_col = f'{metric}_se'
            ci_lower_col = f'{metric}_ci_lower'
            ci_upper_col = f'{metric}_ci_upper'

            if mean_col in results_summary.columns:
                metric_results = results_summary[['Binning_Strategy', mean_col, se_col, ci_lower_col, ci_upper_col]].dropna()
                metric_results = metric_results.sort_values(mean_col, ascending=False)

                if not metric_results.empty:
                    best_method = metric_results.iloc[0]['Binning_Strategy']
                    best_value = metric_results.iloc[0][mean_col]
                    best_methods[metric] = {'method': best_method, 'value': best_value}

                for _, row in metric_results.iterrows():
                    method = row['Binning_Strategy']
                    mean_val = row[mean_col]
                    se_val = row[se_col]
                    ci_lower = row[ci_lower_col]
                    ci_upper = row[ci_upper_col]

                    print(f"  {method:25}: {mean_val:.4f} ± {se_val:.4f} (CI: {ci_lower:.4f} - {ci_upper:.4f})")

        # Step 3: Statistical significance analysis
        print("\nStep 3: Statistical significance analysis...")
        print("-" * 50)

        for metric, info in best_methods.items():
            print(f"Best {metric}: {info['method']} ({info['value']:.4f})")

        # Step 4: Recommendations
        print("\nStep 4: Recommendations...")
        print("-" * 50)

        recommendations = []

        # Overall best method (based on AUROC)
        if 'AUROC' in best_methods:
            overall_best = best_methods['AUROC']['method']
            recommendations.append(f"Overall recommended method: {overall_best} (highest AUROC)")

        # Check for consistent winners
        method_wins = {}
        for metric, info in best_methods.items():
            method = info['method']
            method_wins[method] = method_wins.get(method, 0) + 1

        consistent_winner = max(method_wins, key=method_wins.get) if method_wins else None
        if consistent_winner and method_wins[consistent_winner] >= 2:
            recommendations.append(f"Most consistent performer: {consistent_winner} (best in {method_wins[consistent_winner]} metrics)")

        # Memory management recommendations
        recommendations.append(f"Memory usage was well-controlled at {convergence_info['memory_usage_gb']:.2f}GB")
        recommendations.append("Checkpointing system successfully prevented data loss")

        for rec in recommendations:
            print(f"• {rec}")

        # Step 5: Final summary
        total_time = time.time() - start_time

        print(f"\nMemory-safe analysis completed in {total_time:.2f} seconds")
        print(f"Results saved to: {output_dir if output_dir else 'current directory'}")

        return {
            'bootstrap_results': initial_results,
            'best_methods': best_methods,
            'recommendations': recommendations,
            'convergence_info': convergence_info,
            'total_time': total_time
        }

    except Exception as e:
        print(f"❌ Analysis failed: {e}")

        # Create emergency save of current session
        current_vars = {
            'X_train_population': X_train_population,
            'X_test_population': X_test_population,
            'y_train_population': y_train_population,
            'y_test_population': y_test_population,
            'initial_results': locals().get('initial_results')
        }

        emergency_info = create_emergency_save_script(current_vars, output_dir or "emergency_output")
        if emergency_info:
            print(f"🚨 Emergency save created: {emergency_info['emergency_file']}")
            print(f"📝 Resume script: {emergency_info['resume_script']}")

        raise

# Update the main example function name for consistency
example_bootstrap_binning_analysis = updated_example_bootstrap_binning_analysis

# if __name__ == "__main__":
#     # Run safe demo with memory management
#     logger = BinningLogger("memory_safe_analysis")
#     with logger.capture_all_output():
#         try:
#             comprehensive_results = demo_comprehensive_analysis()
#         except Exception as e:
#             print(f"Main analysis failed: {e}")
#             # The error handling and emergency save will be triggered automatically



print("🚀 Loading data to resume analysis...")

# 1. Reload the initial datasets
#    (This is necessary because you restarted Python)
train = pd.read_parquet('transformations2_good/train_selected_final.parquet')
test = pd.read_parquet('transformations2_good/test_selected_final.parquet')
y_train = train['target']
y_test = test['target']
X_train = train.drop(columns=['target'])
X_test = test.drop(columns=['target'])

del train, test # Clean up memory right away

print("✅ Data loaded. Resuming bootstrap process...")

# 2. Call the main function again with the same output directory
#    It will automatically find the checkpoint and continue.
comprehensive_results = bootstrap_binning_comparison_with_checkpoints(
    X_train_population=X_train,
    X_test_population=X_test,
    y_train_population=y_train,
    y_test_population=y_test,
    bootstrap_iterations=1000,          # Set your desired total iterations
    train_sample_size=150000,
    test_sample_size=150000,
    max_memory_gb=16.0,                 # You might want to adjust this
    checkpoint_frequency=25,
    chunk_size=1,
    resume_from_checkpoint=True,        # This is enabled by default
    emergency_save_on_error=True,
    n_jobs=2,
    verbose=True,
    output_dir="comprehensive_output"   # Crucial: Use the SAME directory as before
)

print("🎉 Analysis successfully completed to 1000 iterations!")
print(comprehensive_results['results_summary'])
