"""
Performance Monitoring and Profiling Module
Tracks system resources, processing speed, and optimization metrics
"""

import time
import psutil
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from pathlib import Path
import json
import gc
from contextlib import contextmanager

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config.settings import PATHS, PERFORMANCE

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_available_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_temperature: float = 0.0
    gpu_utilization: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_gb': self.memory_used_gb,
            'memory_available_gb': self.memory_available_gb,
            'gpu_memory_used_gb': self.gpu_memory_used_gb,
            'gpu_memory_total_gb': self.gpu_memory_total_gb,
            'gpu_temperature': self.gpu_temperature,
            'gpu_utilization': self.gpu_utilization,
            'disk_io_read_mb': self.disk_io_read_mb,
            'disk_io_write_mb': self.disk_io_write_mb
        }


@dataclass
class ProcessingMetrics:
    """Processing performance metrics"""
    task_name: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    frames_processed: int = 0
    frames_failed: int = 0
    bytes_processed: int = 0
    peak_memory_gb: float = 0.0
    peak_gpu_memory_gb: float = 0.0
    avg_frame_processing_time: float = 0.0
    
    @property
    def total_time(self) -> float:
        return self.end_time - self.start_time if self.end_time > 0 else 0
    
    @property
    def frames_per_second(self) -> float:
        return self.frames_processed / self.total_time if self.total_time > 0 else 0
    
    @property
    def success_rate(self) -> float:
        total = self.frames_processed + self.frames_failed
        return (self.frames_processed / total * 100) if total > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'task_name': self.task_name,
            'total_time': self.total_time,
            'frames_processed': self.frames_processed,
            'frames_failed': self.frames_failed,
            'frames_per_second': self.frames_per_second,
            'success_rate': self.success_rate,
            'bytes_processed': self.bytes_processed,
            'peak_memory_gb': self.peak_memory_gb,
            'peak_gpu_memory_gb': self.peak_gpu_memory_gb,
            'avg_frame_processing_time': self.avg_frame_processing_time
        }


class PerformanceMonitor:
    """System performance monitoring and profiling"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Data storage
        self.system_metrics_history: List[SystemMetrics] = []
        self.processing_metrics_history: List[ProcessingMetrics] = []
        self.current_processing: Optional[ProcessingMetrics] = None
        
        # Resource tracking
        self._last_disk_io = None
        self._baseline_memory = None
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        """Start system monitoring in background thread"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread"""
        while self.is_monitoring:
            try:
                metrics = self._collect_system_metrics()
                with self._lock:
                    self.system_metrics_history.append(metrics)
                    
                    # Keep only recent metrics (last 1000 samples)
                    if len(self.system_metrics_history) > 1000:
                        self.system_metrics_history = self.system_metrics_history[-1000:]
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        metrics = SystemMetrics()
        
        try:
            # CPU and Memory
            metrics.cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            metrics.memory_percent = memory.percent
            metrics.memory_used_gb = memory.used / (1024**3)
            metrics.memory_available_gb = memory.available / (1024**3)
            
            # Disk I/O
            if self._last_disk_io:
                current_io = psutil.disk_io_counters()
                if current_io:
                    time_delta = time.time() - self._last_disk_io['timestamp']
                    if time_delta > 0:
                        read_delta = current_io.read_bytes - self._last_disk_io['read_bytes']
                        write_delta = current_io.write_bytes - self._last_disk_io['write_bytes']
                        
                        metrics.disk_io_read_mb = (read_delta / (1024**2)) / time_delta
                        metrics.disk_io_write_mb = (write_delta / (1024**2)) / time_delta
            
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self._last_disk_io = {
                    'timestamp': time.time(),
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes
                }
            
            # GPU metrics (if available)
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                    gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                    
                    metrics.gpu_memory_used_gb = gpu_memory_allocated
                    
                    # Get GPU properties
                    if torch.cuda.device_count() > 0:
                        props = torch.cuda.get_device_properties(0)
                        metrics.gpu_memory_total_gb = props.total_memory / (1024**3)
                        
                        # Try to get GPU utilization (requires nvidia-ml-py)
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            
                            # GPU utilization
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            metrics.gpu_utilization = util.gpu
                            
                            # GPU temperature
                            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                            metrics.gpu_temperature = temp
                            
                        except ImportError:
                            pass  # nvidia-ml-py not available
                        except Exception:
                            pass  # Other nvidia-ml errors
                            
                except Exception:
                    pass  # GPU monitoring failed
        
        except Exception as e:
            logger.warning(f"Error collecting system metrics: {e}")
        
        return metrics
    
    @contextmanager
    def track_processing(self, task_name: str):
        """Context manager for tracking processing performance"""
        metrics = ProcessingMetrics(task_name=task_name, start_time=time.time())
        
        # Set baseline memory
        if self._baseline_memory is None:
            self._baseline_memory = self._get_current_memory_usage()
        
        try:
            with self._lock:
                self.current_processing = metrics
            
            yield metrics
            
        finally:
            metrics.end_time = time.time()
            
            # Record peak memory usage
            current_memory = self._get_current_memory_usage()
            metrics.peak_memory_gb = max(0, current_memory - self._baseline_memory)
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                metrics.peak_gpu_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
                torch.cuda.reset_peak_memory_stats()
            
            with self._lock:
                self.processing_metrics_history.append(metrics)
                self.current_processing = None
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
        except:
            return 0
    
    def update_processing_stats(self, frames_processed: int = 0, frames_failed: int = 0,
                              bytes_processed: int = 0):
        """Update current processing statistics"""
        if self.current_processing:
            self.current_processing.frames_processed += frames_processed
            self.current_processing.frames_failed += frames_failed
            self.current_processing.bytes_processed += bytes_processed
    
    def get_current_stats(self) -> Dict:
        """Get current performance statistics"""
        with self._lock:
            current_system = self._collect_system_metrics()
            
            stats = {
                "system": current_system.to_dict(),
                "processing": self.current_processing.to_dict() if self.current_processing else None,
                "monitoring_active": self.is_monitoring,
                "history_length": len(self.system_metrics_history)
            }
            
            # Add memory warnings
            if current_system.memory_percent > 90:
                stats["warnings"] = stats.get("warnings", [])
                stats["warnings"].append("High memory usage detected")
            
            if current_system.gpu_memory_used_gb > 0.9 * current_system.gpu_memory_total_gb:
                stats["warnings"] = stats.get("warnings", [])
                stats["warnings"].append("High GPU memory usage detected")
            
            return stats
    
    def get_performance_summary(self, last_n_tasks: int = 10) -> Dict:
        """Get performance summary for recent tasks"""
        with self._lock:
            recent_tasks = self.processing_metrics_history[-last_n_tasks:] if last_n_tasks > 0 else self.processing_metrics_history
            recent_system = self.system_metrics_history[-100:] if len(self.system_metrics_history) > 100 else self.system_metrics_history
        
        if not recent_tasks:
            return {"message": "No processing tasks recorded"}
        
        # Calculate averages
        avg_fps = sum(task.frames_per_second for task in recent_tasks) / len(recent_tasks)
        avg_success_rate = sum(task.success_rate for task in recent_tasks) / len(recent_tasks)
        total_frames = sum(task.frames_processed for task in recent_tasks)
        total_time = sum(task.total_time for task in recent_tasks)
        
        summary = {
            "tasks_analyzed": len(recent_tasks),
            "total_frames_processed": total_frames,
            "total_processing_time": total_time,
            "average_fps": avg_fps,
            "average_success_rate": avg_success_rate,
            "recent_tasks": [task.to_dict() for task in recent_tasks[-5:]]  # Last 5 tasks
        }
        
        # System performance averages
        if recent_system:
            summary["system_performance"] = {
                "avg_cpu_percent": sum(m.cpu_percent for m in recent_system) / len(recent_system),
                "avg_memory_percent": sum(m.memory_percent for m in recent_system) / len(recent_system),
                "avg_gpu_memory_gb": sum(m.gpu_memory_used_gb for m in recent_system) / len(recent_system)
            }
        
        return summary
    
    def save_metrics_to_file(self, filepath: str = None):
        """Save metrics history to file"""
        if filepath is None:
            filepath = str(PATHS["logs_dir"] / f"performance_metrics_{int(time.time())}.json")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with self._lock:
                data = {
                    "timestamp": time.time(),
                    "system_metrics": [m.to_dict() for m in self.system_metrics_history],
                    "processing_metrics": [m.to_dict() for m in self.processing_metrics_history],
                    "summary": self.get_performance_summary()
                }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Performance metrics saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            return None
    
    def cleanup(self):
        """Cleanup and reset monitoring"""
        self.stop_monitoring()
        with self._lock:
            self.system_metrics_history.clear()
            self.processing_metrics_history.clear()
            self.current_processing = None
        gc.collect()


class MemoryProfiler:
    """Memory usage profiler for detailed analysis"""
    
    def __init__(self):
        self.checkpoints = {}
        self.baseline = None
    
    def set_baseline(self, name: str = "baseline"):
        """Set memory baseline"""
        self.baseline = self._get_memory_info()
        self.checkpoints[name] = self.baseline
        logger.debug(f"Memory baseline set: {self.baseline['total_mb']:.1f} MB")
    
    def checkpoint(self, name: str) -> Dict:
        """Create memory checkpoint"""
        current = self._get_memory_info()
        self.checkpoints[name] = current
        
        if self.baseline:
            diff = current['total_mb'] - self.baseline['total_mb']
            logger.debug(f"Memory checkpoint '{name}': {current['total_mb']:.1f} MB (+{diff:.1f} MB from baseline)")
        
        return current
    
    def _get_memory_info(self) -> Dict:
        """Get detailed memory information"""
        info = {
            'timestamp': time.time(),
            'total_mb': 0,
            'process_mb': 0,
            'gpu_mb': 0
        }
        
        try:
            # Process memory
            process = psutil.Process()
            info['process_mb'] = process.memory_info().rss / (1024 * 1024)
            info['total_mb'] = info['process_mb']
            
            # GPU memory
            if TORCH_AVAILABLE and torch.cuda.is_available():
                info['gpu_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
                info['total_mb'] += info['gpu_mb']
                
        except Exception as e:
            logger.warning(f"Memory profiling error: {e}")
        
        return info
    
    def get_report(self) -> str:
        """Generate memory usage report"""
        if not self.checkpoints:
            return "No memory checkpoints recorded"
        
        report = ["Memory Usage Report", "=" * 20]
        
        if self.baseline:
            report.append(f"Baseline: {self.baseline['total_mb']:.1f} MB")
            report.append("")
        
        for name, info in self.checkpoints.items():
            if name == "baseline":
                continue
                
            line = f"{name}: {info['total_mb']:.1f} MB"
            if self.baseline:
                diff = info['total_mb'] - self.baseline['total_mb']
                line += f" (+{diff:.1f} MB)"
            
            report.append(line)
        
        return "\n".join(report)