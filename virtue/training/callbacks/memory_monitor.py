"""
Memory monitoring callbacks for training
"""

import torch
import psutil
import time
import threading
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import matplotlib.pyplot as plt

try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False

class MemoryMonitor:
    """
    Monitor GPU and CPU memory usage during training
    """
    
    def __init__(
        self,
        log_interval: int = 100,
        alert_threshold: float = 0.9,
        history_length: int = 1000,
        enable_alerts: bool = True
    ):
        self.log_interval = log_interval
        self.alert_threshold = alert_threshold
        self.history_length = history_length
        self.enable_alerts = enable_alerts
        
        # Memory history
        self.gpu_memory_history = deque(maxlen=history_length)
        self.cpu_memory_history = deque(maxlen=history_length)
        self.step_history = deque(maxlen=history_length)
        
        # Peak memory tracking
        self.peak_gpu_memory = 0.0
        self.peak_cpu_memory = 0.0
        
        # Alert tracking
        self.last_alert_time = 0.0
        self.alert_cooldown = 60.0  # seconds
        
        print(f"MemoryMonitor initialized - GPU available: {torch.cuda.is_available()}")
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information"""
        
        if not torch.cuda.is_available():
            return {}
        
        try:
            # PyTorch CUDA memory
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            
            # NVIDIA-ML-PY info
            if GPU_UTIL_AVAILABLE:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Assume single GPU
                    total_memory = gpu.memoryTotal / 1024  # GB
                    used_memory = gpu.memoryUsed / 1024    # GB
                    free_memory = gpu.memoryFree / 1024    # GB
                    utilization = gpu.memoryUtil
                else:
                    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    used_memory = allocated
                    free_memory = total_memory - used_memory
                    utilization = used_memory / total_memory
            else:
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                used_memory = allocated
                free_memory = total_memory - used_memory
                utilization = used_memory / total_memory
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total_memory,
                'used_gb': used_memory,
                'free_gb': free_memory,
                'utilization': utilization
            }
            
        except Exception as e:
            print(f"Error getting GPU memory info: {e}")
            return {}
    
    def get_cpu_memory_info(self) -> Dict[str, float]:
        """Get current CPU memory information"""
        
        try:
            memory = psutil.virtual_memory()
            
            return {
                'total_gb': memory.total / 1024**3,
                'used_gb': memory.used / 1024**3,
                'free_gb': memory.available / 1024**3,
                'utilization': memory.percent / 100.0
            }
            
        except Exception as e:
            print(f"Error getting CPU memory info: {e}")
            return {}
    
    def check_memory_alerts(self, gpu_info: Dict, cpu_info: Dict, step: int):
        """Check for memory alerts"""
        
        if not self.enable_alerts:
            return
        
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        alerts = []
        
        # GPU memory alerts
        if gpu_info.get('utilization', 0) > self.alert_threshold:
            alerts.append(f"GPU memory usage: {gpu_info['utilization']:.1%}")
        
        # CPU memory alerts  
        if cpu_info.get('utilization', 0) > self.alert_threshold:
            alerts.append(f"CPU memory usage: {cpu_info['utilization']:.1%}")
        
        if alerts:
            print(f"\n⚠️  MEMORY ALERT (Step {step}):")
            for alert in alerts:
                print(f"   {alert}")
            print("")
            self.last_alert_time = current_time
    
    def on_batch_end(self, step: int, metrics: Dict[str, Any]):
        """Callback for batch end"""
        
        if step % self.log_interval != 0:
            return
        
        # Get memory info
        gpu_info = self.get_gpu_memory_info()
        cpu_info = self.get_cpu_memory_info()
        
        # Update history
        self.step_history.append(step)
        if gpu_info:
            self.gpu_memory_history.append(gpu_info['utilization'])
            self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_info['utilization'])
        
        if cpu_info:
            self.cpu_memory_history.append(cpu_info['utilization'])
            self.peak_cpu_memory = max(self.peak_cpu_memory, cpu_info['utilization'])
        
        # Check alerts
        self.check_memory_alerts(gpu_info, cpu_info, step)
        
        # Log memory info
        log_msg = f"Step {step} Memory:"
        if gpu_info:
            log_msg += f" GPU: {gpu_info['used_gb']:.1f}/{gpu_info['total_gb']:.1f}GB ({gpu_info['utilization']:.1%})"
        if cpu_info:
            log_msg += f" CPU: {cpu_info['used_gb']:.1f}/{cpu_info['total_gb']:.1f}GB ({cpu_info['utilization']:.1%})"
        
        print(log_msg)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """Callback for epoch end"""
        
        print(f"\nEpoch {epoch} Memory Summary:")
        print(f"  Peak GPU Memory: {self.peak_gpu_memory:.1%}")
        print(f"  Peak CPU Memory: {self.peak_cpu_memory:.1%}")
    
    def plot_memory_usage(self, save_path: Optional[str] = None):
        """Plot memory usage history"""
        
        if len(self.step_history) == 0:
            print("No memory history to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        steps = list(self.step_history)
        
        # GPU memory plot
        if self.gpu_memory_history:
            gpu_memory = list(self.gpu_memory_history)
            ax1.plot(steps, gpu_memory, 'b-', linewidth=2, label='GPU Memory')
            ax1.axhline(y=self.alert_threshold, color='r', linestyle='--', alpha=0.7, label='Alert Threshold')
            ax1.set_ylabel('GPU Memory Utilization')
            ax1.set_ylim(0, 1)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # CPU memory plot  
        if self.cpu_memory_history:
            cpu_memory = list(self.cpu_memory_history)
            ax2.plot(steps, cpu_memory, 'g-', linewidth=2, label='CPU Memory')
            ax2.axhline(y=self.alert_threshold, color='r', linestyle='--', alpha=0.7, label='Alert Threshold')
            ax2.set_ylabel('CPU Memory Utilization')
            ax2.set_xlabel('Training Step')
            ax2.set_ylim(0, 1)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.title('Memory Usage During Training')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Memory plot saved to {save_path}")
        
        plt.show()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory usage summary"""
        
        return {
            'peak_gpu_memory': self.peak_gpu_memory,
            'peak_cpu_memory': self.peak_cpu_memory,
            'current_gpu_info': self.get_gpu_memory_info(),
            'current_cpu_info': self.get_cpu_memory_info(),
            'history_length': len(self.step_history)
        }

class GPUMemoryCallback:
    """
    Lightweight GPU memory callback
    """
    
    def __init__(self, log_interval: int = 500):
        self.log_interval = log_interval
        self.step_count = 0
    
    def __call__(self, step: int, metrics: Dict[str, Any]):
        """Callback function"""
        self.step_count += 1
        
        if self.step_count % self.log_interval == 0:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"Step {step}: GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

class MemoryProfiler:
    """
    Detailed memory profiler for debugging
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.profiles = []
        
    def start_profiling(self, name: str):
        """Start memory profiling"""
        if not self.enabled:
            return
            
        profile = {
            'name': name,
            'start_time': time.time(),
            'start_memory': self._get_memory_snapshot()
        }
        
        self.profiles.append(profile)
    
    def end_profiling(self, name: str):
        """End memory profiling"""
        if not self.enabled:
            return
            
        # Find matching profile
        profile = None
        for p in reversed(self.profiles):
            if p['name'] == name and 'end_time' not in p:
                profile = p
                break
        
        if profile is None:
            print(f"Warning: No matching profile found for {name}")
            return
        
        profile['end_time'] = time.time()
        profile['end_memory'] = self._get_memory_snapshot()
        profile['duration'] = profile['end_time'] - profile['start_time']
        
        # Calculate memory differences
        start_mem = profile['start_memory']
        end_mem = profile['end_memory']
        
        profile['memory_delta'] = {}
        for key in start_mem:
            if key in end_mem:
                profile['memory_delta'][key] = end_mem[key] - start_mem[key]
        
        # Print summary
        print(f"\nMemory Profile: {name}")
        print(f"  Duration: {profile['duration']:.2f}s")
        if 'gpu_allocated' in profile['memory_delta']:
            print(f"  GPU Memory Delta: {profile['memory_delta']['gpu_allocated']:.2f}GB")
        if 'cpu_used' in profile['memory_delta']:
            print(f"  CPU Memory Delta: {profile['memory_delta']['cpu_used']:.2f}GB")
    
    def _get_memory_snapshot(self) -> Dict[str, float]:
        """Get current memory snapshot"""
        snapshot = {}
        
        # GPU memory
        if torch.cuda.is_available():
            snapshot['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3
            snapshot['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3
        
        # CPU memory
        try:
            memory = psutil.virtual_memory()
            snapshot['cpu_used'] = memory.used / 1024**3
            snapshot['cpu_available'] = memory.available / 1024**3
        except:
            pass
        
        return snapshot
    
    def get_report(self) -> str:
        """Generate memory profiling report"""
        if not self.profiles:
            return "No profiling data available"
        
        report = "Memory Profiling Report\n" + "="*50 + "\n"
        
        for profile in self.profiles:
            if 'end_time' not in profile:
                continue
                
            report += f"\nProfile: {profile['name']}\n"
            report += f"  Duration: {profile['duration']:.2f}s\n"
            
            for key, delta in profile['memory_delta'].items():
                report += f"  {key}: {delta:+.2f}GB\n"
        
        return report

def clear_gpu_cache():
    """Clear GPU cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_memory_summary() -> str:
    """Get formatted memory summary"""
    lines = []
    
    # GPU Memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        if GPU_UTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    total = gpu.memoryTotal / 1024
                    used = gpu.memoryUsed / 1024
                    lines.append(f"GPU Memory: {used:.1f}/{total:.1f}GB ({gpu.memoryUtil:.1%})")
            except:
                pass
        
        lines.append(f"PyTorch GPU: Allocated={allocated:.1f}GB, Reserved={reserved:.1f}GB")
    
    # CPU Memory
    try:
        memory = psutil.virtual_memory()
        lines.append(f"CPU Memory: {memory.used/1024**3:.1f}/{memory.total/1024**3:.1f}GB ({memory.percent:.1%})")
    except:
        pass
    
    return "\n".join(lines)
