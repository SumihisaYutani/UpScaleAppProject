"""
AMD GPU Detection Module
Detects AMD GPUs and ROCm availability for AI processing
"""

import logging
import subprocess
import sys
import os
from typing import Dict, List, Optional, Tuple
import platform

logger = logging.getLogger(__name__)


class AMDGPUDetector:
    """Detect and configure AMD GPU support"""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.amd_gpus = []
        self.rocm_available = False
        self.vulkan_available = False
        self._detection_complete = False
    
    def detect_all(self) -> Dict:
        """Perform complete AMD GPU detection"""
        if self._detection_complete:
            return self.get_detection_summary()
        
        try:
            self.amd_gpus = self._detect_amd_gpus()
            self.rocm_available = self._check_rocm_availability()
            self.vulkan_available = self._check_vulkan_availability()
            self._detection_complete = True
            
            logger.info(f"AMD GPU detection complete: {len(self.amd_gpus)} GPUs found")
            
        except Exception as e:
            logger.error(f"AMD GPU detection failed: {e}")
        
        return self.get_detection_summary()
    
    def _detect_amd_gpus(self) -> List[Dict]:
        """Detect AMD GPUs using multiple methods"""
        gpus = []
        
        # Method 1: Try wmic on Windows
        if self.platform == "windows":
            gpus.extend(self._detect_amd_gpus_wmic())
        
        # Method 2: Try lspci on Linux
        elif self.platform == "linux":
            gpus.extend(self._detect_amd_gpus_lspci())
        
        # Method 3: Try PyTorch ROCm detection
        gpus.extend(self._detect_amd_gpus_pytorch())
        
        # Method 4: Try OpenCL detection
        gpus.extend(self._detect_amd_gpus_opencl())
        
        # Remove duplicates based on device name
        unique_gpus = []
        seen_names = set()
        for gpu in gpus:
            if gpu['name'] not in seen_names:
                unique_gpus.append(gpu)
                seen_names.add(gpu['name'])
        
        return unique_gpus
    
    def _detect_amd_gpus_wmic(self) -> List[Dict]:
        """Detect AMD GPUs using Windows wmic"""
        gpus = []
        try:
            result = subprocess.run([
                'wmic', 'path', 'win32_VideoController', 'get', 
                'name,PNPDeviceID,AdapterRAM', '/format:csv'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line and 'AMD' in line.upper() or 'RADEON' in line.upper():
                        parts = [p.strip() for p in line.split(',') if p.strip()]
                        if len(parts) >= 3:
                            gpus.append({
                                'name': parts[2],  # Name
                                'memory': self._parse_memory(parts[1]) if len(parts) > 1 else 0,  # AdapterRAM
                                'id': parts[3] if len(parts) > 3 else '',  # PNPDeviceID
                                'method': 'wmic'
                            })
        except Exception as e:
            logger.debug(f"WMIC detection failed: {e}")
        
        return gpus
    
    def _detect_amd_gpus_lspci(self) -> List[Dict]:
        """Detect AMD GPUs using Linux lspci"""
        gpus = []
        try:
            result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if any(keyword in line.upper() for keyword in ['AMD', 'RADEON', 'ATI']):
                        if 'VGA' in line or 'Display' in line or '3D' in line:
                            # Extract device name and ID
                            parts = line.split(': ', 1)
                            if len(parts) == 2:
                                device_id = parts[0].strip()
                                device_name = parts[1].strip()
                                gpus.append({
                                    'name': device_name,
                                    'memory': 0,  # Not available via lspci
                                    'id': device_id,
                                    'method': 'lspci'
                                })
        except Exception as e:
            logger.debug(f"lspci detection failed: {e}")
        
        return gpus
    
    def _detect_amd_gpus_pytorch(self) -> List[Dict]:
        """Detect AMD GPUs using PyTorch ROCm"""
        gpus = []
        try:
            import torch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                # Check if ROCm is available
                if 'rocm' in torch.__version__ or hasattr(torch.version, 'hip'):
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        gpus.append({
                            'name': props.name,
                            'memory': props.total_memory,
                            'id': f'cuda:{i}',
                            'compute_capability': f"{props.major}.{props.minor}",
                            'method': 'pytorch_rocm'
                        })
        except Exception as e:
            logger.debug(f"PyTorch ROCm detection failed: {e}")
        
        return gpus
    
    def _detect_amd_gpus_opencl(self) -> List[Dict]:
        """Detect AMD GPUs using OpenCL"""
        gpus = []
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            for platform in platforms:
                if 'AMD' in platform.name.upper():
                    devices = platform.get_devices(cl.device_type.GPU)
                    for device in devices:
                        gpus.append({
                            'name': device.name.strip(),
                            'memory': device.global_mem_size,
                            'id': f'opencl:{device.int_ptr}',
                            'vendor': device.vendor.strip(),
                            'method': 'opencl'
                        })
        except Exception as e:
            logger.debug(f"OpenCL detection failed: {e}")
        
        return gpus
    
    def _check_rocm_availability(self) -> bool:
        """Check if ROCm is available"""
        try:
            # Method 1: Check PyTorch ROCm
            import torch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                if 'rocm' in torch.__version__.lower() or hasattr(torch.version, 'hip'):
                    logger.info("PyTorch with ROCm support detected")
                    return True
        except Exception:
            pass
        
        try:
            # Method 2: Check for ROCm installation
            if self.platform == "linux":
                result = subprocess.run(['rocm-smi', '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    logger.info("ROCm system installation detected")
                    return True
        except Exception:
            pass
        
        try:
            # Method 3: Check for HIP
            result = subprocess.run(['hipconfig', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("HIP installation detected")
                return True
        except Exception:
            pass
        
        return False
    
    def _check_vulkan_availability(self) -> bool:
        """Check if Vulkan is available"""
        try:
            # Try vulkan-tools
            result = subprocess.run(['vulkaninfo', '--summary'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and 'AMD' in result.stdout:
                return True
        except Exception:
            pass
        
        try:
            # Try vkcube (alternative check)
            result = subprocess.run(['vkcube', '--validate'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return True
        except Exception:
            pass
        
        return False
    
    def _parse_memory(self, memory_str: str) -> int:
        """Parse memory string to bytes"""
        try:
            if isinstance(memory_str, str):
                # Remove non-numeric characters except for unit indicators
                import re
                numbers = re.findall(r'\d+', memory_str)
                if numbers:
                    return int(numbers[0])
            elif isinstance(memory_str, (int, float)):
                return int(memory_str)
        except Exception:
            pass
        return 0
    
    def get_detection_summary(self) -> Dict:
        """Get summary of detection results"""
        return {
            'amd_gpus_found': len(self.amd_gpus),
            'amd_gpus': self.amd_gpus,
            'rocm_available': self.rocm_available,
            'vulkan_available': self.vulkan_available,
            'platform': self.platform,
            'recommended_backend': self._get_recommended_backend()
        }
    
    def _get_recommended_backend(self) -> str:
        """Get recommended backend based on detection results"""
        if self.rocm_available and self.amd_gpus:
            return 'rocm'
        elif self.vulkan_available and self.amd_gpus:
            return 'vulkan'
        else:
            return 'cpu'
    
    def get_best_gpu(self) -> Optional[Dict]:
        """Get the best AMD GPU for processing"""
        if not self.amd_gpus:
            return None
        
        # Sort by memory (highest first)
        sorted_gpus = sorted(self.amd_gpus, key=lambda x: x.get('memory', 0), reverse=True)
        return sorted_gpus[0]
    
    def install_rocm_pytorch(self) -> bool:
        """Install PyTorch with ROCm support"""
        try:
            logger.info("Installing PyTorch with ROCm support...")
            
            # ROCm PyTorch installation command
            cmd = [
                sys.executable, '-m', 'pip', 'install', 
                'torch', 'torchvision', 'torchaudio', 
                '--index-url', 'https://download.pytorch.org/whl/rocm6.0'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("ROCm PyTorch installation completed")
                return True
            else:
                logger.error(f"ROCm PyTorch installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to install ROCm PyTorch: {e}")
            return False


def detect_amd_gpu_support() -> Dict:
    """Convenience function to detect AMD GPU support"""
    detector = AMDGPUDetector()
    return detector.detect_all()


def get_amd_gpu_info() -> Dict:
    """Get comprehensive AMD GPU information"""
    detector = AMDGPUDetector()
    summary = detector.detect_all()
    
    # Add additional system information
    summary.update({
        'python_version': sys.version,
        'platform_details': platform.platform(),
        'architecture': platform.architecture()[0]
    })
    
    return summary


if __name__ == "__main__":
    # Test the detector
    import pprint
    
    print("AMD GPU Detection Test")
    print("=" * 40)
    
    info = get_amd_gpu_info()
    pprint.pprint(info)