"""
UpScale App - GPU Detection Module
Consolidated GPU detection for all supported backends
"""

import os
import subprocess
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class GPUDetector:
    """Universal GPU detection and capability assessment"""
    
    def __init__(self, resource_manager):
        self.resource_manager = resource_manager
        self._gpu_cache = None
        
    def detect_gpus(self) -> Dict[str, Any]:
        """Detect all available GPUs and their capabilities"""
        if self._gpu_cache is not None:
            return self._gpu_cache
        
        gpu_info = {
            'nvidia': self._detect_nvidia_gpus(),
            'amd': self._detect_amd_gpus(),
            'intel': self._detect_intel_gpus(),
            'vulkan': self._detect_vulkan_gpus(),
            'best_backend': 'cpu'
        }
        
        # Determine best backend
        gpu_info['best_backend'] = self._determine_best_backend(gpu_info)
        
        self._gpu_cache = gpu_info
        logger.info(f"GPU Detection completed - Best backend: {gpu_info['best_backend']}")
        
        return gpu_info
    
    def _detect_nvidia_gpus(self) -> Dict[str, Any]:
        """Detect NVIDIA GPUs"""
        info = {
            'available': False,
            'gpus': [],
            'driver_version': None,
            'cuda_available': False
        }
        
        try:
            # Try nvidia-smi first
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 2:
                        info['gpus'].append({
                            'name': parts[0].strip(),
                            'memory_mb': int(parts[1]) if parts[1].isdigit() else 0,
                            'driver_version': parts[2].strip() if len(parts) > 2 else 'Unknown'
                        })
                
                if info['gpus']:
                    info['available'] = True
                    info['driver_version'] = info['gpus'][0]['driver_version']
                    
            # Check CUDA availability (basic check)
            try:
                cuda_result = subprocess.run(
                    ['nvcc', '--version'], 
                    capture_output=True, text=True, timeout=5
                )
                info['cuda_available'] = cuda_result.returncode == 0
            except:
                pass
                
        except Exception as e:
            logger.debug(f"NVIDIA GPU detection failed: {e}")
        
        return info
    
    def _detect_amd_gpus(self) -> Dict[str, Any]:
        """Detect AMD GPUs"""
        info = {
            'available': False,
            'gpus': [],
            'rocm_available': False
        }
        
        try:
            # Try Windows WMI approach
            if os.name == 'nt':
                try:
                    result = subprocess.run([
                        'wmic', 'path', 'win32_VideoController',
                        'get', 'name,AdapterRAM', '/format:csv'
                    ], capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')[1:]  # Skip header
                        for line in lines:
                            if line.strip() and 'AMD' in line.upper():
                                parts = line.split(',')
                                if len(parts) >= 3:
                                    name = parts[2].strip()
                                    memory_str = parts[1].strip()
                                    memory_mb = 0
                                    
                                    if memory_str and memory_str.isdigit():
                                        memory_mb = int(memory_str) // (1024 * 1024)
                                    
                                    if name:
                                        info['gpus'].append({
                                            'name': name,
                                            'memory_mb': memory_mb
                                        })
                                        info['available'] = True
                                        
                except Exception as e:
                    logger.debug(f"WMI AMD detection failed: {e}")
            
            # Check ROCm availability
            try:
                rocm_result = subprocess.run(
                    ['rocm-smi', '--showid'], 
                    capture_output=True, text=True, timeout=5
                )
                info['rocm_available'] = rocm_result.returncode == 0
            except:
                pass
                
        except Exception as e:
            logger.debug(f"AMD GPU detection failed: {e}")
        
        return info
    
    def _detect_intel_gpus(self) -> Dict[str, Any]:
        """Detect Intel GPUs"""
        info = {
            'available': False,
            'gpus': []
        }
        
        try:
            # Basic Intel GPU detection (Windows)
            if os.name == 'nt':
                result = subprocess.run([
                    'wmic', 'path', 'win32_VideoController',
                    'get', 'name', '/format:csv'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]
                    for line in lines:
                        if line.strip() and 'Intel' in line:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                name = parts[1].strip()
                                if name:
                                    info['gpus'].append({'name': name})
                                    info['available'] = True
                                    
        except Exception as e:
            logger.debug(f"Intel GPU detection failed: {e}")
        
        return info
    
    def _detect_vulkan_gpus(self) -> Dict[str, Any]:
        """Detect Vulkan-capable GPUs"""
        info = {
            'available': False,
            'devices': []
        }
        
        try:
            # Try vulkaninfo if available
            vulkaninfo_path = self.resource_manager.get_binary_path('vulkaninfo')
            if vulkaninfo_path:
                result = subprocess.run(
                    [vulkaninfo_path, '--summary'],
                    capture_output=True, text=True, timeout=10
                )
                
                if result.returncode == 0:
                    # Parse vulkaninfo output for device names
                    lines = result.stdout.split('\n')
                    for line in lines:
                        line = line.strip()
                        if 'deviceName' in line or 'GPU' in line:
                            # Extract device name
                            if '=' in line:
                                device_name = line.split('=')[1].strip().strip('"')
                                if device_name and device_name not in [d.get('name') for d in info['devices']]:
                                    info['devices'].append({'name': device_name})
                    
                    if info['devices']:
                        info['available'] = True
            
        except Exception as e:
            logger.debug(f"Vulkan GPU detection failed: {e}")
        
        return info
    
    def _determine_best_backend(self, gpu_info: Dict) -> str:
        """Determine the best available GPU backend"""
        # Priority order: NVIDIA CUDA > AMD > Vulkan > CPU
        
        if gpu_info['nvidia']['available'] and gpu_info['nvidia']['cuda_available']:
            return 'nvidia_cuda'
        elif gpu_info['nvidia']['available']:
            return 'nvidia'
        elif gpu_info['amd']['available'] and gpu_info['amd']['rocm_available']:
            return 'amd_rocm'
        elif gpu_info['amd']['available']:
            return 'amd'
        elif gpu_info['vulkan']['available']:
            return 'vulkan'
        else:
            return 'cpu'
    
    def get_gpu_summary(self) -> Dict[str, Any]:
        """Get a summary of GPU capabilities"""
        gpu_info = self.detect_gpus()
        
        summary = {
            'best_backend': gpu_info['best_backend'],
            'has_nvidia': gpu_info['nvidia']['available'],
            'has_amd': gpu_info['amd']['available'],
            'has_vulkan': gpu_info['vulkan']['available'],
            'total_gpus': 0,
            'primary_gpu': None
        }
        
        # Count total GPUs and find primary
        all_gpus = []
        if gpu_info['nvidia']['gpus']:
            all_gpus.extend(gpu_info['nvidia']['gpus'])
            if not summary['primary_gpu']:
                summary['primary_gpu'] = gpu_info['nvidia']['gpus'][0]
        
        if gpu_info['amd']['gpus']:
            all_gpus.extend(gpu_info['amd']['gpus'])
            if not summary['primary_gpu']:
                summary['primary_gpu'] = gpu_info['amd']['gpus'][0]
        
        summary['total_gpus'] = len(all_gpus)
        
        return summary
    
    def get_recommended_settings(self) -> Dict[str, Any]:
        """Get recommended processing settings based on hardware"""
        gpu_info = self.detect_gpus()
        summary = self.get_gpu_summary()
        
        settings = {
            'use_gpu': summary['best_backend'] != 'cpu',
            'backend': summary['best_backend'],
            'batch_size': 1,
            'max_resolution': (1920, 1080),
            'recommended_scale': 2.0
        }
        
        # Adjust settings based on GPU memory
        if summary['primary_gpu']:
            memory_gb = summary['primary_gpu'].get('memory_mb', 0) / 1024
            
            if memory_gb >= 8:
                settings['batch_size'] = 4
                settings['max_resolution'] = (3840, 2160)  # 4K
                settings['recommended_scale'] = 4.0
            elif memory_gb >= 4:
                settings['batch_size'] = 2
                settings['max_resolution'] = (2560, 1440)  # 1440p
                settings['recommended_scale'] = 2.0
        
        return settings