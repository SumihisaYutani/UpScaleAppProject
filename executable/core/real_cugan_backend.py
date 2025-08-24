"""
Real-CUGAN Backend for AI Processing
Optimized for anime and illustration upscaling
"""

import os
import subprocess
import tempfile
import logging
import time
import psutil
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

logger = logging.getLogger(__name__)

class RealCUGANBackend:
    """Real-CUGAN backend for high-quality anime/illustration upscaling"""
    
    def __init__(self, resource_manager, gpu_info: Dict[str, Any]):
        self.resource_manager = resource_manager
        self.gpu_info = gpu_info
        self.available = False
        
        # Real-CUGAN model configurations
        self.models = {
            'conservative': {
                'name': 'conservative',
                'description': 'Conservative model - minimal artifacts',
                'file': 'models-se'
            },
            'denoise1x': {
                'name': 'denoise1x', 
                'description': 'Light denoising + 1x scale',
                'file': 'models-nose'
            },
            'denoise2x': {
                'name': 'denoise2x',
                'description': 'Medium denoising + 2x scale', 
                'file': 'models-pro'
            },
            'denoise3x': {
                'name': 'denoise3x',
                'description': 'Strong denoising + 3x scale',
                'file': 'models-pro'
            }
        }
        
        self.default_model = 'denoise2x'  # Best balance for anime content
        self.current_model = self.default_model
        
        logger.info("Initializing Real-CUGAN backend...")
        self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if Real-CUGAN is available and functional"""
        try:
            # Check for Real-CUGAN executable
            realcugan_path = self.resource_manager.get_binary_path('realcugan-ncnn-vulkan')
            if not realcugan_path:
                logger.warning("Real-CUGAN executable not found")
                return False
            
            # Test Real-CUGAN with help command
            result = subprocess.run([realcugan_path, '-h'], 
                                  capture_output=True, text=True, timeout=10,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
            
            
            # Real-CUGAN outputs help to stdout and may return non-zero exit code
            if 'realcugan-ncnn-vulkan' in result.stdout or 'Usage:' in result.stdout:
                logger.info("Real-CUGAN executable found and functional")
                self.available = True
                
                # Log GPU compatibility
                if self.gpu_info.get('vulkan', {}).get('available'):
                    logger.info("Vulkan GPU acceleration available for Real-CUGAN")
                else:
                    logger.warning("No Vulkan support detected - Real-CUGAN will use CPU")
                
                return True
            else:
                logger.warning(f"Real-CUGAN test failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Real-CUGAN availability check failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if this backend is available"""
        return self.available
    
    def get_info(self) -> Dict[str, Any]:
        """Get backend information"""
        vulkan_available = self.gpu_info.get('vulkan', {}).get('available', False)
        amd_gpu_available = len(self.gpu_info.get('amd', {}).get('gpus', [])) > 0
        nvidia_gpu_available = len(self.gpu_info.get('nvidia', {}).get('gpus', [])) > 0
        
        # Consider GPU acceleration available if we have Vulkan OR discrete GPU
        gpu_available = vulkan_available or amd_gpu_available or nvidia_gpu_available
        
        return {
            'backend': 'Real-CUGAN',  # Add backend key for GUI compatibility
            'name': 'Real-CUGAN',
            'version': 'NCNN-Vulkan',
            'description': 'Real-CUGAN for anime/illustration upscaling',
            'gpu_acceleration': gpu_available,
            'gpu_mode': gpu_available,  # Add gpu_mode for compatibility
            'vulkan_support': vulkan_available,
            'amd_gpu_count': len(self.gpu_info.get('amd', {}).get('gpus', [])),
            'nvidia_gpu_count': len(self.gpu_info.get('nvidia', {}).get('gpus', [])),
            'models': list(self.models.keys()),
            'current_model': self.current_model,
            'available': self.available
        }
    
    def set_model(self, model_name: str) -> bool:
        """Set the Real-CUGAN model to use"""
        if model_name in self.models:
            self.current_model = model_name
            logger.info(f"Real-CUGAN model set to: {model_name} ({self.models[model_name]['description']})")
            return True
        else:
            logger.warning(f"Unknown Real-CUGAN model: {model_name}")
            return False
    
    def _monitor_gpu_status(self) -> Dict[str, Any]:
        """Monitor GPU status and detect issues"""
        vulkan_available = self.gpu_info.get('vulkan', {}).get('available', False)
        amd_gpu_available = len(self.gpu_info.get('amd', {}).get('gpus', [])) > 0
        
        # Consider GPU available if either Vulkan is detected OR AMD GPU is present
        gpu_available = vulkan_available or amd_gpu_available
        
        gpu_status = {
            'timestamp': time.time(),
            'cpu_usage': psutil.cpu_percent(interval=0.1),
            'memory_usage': psutil.virtual_memory().percent,
            'gpu_available': gpu_available,
            'vulkan_available': vulkan_available,
            'amd_gpu_count': len(self.gpu_info.get('amd', {}).get('gpus', [])),
            'nvidia_gpu_count': len(self.gpu_info.get('nvidia', {}).get('gpus', [])),
            'warnings': []
        }
        
        # Check for potential issues
        if gpu_status['cpu_usage'] > 90:
            gpu_status['warnings'].append("High CPU usage detected")
        if gpu_status['memory_usage'] > 85:
            gpu_status['warnings'].append("High memory usage detected")
        
        logger.info(f"GPU Status: {gpu_status}")
        return gpu_status

    def upscale_image(self, input_path: str, output_path: str, scale_factor: float = 2.0, progress_dialog=None) -> bool:
        """Upscale a single image using Real-CUGAN with comprehensive debugging"""
        start_time = time.time()
        gpu_status_start = self._monitor_gpu_status()
        
        try:
            logger.info(f"=== Real-CUGAN Debug Session Start ===")
            logger.info(f"Input: {input_path}")
            logger.info(f"Output: {output_path}")
            logger.info(f"Scale: {scale_factor}")
            logger.info(f"GPU Status: {gpu_status_start}")
            
            realcugan_path = self.resource_manager.get_binary_path('realcugan-ncnn-vulkan')
            if not realcugan_path:
                logger.error("Real-CUGAN executable not found")
                return False
            
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get model name (Real-CUGAN expects relative path from executable directory)
            model_name = self.models[self.current_model]['file']
            executable_dir = Path(realcugan_path).parent
            model_path = executable_dir / model_name
            
            # Check if model directory exists
            if not model_path.exists():
                logger.error(f"Real-CUGAN model directory not found: {model_path}")
                return False
            
            # Build Real-CUGAN command
            cmd = [
                realcugan_path,
                '-i', str(input_path),
                '-o', str(output_path),
                '-s', str(int(scale_factor)),  # Scale factor (1, 2, 3, 4)
                '-m', model_name,  # Model path (relative to working directory)
            ]
            
            # Add GPU acceleration if available
            vulkan_available = self.gpu_info.get('vulkan', {}).get('available', False)
            amd_gpu_available = len(self.gpu_info.get('amd', {}).get('gpus', [])) > 0
            nvidia_gpu_available = len(self.gpu_info.get('nvidia', {}).get('gpus', [])) > 0
            
            # Try GPU acceleration if we have any GPU available
            if vulkan_available or amd_gpu_available or nvidia_gpu_available:
                cmd.extend(['-g', '0'])  # Use GPU 0
                logger.info(f"Using GPU acceleration - Vulkan: {vulkan_available}, AMD: {amd_gpu_available}, NVIDIA: {nvidia_gpu_available}")
            else:
                cmd.extend(['-g', '-1'])  # Use CPU
                logger.info("Using CPU processing - no GPU detected")
            
            # Add additional optimization flags
            cmd.extend([
                '-j', '1:1:1',  # Load/Process/Save thread count
                '-x',  # Use TTA mode for better quality
            ])
            
            logger.info(f"Real-CUGAN command: {' '.join(cmd)}")
            logger.info(f"Working directory: {Path(realcugan_path).parent}")
            
            # Monitor process execution with timeout and status updates
            executable_dir = Path(realcugan_path).parent
            process_start_time = time.time()
            
            logger.info("Starting Real-CUGAN process...")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                cwd=str(executable_dir),  # Set working directory
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=300  # 5 minute timeout to prevent hanging
            )
            
            execution_time = time.time() - process_start_time
            gpu_status_end = self._monitor_gpu_status()
            
            logger.info(f"=== Real-CUGAN Debug Session Complete ===")
            logger.info(f"Execution time: {execution_time:.2f}s")
            logger.info(f"Return code: {result.returncode}")
            logger.info(f"GPU Status End: {gpu_status_end}")
            
            if result.stdout:
                logger.info(f"Real-CUGAN stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"Real-CUGAN stderr: {result.stderr}")
            
            # Check for specific error patterns
            if result.stderr:
                error_lower = result.stderr.lower()
                if 'idle' in error_lower or 'timeout' in error_lower:
                    logger.error("DETECTED: GPU Driver idle/timeout error!")
                if 'vulkan' in error_lower and 'error' in error_lower:
                    logger.error("DETECTED: Vulkan driver error!")
                if 'memory' in error_lower and ('out of' in error_lower or 'insufficient' in error_lower):
                    logger.error("DETECTED: GPU memory error!")
            
            if result.returncode == 0:
                # Verify output file was created
                if Path(output_path).exists():
                    logger.info(f"Real-CUGAN upscaling successful: {input_path} -> {output_path}")
                    return True
                else:
                    logger.error(f"Real-CUGAN output file not created: {output_path}")
                    return False
            else:
                logger.error(f"Real-CUGAN failed with return code {result.returncode}")
                logger.error(f"Real-CUGAN stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired as e:
            gpu_status_timeout = self._monitor_gpu_status()
            execution_time = time.time() - start_time
            logger.error(f"=== Real-CUGAN TIMEOUT ERROR ===")
            logger.error(f"Process timed out after {execution_time:.2f}s")
            logger.error(f"GPU Status at timeout: {gpu_status_timeout}")
            logger.error(f"This likely indicates GPU driver idle detection")
            logger.error(f"Timeout details: {e}")
            return False
            
        except Exception as e:
            gpu_status_error = self._monitor_gpu_status()
            execution_time = time.time() - start_time
            logger.error(f"=== Real-CUGAN UNEXPECTED ERROR ===")
            logger.error(f"Execution time: {execution_time:.2f}s")
            logger.error(f"GPU Status at error: {gpu_status_error}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {e}")
            logger.error(f"This may indicate driver issues or resource conflicts")
            return False
    
    def estimate_processing_time(self, image_path: str, scale_factor: float = 2.0) -> float:
        """Estimate processing time for an image"""
        try:
            # Get image dimensions
            with Image.open(image_path) as img:
                width, height = img.size
                pixels = width * height
            
            # Base processing time per megapixel (GPU vs CPU)
            vulkan_available = self.gpu_info.get('vulkan', {}).get('available', False)
            amd_gpu_available = len(self.gpu_info.get('amd', {}).get('gpus', [])) > 0
            nvidia_gpu_available = len(self.gpu_info.get('nvidia', {}).get('gpus', [])) > 0
            
            if vulkan_available or amd_gpu_available or nvidia_gpu_available:
                # GPU processing - much faster
                base_time_per_mp = 0.5  # seconds per megapixel
                gpu_factor = 1.0
            else:
                # CPU processing - slower
                base_time_per_mp = 3.0  # seconds per megapixel  
                gpu_factor = 6.0
            
            # Model complexity factor
            model_factors = {
                'conservative': 1.0,
                'denoise1x': 1.2,
                'denoise2x': 1.5,
                'denoise3x': 2.0
            }
            
            model_factor = model_factors.get(self.current_model, 1.5)
            
            # Scale factor impact
            scale_impact = scale_factor ** 1.5
            
            # Calculate estimate
            megapixels = pixels / 1000000
            estimated_time = megapixels * base_time_per_mp * model_factor * scale_impact * gpu_factor
            
            return max(0.1, estimated_time)  # Minimum 0.1 seconds
            
        except Exception as e:
            logger.warning(f"Could not estimate processing time: {e}")
            return 2.0  # Default estimate
    
    def get_supported_scales(self) -> list:
        """Get supported scale factors"""
        return [1.0, 2.0, 3.0, 4.0]
    
    def cleanup(self):
        """Clean up backend resources"""
        logger.info("Real-CUGAN backend cleanup completed")
    
    def get_gpu_usage_info(self) -> Dict[str, Any]:
        """Get GPU usage information"""
        vulkan_available = self.gpu_info.get('vulkan', {}).get('available', False)
        amd_gpu_available = len(self.gpu_info.get('amd', {}).get('gpus', [])) > 0
        nvidia_gpu_available = len(self.gpu_info.get('nvidia', {}).get('gpus', [])) > 0
        
        using_gpu = vulkan_available or amd_gpu_available or nvidia_gpu_available
        
        # Determine GPU type and device name
        gpu_type = 'CPU'
        gpu_device = 'CPU Only'
        
        if vulkan_available:
            gpu_type = 'Vulkan'
            vulkan_devices = self.gpu_info.get('vulkan', {}).get('devices', [])
            if vulkan_devices:
                gpu_device = vulkan_devices[0].get('name', 'Unknown Vulkan Device')
        elif amd_gpu_available:
            gpu_type = 'AMD'
            amd_gpus = self.gpu_info.get('amd', {}).get('gpus', [])
            if amd_gpus:
                gpu_device = amd_gpus[0].get('name', 'Unknown AMD GPU')
        elif nvidia_gpu_available:
            gpu_type = 'NVIDIA'
            nvidia_gpus = self.gpu_info.get('nvidia', {}).get('gpus', [])
            if nvidia_gpus:
                gpu_device = nvidia_gpus[0].get('name', 'Unknown NVIDIA GPU')
        
        return {
            'using_gpu': using_gpu,
            'gpu_type': gpu_type,
            'gpu_device': gpu_device,
            'vulkan_available': vulkan_available,
            'amd_gpu_count': len(self.gpu_info.get('amd', {}).get('gpus', [])),
            'nvidia_gpu_count': len(self.gpu_info.get('nvidia', {}).get('gpus', []))
        }