"""
Real-ESRGAN Backend for AI Processing
Optimized for photographic and general image upscaling
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

class RealESRGANBackend:
    """Real-ESRGAN backend for high-quality photographic image upscaling"""
    
    def __init__(self, resource_manager, gpu_info: Dict[str, Any]):
        self.resource_manager = resource_manager
        self.gpu_info = gpu_info
        self.available = False
        
        # Real-ESRGAN model configurations (using actual NCNN models)
        self.models = {
            'general_x4': {
                'name': 'general_x4',
                'description': 'General purpose 4x upscaling',
                'file': 'realesrgan-x4plus',
                'scale': 4
            },
            'anime_x4': {
                'name': 'anime_x4', 
                'description': 'Anime/illustration 4x upscaling',
                'file': 'realesrgan-x4plus-anime',
                'scale': 4
            },
            'anime_x2': {
                'name': 'anime_x2',
                'description': 'Anime video 2x upscaling',
                'file': 'realesr-animevideov3-x2',
                'scale': 2
            },
            'anime_x3': {
                'name': 'anime_x3',
                'description': 'Anime video 3x upscaling',
                'file': 'realesr-animevideov3-x3',
                'scale': 3
            }
        }
        
        self.default_model = 'general_x4'  # Best general purpose model
        self.current_model = self.default_model
        
        logger.info("Initializing Real-ESRGAN backend...")
        self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if Real-ESRGAN is available and functional"""
        try:
            # Check for Real-ESRGAN executable
            realesrgan_path = self.resource_manager.get_binary_path('realesrgan-ncnn-vulkan')
            if not realesrgan_path:
                logger.warning("Real-ESRGAN executable not found")
                return False
            
            # Test Real-ESRGAN with help command
            result = subprocess.run([realesrgan_path, '-h'], 
                                  capture_output=True, text=True, timeout=10,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
            
            # Real-ESRGAN outputs help to stderr and returns non-zero exit code for help
            if ('realesrgan-ncnn-vulkan' in result.stderr or 'Usage:' in result.stderr or
                'model-name' in result.stderr or 'realesr-animevideov3' in result.stderr):
                logger.info("Real-ESRGAN executable found and functional")
                self.available = True
                
                # Log GPU compatibility
                if self.gpu_info.get('vulkan', {}).get('available'):
                    logger.info("Vulkan GPU acceleration available for Real-ESRGAN")
                else:
                    logger.warning("No Vulkan support detected - Real-ESRGAN will use CPU")
                
                return True
            else:
                logger.warning(f"Real-ESRGAN test failed: stdout={result.stdout}, stderr={result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Real-ESRGAN availability check failed: {e}")
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
            'backend': 'Real-ESRGAN',  # Add backend key for GUI compatibility
            'name': 'Real-ESRGAN',
            'version': 'NCNN-Vulkan',
            'description': 'Real-ESRGAN for photographic image upscaling',
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
        """Set the Real-ESRGAN model to use"""
        if model_name in self.models:
            self.current_model = model_name
            logger.info(f"Real-ESRGAN model set to: {model_name} ({self.models[model_name]['description']})")
            return True
        else:
            logger.warning(f"Unknown Real-ESRGAN model: {model_name}")
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

    def upscale_image(self, input_path: str, output_path: str, scale_factor: float = 4.0, progress_dialog=None) -> bool:
        """Upscale a single image using Real-ESRGAN with comprehensive debugging"""
        start_time = time.time()
        gpu_status_start = self._monitor_gpu_status()
        
        try:
            logger.info(f"=== Real-ESRGAN Debug Session Start ===")
            logger.info(f"Input: {input_path}")
            logger.info(f"Output: {output_path}")
            logger.info(f"Scale: {scale_factor}")
            logger.info(f"GPU Status: {gpu_status_start}")
            
            realesrgan_path = self.resource_manager.get_binary_path('realesrgan-ncnn-vulkan')
            if not realesrgan_path:
                logger.error("Real-ESRGAN executable not found")
                return False
            
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get model name and scale
            model_info = self.models[self.current_model]
            model_name = model_info['file']
            model_scale = model_info['scale']
            
            # Adjust scale factor to match model capabilities
            if scale_factor != model_scale:
                logger.warning(f"Requested scale {scale_factor} doesn't match model scale {model_scale}, using model scale")
                scale_factor = model_scale
            
            # Build Real-ESRGAN command (NCNN version uses different parameters)
            cmd = [
                realesrgan_path,
                '-i', str(input_path),
                '-o', str(output_path),
                '-n', model_name,  # NCNN model name
                '-s', str(int(scale_factor)),  # Scale factor
                '-f', 'png',  # Output format
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
            
            logger.info(f"Real-ESRGAN command: {' '.join(cmd)}")
            
            # Monitor process execution with timeout and status updates
            process_start_time = time.time()
            
            logger.info("Starting Real-ESRGAN process...")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=600  # 10 minute timeout for large images
            )
            
            execution_time = time.time() - process_start_time
            gpu_status_end = self._monitor_gpu_status()
            
            logger.info(f"=== Real-ESRGAN Debug Session Complete ===")
            logger.info(f"Execution time: {execution_time:.2f}s")
            logger.info(f"Return code: {result.returncode}")
            logger.info(f"GPU Status End: {gpu_status_end}")
            
            if result.stdout:
                logger.info(f"Real-ESRGAN stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"Real-ESRGAN stderr: {result.stderr}")
            
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
                    logger.info(f"Real-ESRGAN upscaling successful: {input_path} -> {output_path}")
                    return True
                else:
                    logger.error(f"Real-ESRGAN output file not created: {output_path}")
                    return False
            else:
                logger.error(f"Real-ESRGAN failed with return code {result.returncode}")
                logger.error(f"Real-ESRGAN stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired as e:
            gpu_status_timeout = self._monitor_gpu_status()
            execution_time = time.time() - start_time
            logger.error(f"=== Real-ESRGAN TIMEOUT ERROR ===")
            logger.error(f"Process timed out after {execution_time:.2f}s")
            logger.error(f"GPU Status at timeout: {gpu_status_timeout}")
            logger.error(f"This likely indicates GPU driver idle detection or large image processing")
            logger.error(f"Timeout details: {e}")
            return False
            
        except Exception as e:
            gpu_status_error = self._monitor_gpu_status()
            execution_time = time.time() - start_time
            logger.error(f"=== Real-ESRGAN UNEXPECTED ERROR ===")
            logger.error(f"Execution time: {execution_time:.2f}s")
            logger.error(f"GPU Status at error: {gpu_status_error}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {e}")
            logger.error(f"This may indicate driver issues or resource conflicts")
            return False
    
    def estimate_processing_time(self, image_path: str, scale_factor: float = 4.0) -> float:
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
                # GPU processing - faster than CUGAN for photos
                base_time_per_mp = 0.8  # seconds per megapixel
                gpu_factor = 1.0
            else:
                # CPU processing - slower
                base_time_per_mp = 4.0  # seconds per megapixel  
                gpu_factor = 8.0
            
            # Model complexity factor
            model_factors = {
                'general_x2': 1.0,
                'general_x4': 1.5,
                'anime_x4': 1.3,
                'face_x2': 2.0  # Face enhancement is more complex
            }
            
            model_factor = model_factors.get(self.current_model, 1.5)
            
            # Scale factor impact (Real-ESRGAN handles fixed scales better)
            scale_impact = scale_factor ** 1.2
            
            # Calculate estimate
            megapixels = pixels / 1000000
            estimated_time = megapixels * base_time_per_mp * model_factor * scale_impact * gpu_factor
            
            return max(0.2, estimated_time)  # Minimum 0.2 seconds
            
        except Exception as e:
            logger.warning(f"Could not estimate processing time: {e}")
            return 3.0  # Default estimate
    
    def get_supported_scales(self) -> list:
        """Get supported scale factors based on current model"""
        model_info = self.models.get(self.current_model, self.models[self.default_model])
        scale = model_info['scale']
        return [float(scale)]  # Real-ESRGAN models have fixed scales
    
    def cleanup(self):
        """Clean up backend resources"""
        logger.info("Real-ESRGAN backend cleanup completed")
    
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