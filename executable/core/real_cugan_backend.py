"""
Real-CUGAN Backend for AI Processing
Optimized for anime and illustration upscaling
"""

import os
import subprocess
import tempfile
import logging
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
        gpu_available = self.gpu_info.get('vulkan', {}).get('available', False)
        return {
            'name': 'Real-CUGAN',
            'version': 'NCNN-Vulkan',
            'description': 'Real-CUGAN for anime/illustration upscaling',
            'gpu_acceleration': gpu_available,
            'gpu_mode': gpu_available,  # Add gpu_mode for compatibility
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
    
    def upscale_image(self, input_path: str, output_path: str, scale_factor: float = 2.0, progress_dialog=None) -> bool:
        """Upscale a single image using Real-CUGAN"""
        try:
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
            if self.gpu_info.get('vulkan', {}).get('available'):
                cmd.extend(['-g', '0'])  # Use GPU 0
                logger.debug("Using Vulkan GPU acceleration")
            else:
                cmd.extend(['-g', '-1'])  # Use CPU
                logger.debug("Using CPU processing")
            
            # Add additional optimization flags
            cmd.extend([
                '-j', '1:1:1',  # Load/Process/Save thread count
                '-x',  # Use TTA mode for better quality
            ])
            
            logger.info(f"Real-CUGAN command: {' '.join(cmd)}")
            
            # Execute Real-CUGAN with correct working directory
            executable_dir = Path(realcugan_path).parent
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                cwd=str(executable_dir),  # Set working directory
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            logger.info(f"Real-CUGAN execution completed - returncode: {result.returncode}")
            if result.stdout:
                logger.info(f"Real-CUGAN stdout: {result.stdout}")
            if result.stderr:
                logger.info(f"Real-CUGAN stderr: {result.stderr}")
            
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
                
        except Exception as e:
            logger.error(f"Real-CUGAN upscaling error: {e}")
            return False
    
    def estimate_processing_time(self, image_path: str, scale_factor: float = 2.0) -> float:
        """Estimate processing time for an image"""
        try:
            # Get image dimensions
            with Image.open(image_path) as img:
                width, height = img.size
                pixels = width * height
            
            # Base processing time per megapixel (GPU vs CPU)
            if self.gpu_info.get('vulkan', {}).get('available'):
                # GPU processing (Vulkan) - much faster
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
        return {
            'using_gpu': self.gpu_info.get('vulkan', {}).get('available', False),
            'gpu_type': 'Vulkan',
            'gpu_device': self.gpu_info.get('vulkan', {}).get('devices', [{}])[0].get('name', 'Unknown') if self.gpu_info.get('vulkan', {}).get('devices') else 'Unknown'
        }