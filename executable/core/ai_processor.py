"""
UpScale App - AI Processing Module  
Consolidated AI upscaling using available backends
"""

import os
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from PIL import Image
import shutil

from .utils import ProgressCallback, SimpleTimer

logger = logging.getLogger(__name__)

class AIProcessor:
    """Handles AI-based image upscaling using available backends"""
    
    def __init__(self, resource_manager, gpu_info: Dict):
        self.resource_manager = resource_manager
        self.gpu_info = gpu_info
        self.best_backend = gpu_info.get('best_backend', 'cpu')
        
        # Initialize backend
        self.backend = None
        self._initialize_backend()
        
        logger.info(f"AIProcessor initialized - Backend: {self.best_backend}")
    
    def _initialize_backend(self):
        """Initialize the best available AI backend"""
        try:
            logger.info(f"Initializing backend for: {self.best_backend}")
            
            # Always try Waifu2x first for any GPU backend
            if self.best_backend in ['nvidia', 'nvidia_cuda', 'amd', 'amd_rocm', 'vulkan', 'intel']:
                logger.info("Attempting to initialize Waifu2x backend")
                self.backend = Waifu2xExecutableBackend(self.resource_manager, self.gpu_info)
                
                if self.backend.is_available():
                    logger.info(f"Waifu2x backend initialized successfully: {type(self.backend).__name__}")
                    return
                else:
                    logger.warning("Waifu2x backend not available, falling back to simple upscaling")
            
            # Fallback to simple upscaling
            logger.info("Using simple upscaling backend")
            self.backend = SimpleUpscalingBackend()
                
        except Exception as e:
            logger.error(f"Failed to initialize AI backend: {e}")
            import traceback
            traceback.print_exc()
            self.backend = SimpleUpscalingBackend()
    
    def upscale_frames(self, frame_paths: List[str], output_dir: str, 
                      scale_factor: float = 2.0,
                      progress_callback: Optional[Callable] = None) -> List[str]:
        """Upscale a list of frame images"""
        
        if not self.backend:
            raise RuntimeError("No AI backend available")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_frames = []
        total_frames = len(frame_paths)
        
        logger.info(f"Processing {total_frames} frames with {type(self.backend).__name__}")
        
        for i, frame_path in enumerate(frame_paths):
            try:
                # Generate output path
                frame_name = Path(frame_path).stem
                output_path = output_dir / f"{frame_name}_upscaled.png"
                
                # Upscale frame
                success = self.backend.upscale_image(frame_path, str(output_path), scale_factor)
                
                if success:
                    processed_frames.append(str(output_path))
                else:
                    logger.warning(f"Failed to upscale frame: {frame_path}")
                
                # Update progress
                if progress_callback:
                    progress = (i + 1) / total_frames * 100
                    progress_callback(progress, f"Processed frame {i+1}/{total_frames}")
                    
            except Exception as e:
                logger.error(f"Error processing frame {frame_path}: {e}")
        
        logger.info(f"Successfully processed {len(processed_frames)}/{total_frames} frames")
        return processed_frames
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the active backend"""
        if self.backend:
            return self.backend.get_info()
        return {"backend": "none", "available": False}
    
    def cleanup(self):
        """Clean up AI processor resources"""
        if self.backend:
            self.backend.cleanup()

class Waifu2xExecutableBackend:
    """Waifu2x backend using executable binary"""
    
    def __init__(self, resource_manager, gpu_info: Dict):
        self.resource_manager = resource_manager
        self.gpu_info = gpu_info
        self.waifu2x_path = resource_manager.get_binary_path('waifu2x')
        self.gpu_id = self._determine_gpu_id()
        
    def _determine_gpu_id(self) -> int:
        """Determine best GPU ID to use"""
        best_backend = self.gpu_info.get('best_backend', 'cpu')
        
        if best_backend == 'cpu':
            return -1  # CPU mode
        
        # Try to use first available GPU
        if self.gpu_info.get('nvidia', {}).get('available'):
            return 0  # First NVIDIA GPU
        elif self.gpu_info.get('amd', {}).get('available'):
            return 0  # First AMD GPU
        
        return -1  # Default to CPU
    
    def is_available(self) -> bool:
        """Check if Waifu2x executable is available"""
        if not self.waifu2x_path:
            logger.warning("Waifu2x executable path not found")
            return False
        
        try:
            # Test if the executable actually works
            from pathlib import Path
            if not Path(self.waifu2x_path).exists():
                logger.warning(f"Waifu2x executable not found at: {self.waifu2x_path}")
                return False
            
            # Quick test execution to verify it works
            result = self.resource_manager.run_binary('waifu2x', ['-h'], timeout=10, check=False)
            if result.returncode == 0 or 'waifu2x' in result.stdout.lower() or 'usage' in result.stdout.lower():
                logger.info("Waifu2x executable is available and working")
                return True
            else:
                logger.warning(f"Waifu2x executable test failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.warning(f"Waifu2x availability check failed: {e}")
            return False
    
    def upscale_image(self, input_path: str, output_path: str, scale_factor: float = 2.0) -> bool:
        """Upscale single image using Waifu2x"""
        try:
            if not self.waifu2x_path:
                return False
            
            # Convert scale factor to integer (Waifu2x supports 1, 2, 4, 8, 16, 32)
            scale_int = max(1, min(32, int(scale_factor)))
            
            cmd = [
                self.waifu2x_path,
                '-i', input_path,
                '-o', output_path,
                '-s', str(scale_int),
                '-n', '1',  # Noise reduction level
                '-g', str(self.gpu_id),  # GPU ID (-1 for CPU)
                '-m', 'models-cunet',  # Model type
                '-f', 'png'  # Output format
            ]
            
            # Run waifu2x
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=120  # 2 minute timeout per image
            )
            
            if result.returncode != 0:
                logger.debug(f"Waifu2x failed: {result.stderr}")
                return False
            
            # Verify output file exists
            return Path(output_path).exists()
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Waifu2x timed out for image: {input_path}")
            return False
        except Exception as e:
            logger.debug(f"Waifu2x processing error: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get backend information"""
        return {
            "backend": "waifu2x_executable",
            "available": self.is_available(),
            "executable_path": self.waifu2x_path,
            "gpu_id": self.gpu_id,
            "gpu_mode": self.gpu_id >= 0
        }
    
    def cleanup(self):
        """Cleanup backend resources"""
        pass  # Nothing to cleanup for executable backend

class SimpleUpscalingBackend:
    """Fallback simple upscaling using PIL"""
    
    def __init__(self):
        self.method = Image.Resampling.LANCZOS
        
    def is_available(self) -> bool:
        """Simple upscaling is always available"""
        return True
    
    def upscale_image(self, input_path: str, output_path: str, scale_factor: float = 2.0) -> bool:
        """Simple upscaling using PIL"""
        try:
            with Image.open(input_path) as img:
                # Convert RGBA to RGB if necessary
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate new size
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)
                
                # Upscale image
                upscaled = img.resize((new_width, new_height), self.method)
                
                # Save result
                upscaled.save(output_path, 'PNG', quality=95)
                
                return True
                
        except Exception as e:
            logger.error(f"Simple upscaling failed: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get backend information"""
        return {
            "backend": "simple_upscaling",
            "available": True,
            "method": "PIL Lanczos",
            "gpu_mode": False
        }
    
    def cleanup(self):
        """Cleanup backend resources"""
        pass  # Nothing to cleanup for simple backend