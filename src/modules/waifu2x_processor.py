"""
Waifu2x Processing Module
High-quality image upscaling using waifu2x AI models
Optimized for anime-style images and general purpose upscaling
"""

try:
    from waifu2x_ncnn_vulkan import Waifu2x
    WAIFU2X_NCNN_AVAILABLE = True
except ImportError:
    WAIFU2X_NCNN_AVAILABLE = False

try:
    import waifu2x
    WAIFU2X_CHAINER_AVAILABLE = True
except ImportError:
    WAIFU2X_CHAINER_AVAILABLE = False

# Import mock implementation as fallback
try:
    from .mock_waifu2x import MockWaifu2xUpscaler, test_mock_waifu2x_availability
    MOCK_WAIFU2X_AVAILABLE = True
except ImportError:
    try:
        from src.modules.mock_waifu2x import MockWaifu2xUpscaler, test_mock_waifu2x_availability
        MOCK_WAIFU2X_AVAILABLE = True
    except ImportError:
        MOCK_WAIFU2X_AVAILABLE = False

import logging
from PIL import Image
import numpy as np
from typing import List, Optional, Callable, Dict, Any
from pathlib import Path
import tempfile
import os

logger = logging.getLogger(__name__)


class Waifu2xUpscaler:
    """
    High-quality image upscaling using waifu2x models
    Supports both ncnn-vulkan and chainer implementations
    """
    
    def __init__(self, 
                 backend: str = "auto",
                 gpu_id: int = 0,
                 scale: int = 2,
                 noise: int = 1,
                 model: str = "models-cunet"):
        """
        Initialize Waifu2x upscaler
        
        Args:
            backend: "ncnn", "chainer", or "auto"
            gpu_id: GPU device ID (0 for first GPU, -1 for CPU)
            scale: Scaling factor (1, 2, 4, 8, 16, 32)
            noise: Noise reduction level (-1: none, 0-3: weak to strong)
            model: Model type for ncnn backend
        """
        self.backend = backend
        self.gpu_id = gpu_id
        self.scale = scale
        self.noise = noise
        self.model = model
        self._processor = None
        self._available = False
        
        # Determine best available backend
        if backend == "auto":
            if WAIFU2X_NCNN_AVAILABLE:
                self.backend = "ncnn"
            elif WAIFU2X_CHAINER_AVAILABLE:
                self.backend = "chainer"
            elif MOCK_WAIFU2X_AVAILABLE:
                self.backend = "mock"
            else:
                logger.error("No waifu2x backend available")
                return
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the selected backend"""
        try:
            if self.backend == "ncnn" and WAIFU2X_NCNN_AVAILABLE:
                self._initialize_ncnn()
            elif self.backend == "chainer" and WAIFU2X_CHAINER_AVAILABLE:
                self._initialize_chainer()
            elif self.backend == "mock" and MOCK_WAIFU2X_AVAILABLE:
                self._initialize_mock()
            else:
                logger.error(f"Backend {self.backend} not available")
                return
                
            self._available = True
            logger.info(f"Waifu2x initialized with {self.backend} backend")
            
        except Exception as e:
            logger.error(f"Failed to initialize waifu2x backend: {e}")
            self._available = False
    
    def _initialize_ncnn(self):
        """Initialize ncnn-vulkan backend"""
        try:
            self._processor = Waifu2x(
                gpuid=self.gpu_id,
                scale=self.scale,
                noise=self.noise,
                model=self.model
            )
            logger.info(f"NCNN backend initialized - GPU: {self.gpu_id}, Scale: {self.scale}x, Noise: {self.noise}")
        except Exception as e:
            logger.error(f"Failed to initialize NCNN backend: {e}")
            raise
    
    def _initialize_chainer(self):
        """Initialize chainer backend"""
        # Chainer backend initialization would go here
        # This is a placeholder as the actual implementation depends on the specific package
        logger.warning("Chainer backend not fully implemented yet")
        raise NotImplementedError("Chainer backend not implemented")
    
    def _initialize_mock(self):
        """Initialize mock backend"""
        try:
            self._processor = MockWaifu2xUpscaler(
                backend="mock",
                gpu_id=self.gpu_id,
                scale=self.scale,
                noise=self.noise,
                model=self.model
            )
            logger.info(f"Mock backend initialized - Scale: {self.scale}x, Noise: {self.noise}")
        except Exception as e:
            logger.error(f"Failed to initialize mock backend: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if waifu2x is available"""
        return self._available
    
    def upscale_image(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Upscale a single image using waifu2x
        
        Args:
            image: PIL Image to upscale
            
        Returns:
            Upscaled PIL Image or None if failed
        """
        if not self._available:
            logger.error("Waifu2x not available")
            return None
            
        try:
            if self.backend == "ncnn":
                return self._upscale_ncnn(image)
            elif self.backend == "chainer":
                return self._upscale_chainer(image)
            elif self.backend == "mock":
                return self._upscale_mock(image)
            else:
                logger.error(f"Unknown backend: {self.backend}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to upscale image: {e}")
            return None
    
    def _upscale_ncnn(self, image: Image.Image) -> Optional[Image.Image]:
        """Upscale using ncnn-vulkan backend"""
        try:
            # Convert RGBA to RGB if necessary
            if image.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process with waifu2x
            upscaled = self._processor.process(image)
            
            if upscaled is None:
                logger.error("Waifu2x processing returned None")
                return None
                
            return upscaled
            
        except Exception as e:
            logger.error(f"NCNN upscaling failed: {e}")
            return None
    
    def _upscale_chainer(self, image: Image.Image) -> Optional[Image.Image]:
        """Upscale using chainer backend"""
        # Placeholder for chainer implementation
        raise NotImplementedError("Chainer backend not implemented")
    
    def _upscale_mock(self, image: Image.Image) -> Optional[Image.Image]:
        """Upscale using mock backend"""
        try:
            return self._processor.upscale_image(image)
        except Exception as e:
            logger.error(f"Mock upscaling failed: {e}")
            return None
    
    def upscale_frames(self, 
                      frame_files: List[str], 
                      output_dir: str,
                      progress_callback: Optional[Callable] = None) -> List[str]:
        """
        Upscale multiple frame files
        
        Args:
            frame_files: List of input frame file paths
            output_dir: Output directory for upscaled frames
            progress_callback: Optional progress callback function
            
        Returns:
            List of output frame file paths
        """
        if not self._available:
            logger.error("Waifu2x not available")
            return []
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        upscaled_files = []
        total_frames = len(frame_files)
        
        for i, frame_file in enumerate(frame_files):
            try:
                # Load frame
                with Image.open(frame_file) as image:
                    # Upscale
                    upscaled = self.upscale_image(image)
                    
                    if upscaled is None:
                        logger.warning(f"Failed to upscale frame: {frame_file}")
                        continue
                    
                    # Save upscaled frame
                    frame_name = Path(frame_file).stem
                    output_file = output_dir / f"{frame_name}_waifu2x.png"
                    upscaled.save(output_file, "PNG")
                    upscaled_files.append(str(output_file))
                
                # Progress callback
                if progress_callback:
                    progress = (i + 1) / total_frames * 100
                    progress_callback(progress, f"Upscaled frame {i+1}/{total_frames}")
                    
            except Exception as e:
                logger.error(f"Failed to process frame {frame_file}: {e}")
                continue
        
        logger.info(f"Successfully upscaled {len(upscaled_files)}/{total_frames} frames")
        return upscaled_files
    
    def get_supported_scales(self) -> List[int]:
        """Get supported scale factors"""
        if self.backend == "ncnn":
            return [1, 2, 4, 8, 16, 32]
        else:
            return [1, 2, 4]  # Default for other backends
    
    def get_supported_noise_levels(self) -> List[int]:
        """Get supported noise reduction levels"""
        return [-1, 0, 1, 2, 3]  # -1: none, 0-3: weak to strong
    
    def get_available_models(self) -> List[str]:
        """Get available model types"""
        if self.backend == "ncnn":
            return [
                "models-cunet",      # Default, balanced quality/speed
                "models-upconv_7_anime_style_art_rgb",  # Anime style
                "models-upconv_7_photo",  # Photographic images
            ]
        else:
            return ["default"]
    
    def update_settings(self, 
                       scale: Optional[int] = None,
                       noise: Optional[int] = None,
                       model: Optional[str] = None):
        """
        Update processing settings
        
        Args:
            scale: New scale factor
            noise: New noise reduction level
            model: New model type
        """
        settings_changed = False
        
        if scale is not None and scale != self.scale:
            self.scale = scale
            settings_changed = True
            
        if noise is not None and noise != self.noise:
            self.noise = noise
            settings_changed = True
            
        if model is not None and model != self.model:
            self.model = model
            settings_changed = True
        
        # Reinitialize if settings changed
        if settings_changed and self._available:
            logger.info("Reinitializing waifu2x with new settings")
            self._initialize_backend()
    
    def get_info(self) -> Dict[str, Any]:
        """Get processor information"""
        return {
            "backend": self.backend,
            "available": self._available,
            "gpu_id": self.gpu_id,
            "scale": self.scale,
            "noise": self.noise,
            "model": self.model,
            "supported_scales": self.get_supported_scales(),
            "supported_noise_levels": self.get_supported_noise_levels(),
            "available_models": self.get_available_models()
        }


def create_waifu2x_upscaler(**kwargs) -> Waifu2xUpscaler:
    """Factory function to create waifu2x upscaler"""
    return Waifu2xUpscaler(**kwargs)


def test_waifu2x_availability() -> Dict[str, bool]:
    """Test waifu2x backend availability"""
    return {
        "ncnn": WAIFU2X_NCNN_AVAILABLE,
        "chainer": WAIFU2X_CHAINER_AVAILABLE,
        "mock": MOCK_WAIFU2X_AVAILABLE,
        "any_available": WAIFU2X_NCNN_AVAILABLE or WAIFU2X_CHAINER_AVAILABLE or MOCK_WAIFU2X_AVAILABLE
    }


# Convenience class for backward compatibility
class Waifu2xProcessor(Waifu2xUpscaler):
    """Alias for backward compatibility"""
    pass