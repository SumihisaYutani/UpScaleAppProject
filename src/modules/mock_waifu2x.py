"""
Mock Waifu2x Implementation for Testing GUI without Real Dependencies
This provides a fallback implementation when actual waifu2x packages are not available
"""

import logging
from PIL import Image
from typing import List, Optional, Callable, Dict, Any
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class MockWaifu2xUpscaler:
    """
    Mock implementation of Waifu2x upscaler for GUI testing
    Performs simple upscaling using traditional algorithms
    """
    
    def __init__(self, 
                 backend: str = "mock",
                 gpu_id: int = 0,
                 scale: int = 2,
                 noise: int = 1,
                 model: str = "models-cunet"):
        self.backend = "mock"
        self.gpu_id = gpu_id
        self.scale = scale
        self.noise = noise
        self.model = model
        self._available = True
        
        logger.info(f"Mock Waifu2x initialized - Scale: {self.scale}x, Noise: {self.noise}")
    
    def is_available(self) -> bool:
        """Check if mock waifu2x is available"""
        return self._available
    
    def upscale_image(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Mock upscale using simple algorithms
        Simulates waifu2x processing time and applies basic upscaling
        """
        if not self._available:
            logger.error("Mock Waifu2x not available")
            return None
        
        try:
            # Convert RGBA to RGB if necessary
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Simulate processing time
            time.sleep(0.1)  # Brief delay to simulate processing
            
            # Calculate new dimensions
            width, height = image.size
            new_width = int(width * self.scale)
            new_height = int(height * self.scale)
            
            # Use LANCZOS for high quality resizing (simulating waifu2x)
            upscaled = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Apply slight sharpening to simulate AI enhancement
            from PIL import ImageFilter, ImageEnhance
            
            # Apply unsharp mask for better edge definition
            upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            
            # Enhance contrast slightly
            enhancer = ImageEnhance.Contrast(upscaled)
            upscaled = enhancer.enhance(1.1)
            
            logger.debug(f"Mock upscaled image from {width}x{height} to {new_width}x{new_height}")
            return upscaled
            
        except Exception as e:
            logger.error(f"Mock upscaling failed: {e}")
            return None
    
    def upscale_frames(self, 
                      frame_files: List[str], 
                      output_dir: str,
                      progress_callback: Optional[Callable] = None) -> List[str]:
        """Mock upscale multiple frame files"""
        if not self._available:
            logger.error("Mock Waifu2x not available")
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
                    output_file = output_dir / f"{frame_name}_mock_waifu2x.png"
                    upscaled.save(output_file, "PNG")
                    upscaled_files.append(str(output_file))
                
                # Progress callback
                if progress_callback:
                    progress = (i + 1) / total_frames * 100
                    progress_callback(progress, f"Mock upscaled frame {i+1}/{total_frames}")
                    
            except Exception as e:
                logger.error(f"Failed to process frame {frame_file}: {e}")
                continue
        
        logger.info(f"Mock Waifu2x: Successfully upscaled {len(upscaled_files)}/{total_frames} frames")
        return upscaled_files
    
    def get_supported_scales(self) -> List[int]:
        """Get supported scale factors"""
        return [1, 2, 4, 8, 16, 32]
    
    def get_supported_noise_levels(self) -> List[int]:
        """Get supported noise reduction levels"""
        return [-1, 0, 1, 2, 3]
    
    def get_available_models(self) -> List[str]:
        """Get available model types"""
        return [
            "models-cunet",
            "models-upconv_7_anime_style_art_rgb",
            "models-upconv_7_photo",
        ]
    
    def update_settings(self, 
                       scale: Optional[int] = None,
                       noise: Optional[int] = None,
                       model: Optional[str] = None):
        """Update processing settings"""
        if scale is not None:
            self.scale = scale
            logger.info(f"Mock Waifu2x: Updated scale to {scale}")
            
        if noise is not None:
            self.noise = noise
            logger.info(f"Mock Waifu2x: Updated noise level to {noise}")
            
        if model is not None:
            self.model = model
            logger.info(f"Mock Waifu2x: Updated model to {model}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get processor information"""
        return {
            "backend": "mock",
            "available": self._available,
            "gpu_id": self.gpu_id,
            "scale": self.scale,
            "noise": self.noise,
            "model": self.model,
            "supported_scales": self.get_supported_scales(),
            "supported_noise_levels": self.get_supported_noise_levels(),
            "available_models": self.get_available_models(),
            "note": "This is a mock implementation for testing purposes"
        }


def test_mock_waifu2x_availability() -> Dict[str, bool]:
    """Test mock waifu2x availability"""
    return {
        "mock": True,
        "ncnn": False,
        "chainer": False,
        "any_available": True
    }


def create_mock_waifu2x_upscaler(**kwargs) -> MockWaifu2xUpscaler:
    """Factory function to create mock waifu2x upscaler"""
    return MockWaifu2xUpscaler(**kwargs)