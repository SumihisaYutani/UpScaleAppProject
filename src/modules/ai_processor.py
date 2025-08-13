"""
AI Processing Module
Handles AI-powered image upscaling using Stable Diffusion and Waifu2x
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import logging
from PIL import Image
from typing import List, Optional, Callable, Dict, Any
from pathlib import Path
import numpy as np

try:
    from diffusers import StableDiffusionImg2ImgPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    from .waifu2x_processor import Waifu2xUpscaler, test_waifu2x_availability
    WAIFU2X_AVAILABLE = True
except ImportError:
    WAIFU2X_AVAILABLE = False

from config.settings import AI_SETTINGS, PERFORMANCE

logger = logging.getLogger(__name__)


class AIUpscaler:
    """AI-powered image upscaling using Stable Diffusion"""
    
    def __init__(self):
        if not TORCH_AVAILABLE or not DIFFUSERS_AVAILABLE:
            logger.warning("AI dependencies not available. AI upscaling disabled.")
            self._available = False
            return
            
        self.model_name = AI_SETTINGS["model_name"]
        self.device = AI_SETTINGS["device"]
        self.batch_size = AI_SETTINGS["batch_size"]
        self.guidance_scale = AI_SETTINGS["guidance_scale"]
        self.num_inference_steps = AI_SETTINGS["num_inference_steps"]
        self.pipeline = None
        self._model_loaded = False
        self._available = True
    
    def load_model(self) -> bool:
        """
        Load the Stable Diffusion model
        
        Returns:
            bool: Success status
        """
        if not self._available:
            logger.error("AI dependencies not available")
            return False
            
        try:
            if self._model_loaded:
                return True
            
            logger.info(f"Loading AI model: {self.model_name}")
            
            # Load pipeline with optimizations
            self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory efficient attention if available
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing()
            
            if hasattr(self.pipeline, "enable_vae_slicing"):
                self.pipeline.enable_vae_slicing()
            
            # Enable CPU offloading if needed
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    self.pipeline.enable_sequential_cpu_offload()
                except:
                    pass  # Not available in all versions
            
            self._model_loaded = True
            logger.info("AI model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load AI model: {e}")
            return False
    
    def upscale_image(self, image_path: str, output_path: str,
                     scale_factor: float = 1.5, prompt: str = None) -> bool:
        """
        Upscale single image using AI
        
        Args:
            image_path (str): Input image path
            output_path (str): Output image path
            scale_factor (float): Upscaling factor
            prompt (str): Text prompt for AI guidance
            
        Returns:
            bool: Success status
        """
        if not self._available:
            logger.error("AI dependencies not available")
            return False
            
        try:
            if not self._model_loaded:
                if not self.load_model():
                    return False
            
            # Load and preprocess image
            original_image = Image.open(image_path).convert("RGB")
            width, height = original_image.size
            
            # Calculate new dimensions
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Resize image to target size first
            resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Default prompt for upscaling
            if prompt is None:
                prompt = "high quality, detailed, sharp, crisp, clear, 4k, upscaled, enhanced"
            
            # Generate upscaled image
            with torch.autocast(self.device):
                result = self.pipeline(
                    prompt=prompt,
                    image=resized_image,
                    strength=0.3,  # Low strength to preserve original content
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.num_inference_steps,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                )
            
            # Save result
            upscaled_image = result.images[0]
            upscaled_image.save(output_path, "PNG", optimize=True)
            
            logger.debug(f"Upscaled image: {image_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upscale image {image_path}: {e}")
            return False
    
    def upscale_batch(self, image_paths: List[str], output_dir: str,
                     scale_factor: float = 1.5, prompt: str = None,
                     progress_callback: Callable = None) -> List[str]:
        """
        Upscale multiple images in batches
        
        Args:
            image_paths (List[str]): List of input image paths
            output_dir (str): Output directory
            scale_factor (float): Upscaling factor
            prompt (str): Text prompt for AI guidance
            progress_callback (Callable): Progress callback function
            
        Returns:
            List[str]: List of successfully processed output paths
        """
        if not self._model_loaded:
            if not self.load_model():
                return []
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        successful_outputs = []
        total_images = len(image_paths)
        
        for i, image_path in enumerate(image_paths):
            try:
                # Generate output filename
                input_name = Path(image_path).stem
                output_path = output_dir / f"{input_name}_upscaled.png"
                
                # Process image
                if self.upscale_image(str(image_path), str(output_path), 
                                    scale_factor, prompt):
                    successful_outputs.append(str(output_path))
                
                # Call progress callback if provided
                if progress_callback:
                    progress = (i + 1) / total_images * 100
                    progress_callback(progress, f"Processed {i+1}/{total_images} frames")
                
            except Exception as e:
                logger.error(f"Failed to process image {image_path}: {e}")
                continue
        
        return successful_outputs
    
    def estimate_vram_usage(self, image_size: tuple, batch_size: int = 1) -> float:
        """
        Estimate VRAM usage for given parameters
        
        Args:
            image_size (tuple): (width, height) of images
            batch_size (int): Batch size
            
        Returns:
            float: Estimated VRAM usage in GB
        """
        width, height = image_size
        pixels = width * height
        
        # Rough estimation based on model size and image dimensions
        base_model_vram = 2.0  # GB for the model itself
        pixel_vram_factor = 1e-6  # Very rough estimate
        
        estimated_vram = base_model_vram + (pixels * batch_size * pixel_vram_factor)
        return min(estimated_vram, PERFORMANCE["max_memory_gb"])
    
    def cleanup(self):
        """Clean up model from memory"""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            self._model_loaded = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("AI model cleaned up from memory")


class SimpleUpscaler:
    """Simple non-AI upscaling fallback"""
    
    @staticmethod
    def upscale_image_simple(image_path: str, output_path: str, 
                           scale_factor: float = 1.5) -> bool:
        """
        Simple upscaling using traditional algorithms
        
        Args:
            image_path (str): Input image path
            output_path (str): Output image path
            scale_factor (float): Upscaling factor
            
        Returns:
            bool: Success status
        """
        try:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Use LANCZOS for high quality resizing
            upscaled = image.resize((new_width, new_height), Image.LANCZOS)
            upscaled.save(output_path, "PNG", optimize=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Simple upscaling failed: {e}")
            return False


class UnifiedAIUpscaler:
    """
    Unified AI upscaler supporting multiple backends
    Supports Stable Diffusion, Waifu2x, and simple upscaling
    """
    
    def __init__(self, preferred_method: str = "auto"):
        """
        Initialize unified upscaler
        
        Args:
            preferred_method: "waifu2x", "stable_diffusion", "simple", or "auto"
        """
        self.preferred_method = preferred_method
        self.waifu2x_upscaler = None
        self.sd_upscaler = None
        self.simple_upscaler = SimpleUpscaler()
        
        # Initialize available backends
        self._initialize_backends()
        
        # Determine best method if auto
        if preferred_method == "auto":
            self.preferred_method = self._determine_best_method()
        
        logger.info(f"Unified AI upscaler initialized with method: {self.preferred_method}")
    
    def _initialize_backends(self):
        """Initialize available AI backends"""
        # Initialize Waifu2x if available
        if WAIFU2X_AVAILABLE:
            try:
                self.waifu2x_upscaler = Waifu2xUpscaler()
                if self.waifu2x_upscaler.is_available():
                    logger.info("Waifu2x backend available")
                else:
                    self.waifu2x_upscaler = None
                    logger.warning("Waifu2x backend not available")
            except Exception as e:
                logger.error(f"Failed to initialize Waifu2x: {e}")
                self.waifu2x_upscaler = None
        
        # Initialize Stable Diffusion if available
        if TORCH_AVAILABLE and DIFFUSERS_AVAILABLE:
            try:
                self.sd_upscaler = AIUpscaler()
                if self.sd_upscaler._available:
                    logger.info("Stable Diffusion backend available")
                else:
                    self.sd_upscaler = None
                    logger.warning("Stable Diffusion backend not available")
            except Exception as e:
                logger.error(f"Failed to initialize Stable Diffusion: {e}")
                self.sd_upscaler = None
    
    def _determine_best_method(self) -> str:
        """Determine the best available method"""
        if self.waifu2x_upscaler and self.waifu2x_upscaler.is_available():
            return "waifu2x"  # Prefer waifu2x for general purpose
        elif self.sd_upscaler and self.sd_upscaler._available:
            return "stable_diffusion"
        else:
            return "simple"
    
    def get_available_methods(self) -> List[str]:
        """Get list of available upscaling methods"""
        methods = ["simple"]  # Always available
        
        if self.waifu2x_upscaler and self.waifu2x_upscaler.is_available():
            methods.append("waifu2x")
        
        if self.sd_upscaler and self.sd_upscaler._available:
            methods.append("stable_diffusion")
        
        return methods
    
    def upscale_image(self, 
                     image_path: str, 
                     output_path: str,
                     scale_factor: float = 2.0,
                     method: Optional[str] = None,
                     **kwargs) -> bool:
        """
        Upscale image using specified or preferred method
        
        Args:
            image_path: Input image path
            output_path: Output image path
            scale_factor: Scaling factor
            method: Specific method to use (overrides preferred)
            **kwargs: Additional arguments for specific methods
            
        Returns:
            bool: Success status
        """
        # Use specified method or preferred
        method = method or self.preferred_method
        
        try:
            if method == "waifu2x" and self.waifu2x_upscaler:
                return self._upscale_waifu2x(image_path, output_path, scale_factor, **kwargs)
            elif method == "stable_diffusion" and self.sd_upscaler:
                return self._upscale_stable_diffusion(image_path, output_path, scale_factor, **kwargs)
            else:
                # Fallback to simple upscaling
                return self.simple_upscaler.upscale_image_simple(image_path, output_path, scale_factor)
                
        except Exception as e:
            logger.error(f"Upscaling failed with {method}: {e}")
            # Try fallback to simple method
            if method != "simple":
                logger.info("Falling back to simple upscaling")
                return self.simple_upscaler.upscale_image_simple(image_path, output_path, scale_factor)
            return False
    
    def _upscale_waifu2x(self, image_path: str, output_path: str, scale_factor: float, **kwargs) -> bool:
        """Upscale using Waifu2x"""
        try:
            # Update waifu2x settings if needed
            if scale_factor != self.waifu2x_upscaler.scale:
                # Find closest supported scale
                supported_scales = self.waifu2x_upscaler.get_supported_scales()
                closest_scale = min(supported_scales, key=lambda x: abs(x - scale_factor))
                self.waifu2x_upscaler.update_settings(scale=closest_scale)
            
            # Load and process image
            with Image.open(image_path) as image:
                upscaled = self.waifu2x_upscaler.upscale_image(image)
                
                if upscaled is None:
                    return False
                
                # If exact scale factor doesn't match, resize to target
                if abs(scale_factor - self.waifu2x_upscaler.scale) > 0.1:
                    original_size = image.size
                    target_size = (int(original_size[0] * scale_factor), 
                                 int(original_size[1] * scale_factor))
                    upscaled = upscaled.resize(target_size, Image.LANCZOS)
                
                upscaled.save(output_path, "PNG", optimize=True)
                return True
                
        except Exception as e:
            logger.error(f"Waifu2x upscaling failed: {e}")
            return False
    
    def _upscale_stable_diffusion(self, image_path: str, output_path: str, scale_factor: float, **kwargs) -> bool:
        """Upscale using Stable Diffusion"""
        try:
            prompt = kwargs.get('prompt', None)
            return self.sd_upscaler.upscale_image(image_path, output_path, scale_factor, prompt)
        except Exception as e:
            logger.error(f"Stable Diffusion upscaling failed: {e}")
            return False
    
    def upscale_frames(self, 
                      frame_files: List[str], 
                      output_dir: str,
                      scale_factor: float = 2.0,
                      method: Optional[str] = None,
                      progress_callback: Optional[Callable] = None,
                      **kwargs) -> List[str]:
        """
        Upscale multiple frames
        
        Args:
            frame_files: List of input frame paths
            output_dir: Output directory
            scale_factor: Scaling factor
            method: Specific method to use
            progress_callback: Progress callback function
            **kwargs: Additional method-specific arguments
            
        Returns:
            List of output frame paths
        """
        method = method or self.preferred_method
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        upscaled_files = []
        total_frames = len(frame_files)
        
        # Use specialized batch processing for waifu2x
        if method == "waifu2x" and self.waifu2x_upscaler:
            try:
                return self.waifu2x_upscaler.upscale_frames(
                    frame_files, str(output_dir), progress_callback
                )
            except Exception as e:
                logger.error(f"Batch waifu2x processing failed: {e}")
                method = "simple"  # Fallback
        
        # Process frames individually for other methods
        for i, frame_file in enumerate(frame_files):
            try:
                frame_name = Path(frame_file).stem
                output_file = output_dir / f"{frame_name}_upscaled.png"
                
                success = self.upscale_image(
                    frame_file, str(output_file), scale_factor, method, **kwargs
                )
                
                if success:
                    upscaled_files.append(str(output_file))
                
                if progress_callback:
                    progress = (i + 1) / total_frames * 100
                    progress_callback(progress, f"Processed {i+1}/{total_frames} frames with {method}")
                    
            except Exception as e:
                logger.error(f"Failed to process frame {frame_file}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(upscaled_files)}/{total_frames} frames")
        return upscaled_files
    
    def get_method_info(self) -> Dict[str, Any]:
        """Get information about available methods"""
        info = {
            "preferred_method": self.preferred_method,
            "available_methods": self.get_available_methods(),
            "backends": {}
        }
        
        if self.waifu2x_upscaler:
            info["backends"]["waifu2x"] = self.waifu2x_upscaler.get_info()
        
        if self.sd_upscaler:
            info["backends"]["stable_diffusion"] = {
                "available": self.sd_upscaler._available,
                "model_loaded": self.sd_upscaler._model_loaded,
                "device": getattr(self.sd_upscaler, 'device', 'unknown')
            }
        
        return info
    
    def cleanup(self):
        """Clean up all backends"""
        if self.waifu2x_upscaler:
            # Waifu2x doesn't need explicit cleanup
            pass
        
        if self.sd_upscaler:
            self.sd_upscaler.cleanup()
        
        logger.info("Unified AI upscaler cleaned up")