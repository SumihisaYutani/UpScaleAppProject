"""
AI Processing Module
Handles AI-powered image upscaling using Stable Diffusion
"""

import torch
import logging
from PIL import Image
from typing import List, Optional, Callable
from pathlib import Path
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from config.settings import AI_SETTINGS, PERFORMANCE

logger = logging.getLogger(__name__)


class AIUpscaler:
    """AI-powered image upscaling using Stable Diffusion"""
    
    def __init__(self):
        self.model_name = AI_SETTINGS["model_name"]
        self.device = AI_SETTINGS["device"]
        self.batch_size = AI_SETTINGS["batch_size"]
        self.guidance_scale = AI_SETTINGS["guidance_scale"]
        self.num_inference_steps = AI_SETTINGS["num_inference_steps"]
        self.pipeline = None
        self._model_loaded = False
    
    def load_model(self) -> bool:
        """
        Load the Stable Diffusion model
        
        Returns:
            bool: Success status
        """
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