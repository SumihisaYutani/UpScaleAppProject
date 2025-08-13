"""
Enhanced AI Processing Module
Improved AI processing with better error handling and optimization
"""

import torch
import logging
import time
import gc
from PIL import Image
from typing import List, Optional, Callable, Dict, Tuple
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading
from dataclasses import dataclass

try:
    from diffusers import StableDiffusionImg2ImgPipeline, DiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logging.warning("Diffusers not available. AI upscaling will be disabled.")

from config.settings import AI_SETTINGS, PERFORMANCE, PATHS

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for processing operations"""
    total_frames: int = 0
    processed_frames: int = 0
    failed_frames: int = 0
    start_time: float = 0
    end_time: float = 0
    memory_peak_mb: float = 0
    gpu_memory_peak_mb: float = 0
    
    @property
    def processing_time(self) -> float:
        return self.end_time - self.start_time if self.end_time > 0 else time.time() - self.start_time
    
    @property
    def success_rate(self) -> float:
        return (self.processed_frames / self.total_frames * 100) if self.total_frames > 0 else 0


class EnhancedAIUpscaler:
    """Enhanced AI-powered image upscaler with better error handling and optimization"""
    
    def __init__(self, max_retries: int = 3):
        self.model_name = AI_SETTINGS["model_name"]
        self.device = AI_SETTINGS["device"]
        self.batch_size = AI_SETTINGS["batch_size"]
        self.guidance_scale = AI_SETTINGS["guidance_scale"]
        self.num_inference_steps = AI_SETTINGS["num_inference_steps"]
        self.max_retries = max_retries
        
        # State management
        self.pipeline = None
        self._model_loaded = False
        self._last_error = None
        
        # Processing optimization
        self._processing_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._worker_thread = None
        self._stop_processing = threading.Event()
        
        # Statistics
        self.stats = ProcessingStats()
        
        # Check availability
        self.available = DIFFUSERS_AVAILABLE and self._check_dependencies()
        
        if not self.available:
            logger.warning("Enhanced AI upscaler not available. Falling back to simple upscaling.")
    
    def _check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        try:
            import torch
            import PIL
            return True
        except ImportError as e:
            logger.warning(f"Missing dependency: {e}")
            return False
    
    def load_model(self, force_reload: bool = False) -> bool:
        """
        Load the AI model with enhanced error handling
        
        Args:
            force_reload (bool): Force model reload even if already loaded
            
        Returns:
            bool: Success status
        """
        if not self.available:
            logger.error("AI upscaler dependencies not available")
            return False
        
        if self._model_loaded and not force_reload:
            return True
        
        try:
            logger.info(f"Loading AI model: {self.model_name}")
            self.stats.start_time = time.time()
            
            # Memory management
            if self.pipeline:
                del self.pipeline
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Load with error handling and retries
            for attempt in range(self.max_retries):
                try:
                    # Determine appropriate dtype
                    dtype = torch.float16 if self.device == "cuda" and torch.cuda.is_available() else torch.float32
                    
                    # Load pipeline with safety features disabled for performance
                    self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                        self.model_name,
                        torch_dtype=dtype,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_auth_token=False
                    )
                    
                    # Move to device
                    self.pipeline = self.pipeline.to(self.device)
                    
                    # Apply memory optimizations
                    self._apply_optimizations()
                    
                    # Test the model with a small operation
                    if self._test_model():
                        self._model_loaded = True
                        logger.info(f"AI model loaded successfully in {time.time() - self.stats.start_time:.2f}s")
                        return True
                    else:
                        raise RuntimeError("Model test failed")
                        
                except Exception as e:
                    logger.warning(f"Model loading attempt {attempt + 1} failed: {e}")
                    if attempt == self.max_retries - 1:
                        self._last_error = str(e)
                        logger.error(f"Failed to load model after {self.max_retries} attempts: {e}")
                        return False
                    
                    # Wait before retry
                    time.sleep(1 * (attempt + 1))
            
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Unexpected error loading model: {e}")
            return False
        
        return False
    
    def _apply_optimizations(self):
        """Apply memory and performance optimizations"""
        if not self.pipeline:
            return
        
        try:
            # Enable attention slicing for memory efficiency
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing(1)
            
            # Enable VAE slicing
            if hasattr(self.pipeline, "enable_vae_slicing"):
                self.pipeline.enable_vae_slicing()
            
            # Enable CPU offloading if on CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                if hasattr(self.pipeline, "enable_sequential_cpu_offload"):
                    self.pipeline.enable_sequential_cpu_offload()
                elif hasattr(self.pipeline, "enable_model_cpu_offload"):
                    self.pipeline.enable_model_cpu_offload()
            
            # Use memory efficient attention if available
            if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                except:
                    pass  # xformers not available
                    
        except Exception as e:
            logger.warning(f"Some optimizations failed to apply: {e}")
    
    def _test_model(self) -> bool:
        """Test the loaded model with a small operation"""
        try:
            # Create a small test image
            test_image = Image.new("RGB", (64, 64), color=(128, 128, 128))
            
            # Test generation with minimal parameters
            with torch.autocast(self.device):
                result = self.pipeline(
                    prompt="test",
                    image=test_image,
                    strength=0.1,
                    guidance_scale=1.0,
                    num_inference_steps=1,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                )
            
            return result.images[0] is not None
            
        except Exception as e:
            logger.warning(f"Model test failed: {e}")
            return False
    
    def upscale_image_enhanced(self, image_path: str, output_path: str,
                             scale_factor: float = 1.5, prompt: str = None,
                             quality_settings: Dict = None) -> Dict[str, any]:
        """
        Enhanced image upscaling with detailed result information
        
        Args:
            image_path (str): Input image path
            output_path (str): Output image path
            scale_factor (float): Upscaling factor
            prompt (str): Text prompt for AI guidance
            quality_settings (Dict): Custom quality settings
            
        Returns:
            Dict containing detailed processing results
        """
        result = {
            "success": False,
            "output_path": None,
            "error": None,
            "processing_time": 0,
            "original_size": None,
            "output_size": None,
            "memory_used_mb": 0
        }
        
        start_time = time.time()
        
        try:
            if not self._model_loaded:
                if not self.load_model():
                    result["error"] = f"Failed to load AI model: {self._last_error}"
                    return result
            
            # Load and validate image
            try:
                original_image = Image.open(image_path).convert("RGB")
                result["original_size"] = original_image.size
            except Exception as e:
                result["error"] = f"Failed to load image: {e}"
                return result
            
            # Calculate dimensions
            width, height = original_image.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Apply quality settings
            settings = self._get_quality_settings(quality_settings)
            
            # Resize image first
            resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Prepare prompt
            if prompt is None:
                prompt = "high quality, detailed, sharp, crisp, clear, upscaled, enhanced, masterpiece"
            
            # Monitor memory usage
            memory_before = self._get_memory_usage()
            
            # Generate enhanced image
            with torch.autocast(self.device):
                ai_result = self.pipeline(
                    prompt=prompt,
                    image=resized_image,
                    strength=settings["strength"],
                    guidance_scale=settings["guidance_scale"],
                    num_inference_steps=settings["num_inference_steps"],
                    generator=torch.Generator(device=self.device).manual_seed(settings["seed"])
                )
            
            memory_after = self._get_memory_usage()
            result["memory_used_mb"] = memory_after - memory_before
            
            # Save result
            upscaled_image = ai_result.images[0]
            upscaled_image.save(output_path, "PNG", optimize=True, quality=95)
            
            result["success"] = True
            result["output_path"] = output_path
            result["output_size"] = (new_width, new_height)
            
        except Exception as e:
            result["error"] = f"Image upscaling failed: {e}"
            logger.error(f"Enhanced upscaling error: {e}")
        
        finally:
            result["processing_time"] = time.time() - start_time
        
        return result
    
    def _get_quality_settings(self, custom_settings: Dict = None) -> Dict:
        """Get quality settings with defaults and customizations"""
        default_settings = {
            "strength": 0.3,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": 42
        }
        
        if custom_settings:
            default_settings.update(custom_settings)
        
        return default_settings
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
            else:
                import psutil
                return psutil.Process().memory_info().rss / (1024 * 1024)
        except:
            return 0
    
    def upscale_batch_enhanced(self, image_paths: List[str], output_dir: str,
                              scale_factor: float = 1.5, prompt: str = None,
                              progress_callback: Callable = None,
                              parallel_workers: int = 1) -> Dict[str, any]:
        """
        Enhanced batch processing with parallel support
        
        Args:
            image_paths (List[str]): List of input image paths
            output_dir (str): Output directory
            scale_factor (float): Upscaling factor
            prompt (str): Text prompt for AI guidance
            progress_callback (Callable): Progress callback function
            parallel_workers (int): Number of parallel workers (1 = sequential)
            
        Returns:
            Dict containing batch processing results
        """
        if not self._model_loaded:
            if not self.load_model():
                return {"success": False, "error": "Failed to load AI model"}
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = ProcessingStats()
        self.stats.total_frames = len(image_paths)
        self.stats.start_time = time.time()
        
        successful_outputs = []
        failed_items = []
        
        try:
            if parallel_workers > 1:
                # Parallel processing
                successful_outputs, failed_items = self._process_batch_parallel(
                    image_paths, output_dir, scale_factor, prompt, 
                    progress_callback, parallel_workers
                )
            else:
                # Sequential processing
                successful_outputs, failed_items = self._process_batch_sequential(
                    image_paths, output_dir, scale_factor, prompt, progress_callback
                )
        
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "stats": self.stats
            }
        
        finally:
            self.stats.end_time = time.time()
        
        return {
            "success": len(successful_outputs) > 0,
            "successful_outputs": successful_outputs,
            "failed_items": failed_items,
            "stats": self.stats
        }
    
    def _process_batch_sequential(self, image_paths, output_dir, scale_factor, 
                                prompt, progress_callback) -> Tuple[List[str], List[Dict]]:
        """Sequential batch processing"""
        successful_outputs = []
        failed_items = []
        
        for i, image_path in enumerate(image_paths):
            try:
                input_name = Path(image_path).stem
                output_path = output_dir / f"{input_name}_upscaled.png"
                
                result = self.upscale_image_enhanced(
                    str(image_path), str(output_path), scale_factor, prompt
                )
                
                if result["success"]:
                    successful_outputs.append(str(output_path))
                    self.stats.processed_frames += 1
                else:
                    failed_items.append({"path": image_path, "error": result["error"]})
                    self.stats.failed_frames += 1
                
                # Update progress
                if progress_callback:
                    progress = (i + 1) / len(image_paths) * 100
                    progress_callback(progress, f"Processed {i+1}/{len(image_paths)} frames")
                
                # Memory management
                if (i + 1) % 10 == 0:  # Every 10 images
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                failed_items.append({"path": image_path, "error": str(e)})
                self.stats.failed_frames += 1
                logger.error(f"Failed to process {image_path}: {e}")
        
        return successful_outputs, failed_items
    
    def _process_batch_parallel(self, image_paths, output_dir, scale_factor,
                              prompt, progress_callback, max_workers) -> Tuple[List[str], List[Dict]]:
        """Parallel batch processing (experimental)"""
        # Note: Parallel processing with AI models can be tricky due to GPU memory
        # This is a simplified implementation
        successful_outputs = []
        failed_items = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_path = {}
            for image_path in image_paths:
                input_name = Path(image_path).stem
                output_path = output_dir / f"{input_name}_upscaled.png"
                
                future = executor.submit(
                    self.upscale_image_enhanced,
                    str(image_path), str(output_path), scale_factor, prompt
                )
                future_to_path[future] = image_path
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_path):
                image_path = future_to_path[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result["success"]:
                        successful_outputs.append(result["output_path"])
                        self.stats.processed_frames += 1
                    else:
                        failed_items.append({"path": image_path, "error": result["error"]})
                        self.stats.failed_frames += 1
                
                except Exception as e:
                    failed_items.append({"path": image_path, "error": str(e)})
                    self.stats.failed_frames += 1
                
                # Update progress
                if progress_callback:
                    progress = completed / len(image_paths) * 100
                    progress_callback(progress, f"Completed {completed}/{len(image_paths)} frames")
        
        return successful_outputs, failed_items
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model"""
        info = {
            "available": self.available,
            "model_loaded": self._model_loaded,
            "model_name": self.model_name,
            "device": self.device,
            "last_error": self._last_error
        }
        
        if self._model_loaded and self.pipeline:
            info["model_type"] = type(self.pipeline).__name__
            
        if torch.cuda.is_available():
            info["cuda_memory_allocated"] = torch.cuda.memory_allocated() / (1024**3)
            info["cuda_memory_reserved"] = torch.cuda.memory_reserved() / (1024**3)
        
        return info
    
    def cleanup(self):
        """Enhanced cleanup with better memory management"""
        try:
            if self.pipeline:
                del self.pipeline
                self.pipeline = None
            
            self._model_loaded = False
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("Enhanced AI model cleaned up from memory")
            
        except Exception as e:
            logger.warning(f"Cleanup had some issues: {e}")


# Factory function to choose the appropriate upscaler
def create_upscaler(use_enhanced: bool = True, **kwargs) -> object:
    """
    Factory function to create appropriate upscaler
    
    Args:
        use_enhanced (bool): Whether to use enhanced AI upscaler
        **kwargs: Additional arguments for upscaler
        
    Returns:
        Appropriate upscaler instance
    """
    if use_enhanced and DIFFUSERS_AVAILABLE:
        return EnhancedAIUpscaler(**kwargs)
    else:
        # Fallback to original AI upscaler or simple upscaler
        from .ai_processor import AIUpscaler, SimpleUpscaler
        if DIFFUSERS_AVAILABLE:
            return AIUpscaler()
        else:
            return SimpleUpscaler()