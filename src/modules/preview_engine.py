"""
Real-time Preview Engine
High-performance preview system for instant feedback during parameter adjustment
"""

import cv2
import numpy as np
import time
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, Future

from modules.ai_model_manager import model_manager
from modules.model_selector import model_selector, ProcessingConstraints, ProcessingPriority

logger = logging.getLogger(__name__)


class PreviewQuality(Enum):
    """Preview quality levels"""
    LOW = "low"           # Fast, low quality
    MEDIUM = "medium"     # Balanced
    HIGH = "high"         # Slow, high quality
    ADAPTIVE = "adaptive" # Adapt based on image size


class PreviewMode(Enum):
    """Preview display modes"""
    SPLIT_VERTICAL = "split_vertical"     # Side by side
    SPLIT_HORIZONTAL = "split_horizontal" # Top/bottom
    OVERLAY = "overlay"                   # Overlay with transparency
    DIFFERENCE = "difference"             # Difference map
    ORIGINAL_ONLY = "original_only"       # Show only original
    PROCESSED_ONLY = "processed_only"     # Show only processed


@dataclass
class PreviewRegion:
    """Region of Interest for preview"""
    x: int
    y: int
    width: int
    height: int
    
    def to_slice(self) -> Tuple[slice, slice]:
        """Convert to numpy slice"""
        return slice(self.y, self.y + self.height), slice(self.x, self.x + self.width)
    
    def scale(self, factor: float) -> 'PreviewRegion':
        """Scale region by factor"""
        return PreviewRegion(
            x=int(self.x * factor),
            y=int(self.y * factor),
            width=int(self.width * factor),
            height=int(self.height * factor)
        )


@dataclass
class PreviewSettings:
    """Preview generation settings"""
    quality: PreviewQuality = PreviewQuality.MEDIUM
    mode: PreviewMode = PreviewMode.SPLIT_VERTICAL
    target_size: Tuple[int, int] = (512, 512)  # Maximum preview size
    roi: Optional[PreviewRegion] = None        # Region of interest
    scale_factor: float = 1.5
    model_id: Optional[str] = None
    processing_params: Dict[str, Any] = None
    cache_enabled: bool = True
    max_cache_size: int = 50  # Maximum cached previews
    
    def __post_init__(self):
        if self.processing_params is None:
            self.processing_params = {}
    
    def get_cache_key(self, image_path: str) -> str:
        """Generate cache key for settings"""
        key_parts = [
            image_path,
            self.quality.value,
            str(self.target_size),
            str(self.roi) if self.roi else "None",
            str(self.scale_factor),
            self.model_id or "None",
            str(sorted(self.processing_params.items()))
        ]
        return "|".join(key_parts)


@dataclass
class PreviewResult:
    """Result of preview generation"""
    original_image: np.ndarray
    processed_image: Optional[np.ndarray]
    composite_image: np.ndarray  # Final display image
    processing_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PreviewCache:
    """LRU cache for preview results"""
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.cache: Dict[str, PreviewResult] = {}
        self.access_order: List[str] = []
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[PreviewResult]:
        """Get cached preview result"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, result: PreviewResult):
        """Cache preview result"""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
            
            self.cache[key] = result
            self.access_order.append(key)
    
    def clear(self):
        """Clear all cached results"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": 0.0  # Would track in real implementation
            }


class PreviewEngine:
    """High-performance real-time preview engine"""
    
    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Preview cache
        self.cache = PreviewCache()
        
        # Processing queue for async previews
        self.preview_queue = queue.Queue(maxsize=10)
        self.active_futures: Dict[str, Future] = {}
        
        # Callbacks
        self.preview_callback: Optional[Callable[[str, PreviewResult], None]] = None
        self.error_callback: Optional[Callable[[str, str], None]] = None
        
        # Thread control
        self.shutdown_event = threading.Event()
        
        # Performance tracking
        self.stats = {
            "previews_generated": 0,
            "cache_hits": 0,
            "average_processing_time": 0.0,
            "errors": 0
        }
        
        logger.info("Preview engine initialized")
    
    def generate_preview(self, image_path: str, settings: PreviewSettings,
                        async_mode: bool = True) -> Optional[PreviewResult]:
        """Generate preview with given settings"""
        
        if async_mode:
            # Queue for async processing
            future = self.executor.submit(self._generate_preview_sync, image_path, settings)
            self.active_futures[image_path] = future
            return None
        else:
            # Synchronous processing
            return self._generate_preview_sync(image_path, settings)
    
    def _generate_preview_sync(self, image_path: str, settings: PreviewSettings) -> PreviewResult:
        """Synchronously generate preview"""
        
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = settings.get_cache_key(image_path)
            if settings.cache_enabled:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.stats["cache_hits"] += 1
                    return cached_result
            
            # Load and prepare original image
            original_image = self._load_and_prepare_image(image_path, settings)
            if original_image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Extract region of interest if specified
            if settings.roi:
                roi_slice = settings.roi.to_slice()
                roi_image = original_image[roi_slice]
            else:
                roi_image = original_image
            
            # Generate processed version
            processed_image = None
            processing_error = None
            
            if settings.quality != PreviewQuality.LOW or settings.model_id:
                try:
                    processed_image = self._process_image_region(
                        roi_image, settings, image_path
                    )
                except Exception as e:
                    processing_error = str(e)
                    logger.error(f"Preview processing error: {e}")
            
            # Create composite image based on mode
            composite_image = self._create_composite(
                roi_image, processed_image, settings.mode
            )
            
            # Create result
            processing_time = time.time() - start_time
            result = PreviewResult(
                original_image=roi_image,
                processed_image=processed_image,
                composite_image=composite_image,
                processing_time=processing_time,
                error=processing_error,
                metadata={
                    "image_path": image_path,
                    "settings": settings.__dict__,
                    "roi": settings.roi.__dict__ if settings.roi else None
                }
            )
            
            # Cache result
            if settings.cache_enabled:
                self.cache.put(cache_key, result)
            
            # Update stats
            self.stats["previews_generated"] += 1
            self.stats["average_processing_time"] = (
                (self.stats["average_processing_time"] * (self.stats["previews_generated"] - 1) + 
                 processing_time) / self.stats["previews_generated"]
            )
            
            # Callback if set
            if self.preview_callback:
                try:
                    self.preview_callback(image_path, result)
                except Exception as e:
                    logger.error(f"Preview callback error: {e}")
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            error_msg = f"Preview generation failed: {str(e)}"
            logger.error(error_msg)
            
            if self.error_callback:
                try:
                    self.error_callback(image_path, error_msg)
                except:
                    pass
            
            # Return error result
            return PreviewResult(
                original_image=np.zeros((100, 100, 3), dtype=np.uint8),
                processed_image=None,
                composite_image=np.zeros((100, 100, 3), dtype=np.uint8),
                processing_time=time.time() - start_time,
                error=error_msg
            )
    
    def _load_and_prepare_image(self, image_path: str, settings: PreviewSettings) -> Optional[np.ndarray]:
        """Load and prepare image for preview"""
        
        try:
            # Load image
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to target size if too large
            height, width = image.shape[:2]
            target_width, target_height = settings.target_size
            
            if width > target_width or height > target_height:
                # Calculate scaling factor to fit within target size
                scale_factor = min(target_width / width, target_height / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                image = cv2.resize(image, (new_width, new_height), 
                                 interpolation=cv2.INTER_AREA)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def _process_image_region(self, image: np.ndarray, settings: PreviewSettings,
                            original_path: str) -> np.ndarray:
        """Process image region with AI model"""
        
        # Determine model to use
        model_id = settings.model_id
        if not model_id:
            # Auto-select model
            constraints = ProcessingConstraints(
                priority=ProcessingPriority.SPEED,
                target_scale_factor=settings.scale_factor,
                max_processing_time_seconds=5.0  # Fast preview
            )
            model_id = model_selector.select_optimal_model(original_path, constraints)
        
        if not model_id:
            raise ValueError("No suitable model available")
        
        # Ensure model is loaded
        if not model_manager.is_model_loaded(model_id):
            success = model_manager.load_model(model_id, async_load=False)
            if not success:
                raise ValueError(f"Failed to load model {model_id}")
        
        # Save image to temp file for processing
        temp_dir = Path("temp_preview")
        temp_dir.mkdir(exist_ok=True)
        temp_input = temp_dir / f"preview_input_{int(time.time()*1000)}.png"
        temp_output = temp_dir / f"preview_output_{int(time.time()*1000)}.png"
        
        try:
            # Save input
            cv2.imwrite(str(temp_input), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Process with model
            processing_params = settings.processing_params.copy()
            processing_params.update({
                "num_inference_steps": 10 if settings.quality == PreviewQuality.LOW else 20,
                "guidance_scale": 7.5
            })
            
            result = model_manager.process_image(
                model_id, str(temp_input), str(temp_output),
                settings.scale_factor, **processing_params
            )
            
            if not result.get("success", False):
                raise ValueError(result.get("error", "Processing failed"))
            
            # Load result
            processed_image = cv2.imread(str(temp_output), cv2.IMREAD_COLOR)
            if processed_image is None:
                raise ValueError("Failed to load processed image")
            
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            # Scale to match preview size if needed
            target_height, target_width = image.shape[:2]
            processed_height, processed_width = processed_image.shape[:2]
            
            if (processed_width != target_width * settings.scale_factor or 
                processed_height != target_height * settings.scale_factor):
                
                new_width = int(target_width * settings.scale_factor)
                new_height = int(target_height * settings.scale_factor)
                processed_image = cv2.resize(processed_image, (new_width, new_height))
            
            return processed_image
            
        finally:
            # Cleanup temp files
            try:
                temp_input.unlink(missing_ok=True)
                temp_output.unlink(missing_ok=True)
            except:
                pass
    
    def _create_composite(self, original: np.ndarray, processed: Optional[np.ndarray],
                         mode: PreviewMode) -> np.ndarray:
        """Create composite preview image based on display mode"""
        
        if processed is None:
            # No processed version, return original
            return original.copy()
        
        if mode == PreviewMode.ORIGINAL_ONLY:
            return original.copy()
        elif mode == PreviewMode.PROCESSED_ONLY:
            return processed.copy()
        
        # Ensure images have same height for side-by-side comparison
        orig_h, orig_w = original.shape[:2]
        proc_h, proc_w = processed.shape[:2]
        
        # Resize processed to match original height for comparison
        if proc_h != orig_h:
            aspect_ratio = proc_w / proc_h
            new_width = int(orig_h * aspect_ratio)
            processed_resized = cv2.resize(processed, (new_width, orig_h))
        else:
            processed_resized = processed.copy()
        
        if mode == PreviewMode.SPLIT_VERTICAL:
            # Side by side
            composite = np.hstack([original, processed_resized])
            
        elif mode == PreviewMode.SPLIT_HORIZONTAL:
            # Resize to same width for top/bottom
            if orig_w != processed_resized.shape[1]:
                processed_resized = cv2.resize(processed_resized, (orig_w, processed_resized.shape[0]))
            composite = np.vstack([original, processed_resized])
            
        elif mode == PreviewMode.OVERLAY:
            # Overlay with 50% transparency
            # Resize processed to match original
            processed_overlay = cv2.resize(processed_resized, (orig_w, orig_h))
            composite = cv2.addWeighted(original, 0.5, processed_overlay, 0.5, 0)
            
        elif mode == PreviewMode.DIFFERENCE:
            # Difference map
            processed_diff = cv2.resize(processed_resized, (orig_w, orig_h))
            difference = cv2.absdiff(original, processed_diff)
            # Enhance difference for visibility
            composite = cv2.multiply(difference, 3)
            composite = np.clip(composite, 0, 255).astype(np.uint8)
            
        else:
            # Default to split vertical
            composite = np.hstack([original, processed_resized])
        
        return composite
    
    def cancel_preview(self, image_path: str):
        """Cancel active preview generation"""
        if image_path in self.active_futures:
            future = self.active_futures[image_path]
            future.cancel()
            del self.active_futures[image_path]
    
    def wait_for_preview(self, image_path: str, timeout: float = 10.0) -> Optional[PreviewResult]:
        """Wait for async preview to complete"""
        if image_path not in self.active_futures:
            return None
        
        try:
            future = self.active_futures[image_path]
            result = future.result(timeout=timeout)
            del self.active_futures[image_path]
            return result
        except Exception as e:
            logger.error(f"Error waiting for preview {image_path}: {e}")
            return None
    
    def clear_cache(self):
        """Clear preview cache"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preview engine statistics"""
        cache_stats = self.cache.get_stats()
        return {
            **self.stats,
            **cache_stats,
            "active_previews": len(self.active_futures)
        }
    
    def set_preview_callback(self, callback: Callable[[str, PreviewResult], None]):
        """Set callback for preview completion"""
        self.preview_callback = callback
    
    def set_error_callback(self, callback: Callable[[str, str], None]):
        """Set callback for preview errors"""
        self.error_callback = callback
    
    def shutdown(self):
        """Shutdown preview engine"""
        logger.info("Shutting down preview engine...")
        
        self.shutdown_event.set()
        
        # Cancel all active futures
        for future in self.active_futures.values():
            future.cancel()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Clear cache
        self.cache.clear()
        
        logger.info("Preview engine shutdown complete")


# Global preview engine instance
preview_engine = PreviewEngine()