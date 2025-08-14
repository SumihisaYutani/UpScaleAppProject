"""
Enhanced UpScale Application with improved error handling and performance monitoring
Main application class with comprehensive error handling, performance tracking, and robustness
"""

import os
import logging
import shutil
import traceback
from pathlib import Path
from typing import Optional, Callable, Dict, List, Any
from tqdm import tqdm
import time
import json
from contextlib import contextmanager

from modules.video_processor import VideoProcessor, VideoFrameExtractor
from modules.video_builder import VideoBuilder
from modules.enhanced_ai_processor import EnhancedAIUpscaler, create_upscaler
from modules.ai_processor import SimpleUpscaler
from modules.performance_monitor import PerformanceMonitor, MemoryProfiler
from config.settings import PATHS, VIDEO_SETTINGS, PERFORMANCE, AI_SETTINGS

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    """Custom exception for processing errors"""
    def __init__(self, message: str, error_code: str = "PROCESSING_ERROR", details: Dict = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class EnhancedUpScaleApp:
    """Enhanced main application class for video upscaling with comprehensive error handling"""
    
    def __init__(self, use_ai: bool = True, use_enhanced_ai: bool = True, 
                 temp_cleanup: bool = True, enable_monitoring: bool = True):
        self.use_ai = use_ai
        self.use_enhanced_ai = use_enhanced_ai
        self.temp_cleanup = temp_cleanup
        self.enable_monitoring = enable_monitoring
        
        # Initialize performance monitoring
        self.monitor = PerformanceMonitor() if enable_monitoring else None
        self.memory_profiler = MemoryProfiler()
        
        # Initialize modules with error handling
        try:
            self._initialize_modules()
        except Exception as e:
            logger.error(f"Failed to initialize modules: {e}")
            raise ProcessingError("Module initialization failed", "INIT_ERROR", {"error": str(e)})
        
        # Progress tracking
        self.current_progress = 0
        self.status_message = ""
        self.processing_stats = {}
        
        # Error tracking
        self.error_history: List[Dict] = []
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        
        # Session info
        self.session_id = f"session_{int(time.time())}"
        self.session_start = time.time()
        
        logger.info(f"Enhanced UpScale App initialized (Session: {self.session_id})")
    
    def _initialize_modules(self):
        """Initialize all processing modules with error handling"""
        try:
            # Core modules
            self.video_processor = VideoProcessor()
            self.frame_extractor = VideoFrameExtractor(str(PATHS["temp_dir"]))
            self.video_builder = VideoBuilder(str(PATHS["output_dir"]))
            
            # AI modules - prioritize available backends
            if self.use_ai:
                try:
                    # First try waifu2x since we know it's available
                    from modules.waifu2x_processor import test_waifu2x_availability
                    waifu2x_backends = test_waifu2x_availability()
                    
                    if any(waifu2x_backends.values()):
                        logger.info("Using Waifu2x backend for AI processing")
                        from modules.waifu2x_processor import Waifu2xUpscaler
                        self.ai_upscaler = Waifu2xUpscaler(backend='ncnn', scale=2)
                    elif self.use_enhanced_ai:
                        logger.info("Trying enhanced AI upscaler")
                        self.ai_upscaler = create_upscaler(use_enhanced=True)
                    else:
                        logger.info("Trying standard AI upscaler")
                        from modules.ai_processor import AIUpscaler
                        self.ai_upscaler = AIUpscaler()
                        
                except Exception as e:
                    logger.warning(f"Failed to initialize AI upscaler: {e}")
                    self.use_ai = False
                    self.ai_upscaler = None
            else:
                self.ai_upscaler = None
            
            # Start monitoring if enabled
            if self.monitor:
                self.monitor.start_monitoring()
                self.memory_profiler.set_baseline()
            
        except ImportError as e:
            logger.warning(f"Some AI dependencies missing: {e}")
            self.use_ai = False
            self.ai_upscaler = None
        except Exception as e:
            logger.error(f"Module initialization error: {e}")
            raise
    
    @contextmanager
    def _error_handler(self, operation_name: str):
        """Context manager for comprehensive error handling"""
        start_time = time.time()
        try:
            yield
        except ProcessingError:
            raise  # Re-raise custom processing errors
        except Exception as e:
            error_info = {
                "operation": operation_name,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": time.time(),
                "processing_time": time.time() - start_time,
                "traceback": traceback.format_exc()
            }
            
            self.error_history.append(error_info)
            logger.error(f"Error in {operation_name}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Attempt recovery for certain error types
            if self._should_attempt_recovery(e):
                self._attempt_recovery(operation_name, e)
            
            raise ProcessingError(
                f"Operation '{operation_name}' failed: {str(e)}", 
                "OPERATION_ERROR", 
                error_info
            )
    
    def _should_attempt_recovery(self, error: Exception) -> bool:
        """Determine if recovery should be attempted for this error"""
        if self.recovery_attempts >= self.max_recovery_attempts:
            return False
        
        # Attempt recovery for memory errors, CUDA errors, etc.
        recoverable_errors = [
            "OutOfMemoryError", "CUDA_ERROR", "RuntimeError"
        ]
        
        return any(err_type in str(type(error)) for err_type in recoverable_errors)
    
    def _attempt_recovery(self, operation_name: str, error: Exception):
        """Attempt to recover from errors"""
        self.recovery_attempts += 1
        logger.info(f"Attempting recovery {self.recovery_attempts}/{self.max_recovery_attempts} for {operation_name}")
        
        try:
            # Clear GPU memory if CUDA error
            if "CUDA" in str(error) or "GPU" in str(error):
                self._clear_gpu_memory()
            
            # Garbage collection for memory errors
            if "memory" in str(error).lower():
                self._emergency_cleanup()
            
            # Restart AI model if model-related error
            if self.ai_upscaler and hasattr(self.ai_upscaler, 'cleanup'):
                self.ai_upscaler.cleanup()
                time.sleep(2)  # Wait before restart
                
        except Exception as recovery_error:
            logger.warning(f"Recovery attempt failed: {recovery_error}")
    
    def _clear_gpu_memory(self):
        """Clear GPU memory"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU memory cleared")
        except ImportError:
            pass
    
    def _emergency_cleanup(self):
        """Emergency cleanup to free memory"""
        import gc
        gc.collect()
        
        # Clear temporary files
        temp_dir = PATHS["temp_dir"]
        if temp_dir.exists():
            try:
                for temp_file in temp_dir.rglob("*"):
                    if temp_file.is_file() and temp_file.stat().st_mtime < time.time() - 3600:  # 1 hour old
                        temp_file.unlink()
            except Exception as e:
                logger.warning(f"Temp cleanup failed: {e}")
        
        logger.info("Emergency cleanup performed")
    
    def process_video_enhanced(self, input_path: str, output_path: str = None,
                              scale_factor: float = None, 
                              progress_callback: Callable = None,
                              quality_settings: Dict = None) -> Dict[str, Any]:
        """
        Enhanced video processing with comprehensive error handling and monitoring
        
        Args:
            input_path (str): Path to input MP4 file
            output_path (str): Path for output file (optional)
            scale_factor (float): Upscaling factor (default from settings)
            progress_callback (Callable): Progress callback function
            quality_settings (Dict): Custom quality settings for AI processing
            
        Returns:
            Dict containing processing results with enhanced information
        """
        # Start performance tracking
        if self.monitor:
            performance_context = self.monitor.track_processing(f"video_upscale_{self.session_id}")
            performance_context.__enter__()
        else:
            performance_context = None
        
        result = {
            "success": False,
            "output_path": None,
            "error": None,
            "error_code": None,
            "stats": {},
            "performance": {},
            "session_id": self.session_id,
            "processing_stages": []
        }
        
        self.recovery_attempts = 0  # Reset recovery attempts
        
        try:
            with self._error_handler("video_processing"):
                # Stage 1: Validation
                with self._error_handler("validation"):
                    self._update_progress(5, "Validating input file...", progress_callback)
                    validation = self.video_processor.validate_video_file(input_path)
                    
                    if not validation["valid"]:
                        raise ProcessingError(validation["error"], "VALIDATION_ERROR")
                    
                    video_info = validation["info"]
                    result["stats"]["video_info"] = video_info
                    result["processing_stages"].append({"stage": "validation", "status": "completed", "time": time.time()})
                    
                    self.memory_profiler.checkpoint("after_validation")
                    logger.info(f"Processing video: {video_info['filename']} ({video_info['width']}x{video_info['height']})")
                
                # Stage 2: Setup
                with self._error_handler("setup"):
                    if output_path is None:
                        output_filename = f"{Path(input_path).stem}_upscaled_{self.session_id}.mp4"
                        output_path = str(PATHS["output_dir"] / output_filename)
                    
                    if scale_factor is None:
                        scale_factor = VIDEO_SETTINGS["default_upscale_factor"]
                    
                    original_resolution = (video_info["width"], video_info["height"])
                    new_resolution = self.video_processor.get_upscaled_resolution(
                        video_info["width"], video_info["height"], scale_factor
                    )
                    
                    result["stats"]["scale_factor"] = scale_factor
                    result["stats"]["original_resolution"] = original_resolution
                    result["stats"]["target_resolution"] = new_resolution
                    result["processing_stages"].append({"stage": "setup", "status": "completed", "time": time.time()})
                
                # Stage 3: Time estimation
                with self._error_handler("estimation"):
                    time_estimates = self.video_processor.estimate_processing_time(video_info)
                    result["stats"]["time_estimates"] = time_estimates
                    
                    estimated_memory = self._estimate_memory_requirements(video_info, scale_factor)
                    result["stats"]["memory_estimates"] = estimated_memory
                    
                    # Check if we have enough resources
                    if not self._check_resource_availability(estimated_memory):
                        logger.warning("Insufficient resources detected, proceeding with caution")
                        result["stats"]["resource_warning"] = True
                
                # Stage 4: Frame extraction
                with self._error_handler("frame_extraction"):
                    self._update_progress(15, "Extracting frames...", progress_callback)
                    frame_files = self.frame_extractor.extract_frames(input_path)
                    
                    if not frame_files:
                        raise ProcessingError("Failed to extract frames from video", "EXTRACTION_ERROR")
                    
                    result["stats"]["frame_count"] = len(frame_files)
                    result["processing_stages"].append({"stage": "extraction", "status": "completed", "time": time.time()})
                    self.memory_profiler.checkpoint("after_extraction")
                    
                    if self.monitor:
                        self.monitor.update_processing_stats(frames_processed=0, bytes_processed=video_info.get("size", 0))
                
                # Stage 5: AI processing
                processed_frames = []
                with self._error_handler("ai_processing"):
                    if self.use_ai and self.ai_upscaler:
                        self._update_progress(25, "Loading AI model...", progress_callback)
                        
                        # Enhanced AI processing
                        model_loaded = True  # Default to True
                        if hasattr(self.ai_upscaler, 'load_model'):
                            model_loaded = self.ai_upscaler.load_model()
                            if not model_loaded:
                                logger.warning("Failed to load AI model, falling back to simple upscaling")
                                self.use_ai = False
                        elif hasattr(self.ai_upscaler, '_initialize_ncnn'):
                            # For Waifu2x, initialization happens in constructor
                            logger.info("Waifu2x AI upscaler ready for processing")
                        
                        if self.use_ai:
                            self._update_progress(30, "Processing frames with AI...", progress_callback)
                            processed_frames = self._process_frames_ai_enhanced(
                                frame_files, scale_factor, progress_callback, quality_settings
                            )
                        else:
                            processed_frames = self._process_frames_simple(frame_files, scale_factor, progress_callback)
                    else:
                        processed_frames = self._process_frames_simple(frame_files, scale_factor, progress_callback)
                    
                    if not processed_frames:
                        raise ProcessingError("No frames were successfully processed", "PROCESSING_ERROR")
                    
                    result["stats"]["processed_frame_count"] = len(processed_frames)
                    result["processing_stages"].append({"stage": "ai_processing", "status": "completed", "time": time.time()})
                    self.memory_profiler.checkpoint("after_ai_processing")
                    
                    if self.monitor:
                        self.monitor.update_processing_stats(frames_processed=len(processed_frames))
                
                # Stage 6: Video reconstruction
                with self._error_handler("video_reconstruction"):
                    self._update_progress(85, "Combining frames into video...", progress_callback)
                    
                    fps = video_info.get("frame_rate", 30.0)
                    success = self.video_builder.combine_frames_to_video(
                        processed_frames, output_path, input_path, fps, preserve_audio=True
                    )
                    
                    if not success:
                        raise ProcessingError("Failed to combine frames into video", "RECONSTRUCTION_ERROR")
                    
                    result["processing_stages"].append({"stage": "reconstruction", "status": "completed", "time": time.time()})
                    self.memory_profiler.checkpoint("after_reconstruction")
                
                # Stage 7: Cleanup and finalization
                with self._error_handler("cleanup"):
                    if self.temp_cleanup:
                        self._update_progress(95, "Cleaning up temporary files...", progress_callback)
                        self._cleanup_temp_files(frame_files + processed_frames)
                    
                    # Final validation
                    if not os.path.exists(output_path):
                        raise ProcessingError("Output file was not created", "OUTPUT_ERROR")
                    
                    result["processing_stages"].append({"stage": "cleanup", "status": "completed", "time": time.time()})
                
                # Success
                self._update_progress(100, "Complete!", progress_callback)
                
                result["success"] = True
                result["output_path"] = output_path
                result["stats"]["original_size"] = os.path.getsize(input_path)
                result["stats"]["output_size"] = os.path.getsize(output_path)
                
                logger.info(f"Video upscaling completed successfully: {output_path}")
                
        except ProcessingError as e:
            result["error"] = e.message
            result["error_code"] = e.error_code
            result["error_details"] = e.details
            logger.error(f"Processing failed: {e.message}")
            
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            result["error_code"] = "UNEXPECTED_ERROR"
            logger.error(f"Unexpected processing error: {e}")
            
        finally:
            # Cleanup performance tracking
            if performance_context:
                try:
                    performance_context.__exit__(None, None, None)
                    if self.monitor:
                        result["performance"] = self.monitor.get_performance_summary(1)
                except:
                    pass
            
            # Add memory profiling results
            result["memory_profile"] = self.memory_profiler.get_report()
            
            # Save processing log
            self._save_processing_log(result)
        
        return result
    
    def _process_frames_ai_enhanced(self, frame_files: List[str], scale_factor: float,
                                   progress_callback: Callable, quality_settings: Dict = None) -> List[str]:
        """Process frames using enhanced AI with better error handling"""
        processed_dir = PATHS["temp_dir"] / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        def ai_progress_callback(progress, message):
            overall_progress = 30 + (progress * 0.5)  # Map to 30-80% range
            self._update_progress(overall_progress, f"AI Processing: {message}", progress_callback)
        
        try:
            if hasattr(self.ai_upscaler, 'upscale_batch_enhanced'):
                # Use enhanced batch processing
                result = self.ai_upscaler.upscale_batch_enhanced(
                    frame_files, str(processed_dir), scale_factor,
                    progress_callback=ai_progress_callback
                )
                
                if result["success"]:
                    return result["successful_outputs"]
                else:
                    logger.warning(f"Enhanced AI processing had issues: {result.get('error', 'Unknown error')}")
                    # Fallback to simple processing
                    return self._process_frames_simple(frame_files, scale_factor, progress_callback)
            
            elif hasattr(self.ai_upscaler, 'upscale_frames'):
                # Use Waifu2x frames processing
                logger.info("Using Waifu2x for frame processing")
                return self.ai_upscaler.upscale_frames(
                    frame_files, str(processed_dir), 
                    progress_callback=ai_progress_callback
                )
            else:
                # Use standard AI processing (for compatibility)
                return self.ai_upscaler.upscale_batch(
                    frame_files, str(processed_dir), scale_factor,
                    progress_callback=ai_progress_callback
                )
                
        except Exception as e:
            logger.warning(f"AI processing failed, falling back to simple upscaling: {e}")
            return self._process_frames_simple(frame_files, scale_factor, progress_callback)
    
    def _process_frames_simple(self, frame_files: List[str], scale_factor: float,
                              progress_callback: Callable) -> List[str]:
        """Process frames using simple upscaling with error handling"""
        processed_dir = PATHS["temp_dir"] / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        processed_frames = []
        failed_frames = []
        total_frames = len(frame_files)
        
        for i, frame_file in enumerate(frame_files):
            try:
                output_path = processed_dir / f"frame_{i:06d}_upscaled.png"
                
                if SimpleUpscaler.upscale_image_simple(frame_file, str(output_path), scale_factor):
                    processed_frames.append(str(output_path))
                else:
                    failed_frames.append(frame_file)
                    logger.warning(f"Failed to process frame: {frame_file}")
                
            except Exception as e:
                failed_frames.append(frame_file)
                logger.warning(f"Error processing frame {frame_file}: {e}")
            
            # Update progress
            progress = 30 + ((i + 1) / total_frames * 50)  # Map to 30-80% range
            self._update_progress(progress, f"Processing frame {i+1}/{total_frames}", progress_callback)
        
        if failed_frames:
            logger.warning(f"Failed to process {len(failed_frames)}/{total_frames} frames")
        
        return processed_frames
    
    def _estimate_memory_requirements(self, video_info: Dict, scale_factor: float) -> Dict:
        """Estimate memory requirements for processing"""
        frame_count = video_info.get("frame_count", 0)
        width = video_info.get("width", 640)
        height = video_info.get("height", 480)
        
        # Rough estimates
        frame_size_mb = (width * height * 3) / (1024 * 1024)  # RGB image
        upscaled_frame_size_mb = frame_size_mb * (scale_factor ** 2)
        
        estimates = {
            "per_frame_mb": frame_size_mb,
            "per_upscaled_frame_mb": upscaled_frame_size_mb,
            "temp_storage_gb": (frame_count * upscaled_frame_size_mb) / 1024,
            "peak_memory_gb": upscaled_frame_size_mb * 10 / 1024,  # Assume 10 frames in memory
            "recommended_ram_gb": max(8, (upscaled_frame_size_mb * 20) / 1024)
        }
        
        return estimates
    
    def _check_resource_availability(self, estimates: Dict) -> bool:
        """Check if system has sufficient resources"""
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            available_disk_gb = shutil.disk_usage(str(PATHS["temp_dir"])).free / (1024**3)
            
            memory_ok = available_memory_gb >= estimates.get("recommended_ram_gb", 8)
            disk_ok = available_disk_gb >= estimates.get("temp_storage_gb", 1)
            
            if not memory_ok:
                logger.warning(f"Low memory: {available_memory_gb:.1f}GB available, {estimates.get('recommended_ram_gb', 8):.1f}GB recommended")
            
            if not disk_ok:
                logger.warning(f"Low disk space: {available_disk_gb:.1f}GB available, {estimates.get('temp_storage_gb', 1):.1f}GB needed")
            
            return memory_ok and disk_ok
            
        except Exception as e:
            logger.warning(f"Resource check failed: {e}")
            return True  # Assume OK if check fails
    
    def _save_processing_log(self, result: Dict):
        """Save detailed processing log"""
        try:
            log_dir = PATHS["logs_dir"]
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / f"processing_log_{self.session_id}.json"
            
            log_data = {
                "session_id": self.session_id,
                "timestamp": time.time(),
                "result": result,
                "error_history": self.error_history,
                "system_info": self.get_system_info()
            }
            
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Failed to save processing log: {e}")
    
    def _cleanup_temp_files(self, file_list: List[str]):
        """Enhanced cleanup with better error handling"""
        cleanup_errors = []
        
        for file_path in file_list:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                cleanup_errors.append((file_path, str(e)))
        
        # Clean up empty directories
        try:
            temp_dirs = [PATHS["temp_dir"] / "processed"]
            for temp_dir in temp_dirs:
                if temp_dir.exists() and not any(temp_dir.iterdir()):
                    temp_dir.rmdir()
        except Exception as e:
            cleanup_errors.append(("temp_directories", str(e)))
        
        if cleanup_errors:
            logger.warning(f"Some cleanup operations failed: {len(cleanup_errors)} errors")
            for path, error in cleanup_errors[:5]:  # Log first 5 errors
                logger.debug(f"Cleanup error for {path}: {error}")
        else:
            logger.info("Temporary files cleaned up successfully")
    
    def _update_progress(self, progress: float, message: str, callback: Callable = None):
        """Update progress with validation"""
        self.current_progress = max(0, min(100, progress))  # Clamp to 0-100
        self.status_message = message
        
        if callback:
            try:
                callback(self.current_progress, message)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    def get_system_info_enhanced(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        base_info = self.get_system_info() if hasattr(self, 'get_system_info') else {}
        
        enhanced_info = {
            **base_info,
            "session_id": self.session_id,
            "session_uptime": time.time() - self.session_start,
            "error_count": len(self.error_history),
            "recovery_attempts": self.recovery_attempts,
            "monitoring_enabled": self.enable_monitoring,
            "enhanced_ai_enabled": self.use_enhanced_ai and self.use_ai
        }
        
        # Add performance monitoring data
        if self.monitor:
            try:
                enhanced_info["performance_stats"] = self.monitor.get_current_stats()
            except Exception as e:
                logger.warning(f"Failed to get performance stats: {e}")
        
        return enhanced_info
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get basic system information (fallback)"""
        try:
            import torch
            import platform
            
            info = {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "cuda_available": torch.cuda.is_available() if torch.cuda else False,
                "temp_dir": str(PATHS["temp_dir"]),
                "output_dir": str(PATHS["output_dir"]),
                "max_memory_gb": PERFORMANCE["max_memory_gb"]
            }
            
            if torch.cuda and torch.cuda.is_available():
                info["cuda_device_count"] = torch.cuda.device_count()
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
                info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            return info
            
        except Exception as e:
            logger.warning(f"System info collection failed: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Enhanced cleanup method"""
        try:
            if self.monitor:
                self.monitor.stop_monitoring()
            
            if self.ai_upscaler and hasattr(self.ai_upscaler, 'cleanup'):
                self.ai_upscaler.cleanup()
            
            # Emergency cleanup if needed
            self._emergency_cleanup()
            
            logger.info(f"Enhanced UpScale App cleanup completed (Session: {self.session_id})")
            
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass