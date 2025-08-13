"""
Main UpScale Application
Coordinates all modules to perform video upscaling
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Optional, Callable, Dict
from tqdm import tqdm

from modules.video_processor import VideoProcessor, VideoFrameExtractor
from modules.video_builder import VideoBuilder
from modules.ai_processor import AIUpscaler, SimpleUpscaler
from config.settings import PATHS, VIDEO_SETTINGS, PERFORMANCE

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UpScaleApp:
    """Main application class for video upscaling"""
    
    def __init__(self, use_ai: bool = True, temp_cleanup: bool = True):
        self.use_ai = use_ai
        self.temp_cleanup = temp_cleanup
        
        # Initialize modules
        self.video_processor = VideoProcessor()
        self.frame_extractor = VideoFrameExtractor(str(PATHS["temp_dir"]))
        self.video_builder = VideoBuilder(str(PATHS["output_dir"]))
        self.ai_upscaler = AIUpscaler() if use_ai else None
        
        # Progress tracking
        self.current_progress = 0
        self.status_message = ""
    
    def process_video(self, input_path: str, output_path: str = None,
                     scale_factor: float = None, 
                     progress_callback: Callable = None) -> Dict[str, any]:
        """
        Process video with AI upscaling
        
        Args:
            input_path (str): Path to input MP4 file
            output_path (str): Path for output file (optional)
            scale_factor (float): Upscaling factor (default from settings)
            progress_callback (Callable): Progress callback function
            
        Returns:
            Dict containing processing results
        """
        result = {
            "success": False,
            "output_path": None,
            "error": None,
            "stats": {}
        }
        
        try:
            # Step 1: Validate input file
            self._update_progress(5, "Validating input file...", progress_callback)
            validation = self.video_processor.validate_video_file(input_path)
            
            if not validation["valid"]:
                result["error"] = validation["error"]
                return result
            
            video_info = validation["info"]
            logger.info(f"Processing video: {video_info['filename']}")
            
            # Step 2: Prepare output path
            if output_path is None:
                output_filename = f"{Path(input_path).stem}_upscaled.mp4"
                output_path = str(PATHS["output_dir"] / output_filename)
            
            # Step 3: Calculate upscaling parameters
            if scale_factor is None:
                scale_factor = VIDEO_SETTINGS["default_upscale_factor"]
            
            original_resolution = (video_info["width"], video_info["height"])
            new_resolution = self.video_processor.get_upscaled_resolution(
                video_info["width"], video_info["height"], scale_factor
            )
            
            logger.info(f"Upscaling from {original_resolution} to {new_resolution}")
            
            # Step 4: Estimate processing time
            time_estimates = self.video_processor.estimate_processing_time(video_info)
            result["stats"]["estimated_time"] = time_estimates
            
            # Step 5: Extract frames
            self._update_progress(10, "Extracting frames...", progress_callback)
            frame_files = self.frame_extractor.extract_frames(input_path)
            
            if not frame_files:
                result["error"] = "Failed to extract frames from video"
                return result
            
            logger.info(f"Extracted {len(frame_files)} frames")
            result["stats"]["frame_count"] = len(frame_files)
            
            # Step 6: Process frames with AI
            processed_frames = []
            if self.use_ai and self.ai_upscaler:
                self._update_progress(20, "Loading AI model...", progress_callback)
                if not self.ai_upscaler.load_model():
                    logger.warning("Failed to load AI model, falling back to simple upscaling")
                    self.use_ai = False
            
            if self.use_ai and self.ai_upscaler:
                self._update_progress(25, "Processing frames with AI...", progress_callback)
                processed_frames = self._process_frames_ai(
                    frame_files, scale_factor, progress_callback
                )
            else:
                self._update_progress(25, "Processing frames with simple upscaling...", progress_callback)
                processed_frames = self._process_frames_simple(
                    frame_files, scale_factor, progress_callback
                )
            
            if not processed_frames:
                result["error"] = "Failed to process any frames"
                return result
            
            logger.info(f"Successfully processed {len(processed_frames)} frames")
            result["stats"]["processed_frames"] = len(processed_frames)
            
            # Step 7: Combine frames back into video
            self._update_progress(90, "Combining frames into video...", progress_callback)
            
            fps = video_info.get("frame_rate", 30.0)
            success = self.video_builder.combine_frames_to_video(
                processed_frames, output_path, input_path, fps, preserve_audio=True
            )
            
            if not success:
                result["error"] = "Failed to combine frames into video"
                return result
            
            # Step 8: Cleanup temporary files
            if self.temp_cleanup:
                self._update_progress(95, "Cleaning up temporary files...", progress_callback)
                self._cleanup_temp_files(frame_files + processed_frames)
            
            # Step 9: Final validation
            self._update_progress(100, "Complete!", progress_callback)
            
            result["success"] = True
            result["output_path"] = output_path
            result["stats"]["original_size"] = os.path.getsize(input_path)
            result["stats"]["output_size"] = os.path.getsize(output_path)
            result["stats"]["scale_factor"] = scale_factor
            result["stats"]["original_resolution"] = original_resolution
            result["stats"]["output_resolution"] = new_resolution
            
            logger.info(f"Video upscaling completed: {output_path}")
            
        except Exception as e:
            result["error"] = f"Processing failed: {str(e)}"
            logger.error(f"Video processing error: {e}")
        
        return result
    
    def _process_frames_ai(self, frame_files: list, scale_factor: float,
                          progress_callback: Callable = None) -> list:
        """Process frames using AI upscaling"""
        processed_dir = PATHS["temp_dir"] / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        def ai_progress_callback(progress, message):
            # Map AI progress to overall progress (25-85%)
            overall_progress = 25 + (progress * 0.6)
            self._update_progress(overall_progress, f"AI Processing: {message}", progress_callback)
        
        return self.ai_upscaler.upscale_batch(
            frame_files, str(processed_dir), scale_factor,
            progress_callback=ai_progress_callback
        )
    
    def _process_frames_simple(self, frame_files: list, scale_factor: float,
                              progress_callback: Callable = None) -> list:
        """Process frames using simple upscaling"""
        processed_dir = PATHS["temp_dir"] / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        processed_frames = []
        total_frames = len(frame_files)
        
        for i, frame_file in enumerate(frame_files):
            output_path = processed_dir / f"frame_{i:06d}_upscaled.png"
            
            if SimpleUpscaler.upscale_image_simple(
                frame_file, str(output_path), scale_factor
            ):
                processed_frames.append(str(output_path))
            
            # Update progress (25-85%)
            progress = 25 + ((i + 1) / total_frames * 60)
            self._update_progress(progress, f"Processing frame {i+1}/{total_frames}", 
                                progress_callback)
        
        return processed_frames
    
    def _cleanup_temp_files(self, file_list: list):
        """Clean up temporary files"""
        try:
            for file_path in file_list:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Clean up empty directories
            temp_dirs = [PATHS["temp_dir"] / "processed"]
            for temp_dir in temp_dirs:
                if temp_dir.exists() and not any(temp_dir.iterdir()):
                    temp_dir.rmdir()
            
            logger.info("Temporary files cleaned up")
            
        except Exception as e:
            logger.warning(f"Failed to clean up some temporary files: {e}")
    
    def _update_progress(self, progress: float, message: str, 
                        callback: Callable = None):
        """Update progress and status"""
        self.current_progress = progress
        self.status_message = message
        
        if callback:
            callback(progress, message)
    
    def create_preview(self, input_path: str, output_path: str = None) -> str:
        """
        Create a preview of the upscaling process
        
        Args:
            input_path (str): Input video path
            output_path (str): Output preview path
            
        Returns:
            str: Path to preview file or None if failed
        """
        try:
            if output_path is None:
                preview_filename = f"{Path(input_path).stem}_preview.mp4"
                output_path = str(PATHS["output_dir"] / preview_filename)
            
            # Extract limited frames for preview
            preview_frames = self.frame_extractor.extract_frames(input_path)[:30]  # First 30 frames
            
            if not preview_frames:
                return None
            
            # Simple upscaling for preview (faster)
            processed_frames = self._process_frames_simple(preview_frames, 1.5)
            
            # Create preview video
            if self.video_builder.create_preview_video(processed_frames, output_path):
                self._cleanup_temp_files(preview_frames + processed_frames)
                return output_path
            
        except Exception as e:
            logger.error(f"Preview creation failed: {e}")
        
        return None
    
    def get_system_info(self) -> Dict[str, any]:
        """Get system information for diagnostics"""
        import torch
        import platform
        
        info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "temp_dir": str(PATHS["temp_dir"]),
            "output_dir": str(PATHS["output_dir"]),
            "max_memory_gb": PERFORMANCE["max_memory_gb"]
        }
        
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info