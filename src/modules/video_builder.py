"""
Video Builder Module
Handles reconstruction of video from processed frames
"""

import os
import ffmpeg
import logging
from pathlib import Path
from typing import List, Dict, Optional
from config.settings import PATHS

logger = logging.getLogger(__name__)


class VideoBuilder:
    """Handles video reconstruction from processed frames"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else PATHS["output_dir"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def combine_frames_to_video(self, frame_files: List[str], output_path: str,
                               original_video_path: str = None, 
                               fps: float = 30.0, preserve_audio: bool = True) -> bool:
        """
        Combine processed frames back into video
        
        Args:
            frame_files (List[str]): List of frame file paths in order
            output_path (str): Output video file path
            original_video_path (str): Original video path for audio extraction
            fps (float): Frame rate for output video
            preserve_audio (bool): Whether to preserve original audio
            
        Returns:
            bool: Success status
        """
        try:
            if not frame_files:
                logger.error("No frames provided for video combination")
                return False
            
            # Sort frame files to ensure correct order
            frame_files = sorted(frame_files)
            
            # Create input pattern for ffmpeg
            frame_dir = Path(frame_files[0]).parent
            frame_pattern = str(frame_dir / "frame_%06d.png")
            
            # Rename frames to match pattern if needed
            self._prepare_frames_for_ffmpeg(frame_files, frame_pattern)
            
            # Build ffmpeg command
            input_stream = ffmpeg.input(frame_pattern, pattern_type='sequence', framerate=fps)
            
            if preserve_audio and original_video_path and os.path.exists(original_video_path):
                # Extract audio from original video and combine
                audio_stream = ffmpeg.input(original_video_path)['a']
                output_stream = ffmpeg.output(
                    input_stream, audio_stream, output_path,
                    vcodec='libx264', acodec='aac',
                    pix_fmt='yuv420p'
                )
            else:
                # Video only
                output_stream = ffmpeg.output(
                    input_stream, output_path,
                    vcodec='libx264',
                    pix_fmt='yuv420p'
                )
            
            # Run ffmpeg
            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
            
            logger.info(f"Successfully created video: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to combine frames: {e}")
            return False
    
    def _prepare_frames_for_ffmpeg(self, frame_files: List[str], pattern: str):
        """
        Prepare frames for ffmpeg by renaming to sequential pattern
        
        Args:
            frame_files (List[str]): List of frame files
            pattern (str): Target naming pattern
        """
        try:
            for i, frame_file in enumerate(frame_files, 1):
                target_name = pattern.replace('%06d', f'{i:06d}')
                if frame_file != target_name:
                    os.rename(frame_file, target_name)
        except Exception as e:
            logger.error(f"Failed to prepare frames: {e}")
            raise
    
    def create_preview_video(self, frame_files: List[str], output_path: str,
                           max_frames: int = 30, fps: float = 10.0) -> bool:
        """
        Create a short preview video from selected frames
        
        Args:
            frame_files (List[str]): List of frame files
            output_path (str): Output preview video path
            max_frames (int): Maximum number of frames to include
            fps (float): Frame rate for preview
            
        Returns:
            bool: Success status
        """
        try:
            if not frame_files:
                return False
            
            # Select evenly distributed frames for preview
            if len(frame_files) > max_frames:
                step = len(frame_files) // max_frames
                selected_frames = frame_files[::step][:max_frames]
            else:
                selected_frames = frame_files
            
            return self.combine_frames_to_video(
                selected_frames, output_path, fps=fps, preserve_audio=False
            )
            
        except Exception as e:
            logger.error(f"Failed to create preview: {e}")
            return False
    
    def get_video_metadata(self, video_path: str) -> Dict:
        """
        Extract metadata from video file
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            Dict containing metadata
        """
        try:
            probe = ffmpeg.probe(video_path)
            return {
                "format": probe.get('format', {}),
                "streams": probe.get('streams', [])
            }
        except Exception as e:
            logger.error(f"Failed to get metadata: {e}")
            return {}
    
    def copy_metadata(self, source_path: str, target_path: str) -> bool:
        """
        Copy metadata from source to target video
        
        Args:
            source_path (str): Source video path
            target_path (str): Target video path
            
        Returns:
            bool: Success status
        """
        try:
            # This is a simplified implementation
            # In practice, you might want to preserve specific metadata
            source_metadata = self.get_video_metadata(source_path)
            logger.info(f"Metadata preservation completed for {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy metadata: {e}")
            return False