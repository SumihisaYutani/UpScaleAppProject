"""
UpScale App - Video Processing Module
Consolidated video processing using external binaries
"""

import os
import json
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from PIL import Image

from .utils import ProgressCallback, SimpleTimer, format_duration

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Handles all video processing operations using external binaries"""
    
    def __init__(self, resource_manager, temp_dir: Path):
        self.resource_manager = resource_manager
        self.temp_dir = Path(temp_dir)
        self.frame_dir = self.temp_dir / "frames"
        self.frame_dir.mkdir(exist_ok=True)
        
        logger.info(f"VideoProcessor initialized - Temp dir: {self.temp_dir}")
    
    def validate_video(self, video_path: str) -> Dict[str, Any]:
        """Validate video file and get information"""
        video_path = Path(video_path)
        
        if not video_path.exists():
            return {
                'valid': False,
                'error': 'File does not exist',
                'info': None
            }
        
        try:
            # Use ffprobe to get video information
            ffprobe_path = self.resource_manager.get_binary_path('ffprobe')
            if not ffprobe_path:
                return {
                    'valid': False,
                    'error': 'FFprobe not available',
                    'info': None
                }
            
            cmd = [
                ffprobe_path,
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return {
                    'valid': False,
                    'error': f'FFprobe failed: {result.stderr}',
                    'info': None
                }
            
            # Parse JSON output
            probe_data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in probe_data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                return {
                    'valid': False,
                    'error': 'No video stream found',
                    'info': None
                }
            
            # Extract video information
            format_info = probe_data.get('format', {})
            duration = float(format_info.get('duration', 0))
            file_size = int(format_info.get('size', 0))
            
            info = {
                'filename': video_path.name,
                'path': str(video_path),
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'duration': duration,
                'frame_rate': eval(video_stream.get('r_frame_rate', '30/1')),
                'frame_count': int(duration * eval(video_stream.get('r_frame_rate', '30/1'))),
                'codec_name': video_stream.get('codec_name', 'unknown'),
                'size': file_size,
                'format': format_info.get('format_name', 'unknown')
            }
            
            return {
                'valid': True,
                'error': None,
                'info': info
            }
            
        except Exception as e:
            logger.error(f"Video validation failed: {e}")
            return {
                'valid': False,
                'error': f'Validation error: {str(e)}',
                'info': None
            }
    
    def extract_frames(self, video_path: str, progress_callback: Optional[Callable] = None) -> List[str]:
        """Extract frames from video using FFmpeg"""
        video_path = Path(video_path)
        
        # Clear previous frames
        for frame_file in self.frame_dir.glob("*.png"):
            frame_file.unlink()
        
        logger.info(f"Extracting frames from: {video_path}")
        
        try:
            ffmpeg_path = self.resource_manager.get_binary_path('ffmpeg')
            if not ffmpeg_path:
                raise RuntimeError("FFmpeg not available")
            
            # Extract frames command
            output_pattern = str(self.frame_dir / "frame_%06d.png")
            cmd = [
                ffmpeg_path,
                '-i', str(video_path),
                '-vf', 'fps=fps=30',  # Extract at 30fps max
                '-y',  # Overwrite existing files
                output_pattern
            ]
            
            logger.info("Using FFmpeg for frame extraction...")
            
            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg failed: {result.stderr}")
                raise RuntimeError(f"FFmpeg frame extraction failed: {result.stderr}")
            
            # Get list of extracted frames
            frame_files = sorted(self.frame_dir.glob("frame_*.png"))
            frame_paths = [str(f) for f in frame_files]
            
            logger.info(f"FFmpeg successfully extracted {len(frame_paths)} frames")
            
            if progress_callback:
                progress_callback(100, f"Extracted {len(frame_paths)} frames")
            
            return frame_paths
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            raise
    
    def combine_frames_to_video(self, frame_paths: List[str], output_path: str, 
                               original_video_path: str, fps: float = 30.0,
                               progress_callback: Optional[Callable] = None) -> bool:
        """Combine frames back into video"""
        
        if not frame_paths:
            logger.error("No frames to combine")
            return False
        
        try:
            ffmpeg_path = self.resource_manager.get_binary_path('ffmpeg')
            if not ffmpeg_path:
                raise RuntimeError("FFmpeg not available")
            
            # Create temporary frame list
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for frame_path in frame_paths:
                    f.write(f"file '{frame_path}'\\n")
                    f.write(f"duration {1/fps}\\n")
                frame_list_path = f.name
            
            try:
                # FFmpeg command to combine frames
                cmd = [
                    ffmpeg_path,
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', frame_list_path,
                    '-i', original_video_path,  # For audio track
                    '-c:v', 'libx264',
                    '-c:a', 'copy',  # Copy audio from original
                    '-map', '0:v:0',
                    '-map', '1:a:0?',  # Audio is optional
                    '-r', str(fps),
                    '-pix_fmt', 'yuv420p',
                    '-y',
                    output_path
                ]
                
                logger.info(f"Combining {len(frame_paths)} frames into video...")
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
                
                if result.returncode != 0:
                    logger.error(f"FFmpeg combine failed: {result.stderr}")
                    return False
                
                # Verify output file exists
                if not Path(output_path).exists():
                    logger.error("Output video file was not created")
                    return False
                
                logger.info(f"Successfully created video: {output_path}")
                
                if progress_callback:
                    progress_callback(100, "Video reconstruction complete")
                
                return True
                
            finally:
                # Clean up temporary file
                try:
                    Path(frame_list_path).unlink()
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Frame combination failed: {e}")
            return False
    
    def get_upscaled_resolution(self, width: int, height: int, scale_factor: float) -> tuple:
        """Calculate upscaled resolution"""
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        return (new_width, new_height)
    
    def estimate_processing_time(self, video_info: Dict) -> Dict[str, Any]:
        """Estimate processing time based on video properties"""
        frame_count = video_info.get('frame_count', 0)
        width = video_info.get('width', 640)
        height = video_info.get('height', 480)
        
        # Simple time estimation (very rough)
        pixels_per_frame = width * height
        
        # Estimate based on frame complexity
        time_per_frame_cpu = pixels_per_frame / 500000  # seconds per frame for CPU
        time_per_frame_gpu = pixels_per_frame / 2000000  # seconds per frame for GPU
        
        total_cpu_time = frame_count * time_per_frame_cpu
        total_gpu_time = frame_count * time_per_frame_gpu
        
        return {
            'estimated_cpu_minutes': total_cpu_time / 60,
            'estimated_gpu_minutes': total_gpu_time / 60,
            'frame_count': frame_count,
            'complexity_score': pixels_per_frame / 1000000  # MP rating
        }
    
    def cleanup(self):
        """Clean up video processor resources"""
        try:
            # Clean up frame directory
            if self.frame_dir.exists():
                for frame_file in self.frame_dir.glob("*.png"):
                    frame_file.unlink()
                logger.info("Cleaned up frame files")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")