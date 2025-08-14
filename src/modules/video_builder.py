"""
Video Builder Module - Reconstructs video from processed frames
Combines upscaled frames back into video with optional audio preservation
"""

import os
import logging
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class VideoBuilder:
    """
    Video Builder for combining processed frames back into video
    Supports audio preservation and various output formats with cancellation support
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize VideoBuilder
        
        Args:
            output_dir: Default output directory for videos
        """
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Process management
        self.current_process = None
        self.cancel_requested = False
        
        # Check for FFmpeg
        self._check_ffmpeg()
        
    def _check_ffmpeg(self):
        """Check if FFmpeg is available"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            logger.info("FFmpeg is available")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("FFmpeg not found. Some features may not work.")
            return False
    
    def combine_frames_to_video(
        self, 
        frame_files: List[str], 
        output_path: str,
        original_video_path: Optional[str] = None,
        fps: float = 30.0,
        preserve_audio: bool = True,
        quality: str = "high"
    ) -> bool:
        """
        Combine processed frames back into video
        
        Args:
            frame_files: List of frame file paths in order
            output_path: Output video file path
            original_video_path: Original video for audio extraction
            fps: Frame rate for output video
            preserve_audio: Whether to preserve audio from original
            quality: Video quality preset ('high', 'medium', 'fast')
            
        Returns:
            bool: Success status
        """
        try:
            if not frame_files:
                logger.error("No frame files provided")
                return False
                
            logger.info(f"Combining {len(frame_files)} frames to video: {output_path}")
            
            # Create temporary directory for frame sequence
            temp_dir = Path("temp_video_build")
            temp_dir.mkdir(exist_ok=True)
            
            try:
                # Copy and rename frames for FFmpeg sequence
                self._prepare_frame_sequence(frame_files, temp_dir)
                
                # Build FFmpeg command
                cmd = self._build_ffmpeg_command(
                    temp_dir, output_path, fps, quality, 
                    original_video_path if preserve_audio else None
                )
                
                # Execute FFmpeg with cancellation support
                self.current_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                try:
                    stdout, stderr = self.current_process.communicate()
                    
                    if self.current_process.returncode != 0:
                        raise subprocess.CalledProcessError(
                            self.current_process.returncode, cmd, stderr
                        )
                except Exception as e:
                    if self.cancel_requested:
                        self._kill_current_process()
                        raise Exception("Video building cancelled")
                    raise e
                finally:
                    self.current_process = None
                
                logger.info(f"Successfully created video: {output_path}")
                return True
                
            finally:
                # Cleanup temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Video building failed: {e}")
            return False
    
    def _prepare_frame_sequence(self, frame_files: List[str], temp_dir: Path):
        """Prepare frame sequence for FFmpeg"""
        for i, frame_file in enumerate(frame_files):
            if not os.path.exists(frame_file):
                logger.warning(f"Frame file not found: {frame_file}")
                continue
                
            # Copy frame with sequential naming
            dest_file = temp_dir / f"frame_{i:06d}.png"
            shutil.copy2(frame_file, dest_file)
    
    def _build_ffmpeg_command(
        self, 
        temp_dir: Path, 
        output_path: str, 
        fps: float,
        quality: str,
        audio_source: Optional[str] = None
    ) -> List[str]:
        """Build FFmpeg command for video creation"""
        
        # Quality presets
        quality_presets = {
            "fast": ["-preset", "fast", "-crf", "28"],
            "medium": ["-preset", "medium", "-crf", "23"],
            "high": ["-preset", "slow", "-crf", "18"]
        }
        
        quality_args = quality_presets.get(quality, quality_presets["medium"])
        
        # Base command
        cmd = [
            "ffmpeg", "-y",  # Overwrite output
            "-framerate", str(fps),
            "-i", str(temp_dir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p"
        ] + quality_args
        
        # Add audio if specified
        if audio_source and os.path.exists(audio_source):
            cmd.extend([
                "-i", audio_source,
                "-c:a", "aac",
                "-b:a", "128k",
                "-shortest"  # Match shortest stream
            ])
        
        cmd.append(output_path)
        
        return cmd
    
    def extract_audio(self, video_path: str, output_path: str) -> bool:
        """Extract audio from video file
        
        Args:
            video_path: Source video file
            output_path: Output audio file path
            
        Returns:
            bool: Success status
        """
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "copy",
                output_path
            ]
            
            self.current_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            
            try:
                stdout, stderr = self.current_process.communicate()
                
                if self.current_process.returncode != 0:
                    raise subprocess.CalledProcessError(
                        self.current_process.returncode, cmd, stderr
                    )
                    
                logger.info(f"Audio extracted to: {output_path}")
                return True
            finally:
                self.current_process = None
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio extraction failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Audio extraction error: {e}")
            return False
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information using FFprobe
        
        Args:
            video_path: Video file path
            
        Returns:
            Dict with video information
        """
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            import json
            data = json.loads(result.stdout)
            
            # Extract relevant info
            video_stream = None
            audio_stream = None
            
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video" and not video_stream:
                    video_stream = stream
                elif stream.get("codec_type") == "audio" and not audio_stream:
                    audio_stream = stream
            
            info = {
                "duration": float(data.get("format", {}).get("duration", 0)),
                "size": int(data.get("format", {}).get("size", 0)),
                "has_video": video_stream is not None,
                "has_audio": audio_stream is not None
            }
            
            if video_stream:
                info.update({
                    "width": int(video_stream.get("width", 0)),
                    "height": int(video_stream.get("height", 0)),
                    "fps": eval(video_stream.get("r_frame_rate", "0/1")),
                    "codec": video_stream.get("codec_name", "unknown")
                })
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return {}
    
    def create_test_video(
        self, 
        output_path: str, 
        width: int = 640, 
        height: int = 480,
        duration: float = 5.0, 
        fps: float = 30.0
    ) -> bool:
        """Create a test video for debugging
        
        Args:
            output_path: Output video path
            width: Video width
            height: Video height  
            duration: Video duration in seconds
            fps: Frame rate
            
        Returns:
            bool: Success status
        """
        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"testsrc=duration={duration}:size={width}x{height}:rate={fps}",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_path
            ]
            
            self.current_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            
            try:
                stdout, stderr = self.current_process.communicate()
                
                if self.current_process.returncode != 0:
                    raise subprocess.CalledProcessError(
                        self.current_process.returncode, cmd, stderr
                    )
                    
                logger.info(f"Test video created: {output_path}")
                return True
            finally:
                self.current_process = None
            
        except Exception as e:
            logger.error(f"Test video creation failed: {e}")
            return False
    
    def cancel_processing(self):
        """Request cancellation of current processing"""
        self.cancel_requested = True
        self._kill_current_process()
    
    def _kill_current_process(self):
        """Kill the current subprocess if it exists"""
        if self.current_process is None:
            return
            
        try:
            if self.current_process.poll() is None:  # Process is still running
                logger.info(f"Terminating FFmpeg process PID: {self.current_process.pid}")
                
                import os
                if os.name == 'nt':  # Windows
                    # Use taskkill to terminate the entire process tree
                    subprocess.run([
                        "taskkill", "/F", "/T", "/PID", str(self.current_process.pid)
                    ], capture_output=True)
                else:  # Unix-like
                    # Send SIGTERM first, then SIGKILL if necessary
                    self.current_process.terminate()
                    try:
                        self.current_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.current_process.kill()
                        self.current_process.wait()
                
                logger.info("FFmpeg process terminated successfully")
                
        except Exception as e:
            logger.error(f"Error terminating FFmpeg process: {e}")
        finally:
            self.current_process = None
    
    def reset_cancel_flag(self):
        """Reset the cancellation flag for new operations"""
        self.cancel_requested = False