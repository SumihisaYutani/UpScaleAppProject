"""
UpScale App - Video Processing Module
Consolidated video processing using external binaries
"""

import os
import json
import subprocess
import tempfile
import logging
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from PIL import Image

from .utils import ProgressCallback, SimpleTimer, format_duration
from .fast_frame_extractor import FastFrameExtractor

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Handles all video processing operations using external binaries"""
    
    def __init__(self, resource_manager, temp_dir: Path, gpu_info: Dict = None):
        self.resource_manager = resource_manager
        self.temp_dir = Path(temp_dir)
        self.frame_dir = self.temp_dir / "frames"
        self.gpu_info = gpu_info or {}
        
        # Create frame directory with error handling
        try:
            self.frame_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            if e.winerror == 183:  # File already exists
                logger.warning(f"Frame directory creation race condition, continuing: {self.frame_dir}")
                if not (self.frame_dir.exists() and self.frame_dir.is_dir()):
                    raise RuntimeError(f"Frame directory is not usable: {self.frame_dir}")
            else:
                raise
        
        # Initialize FastFrameExtractor with GPU support for optimized processing
        self.fast_extractor = FastFrameExtractor(resource_manager, str(self.temp_dir), self.gpu_info)
        
        logger.info(f"VideoProcessor initialized - Temp dir: {self.temp_dir}")
        logger.info(f"FastFrameExtractor initialized with GPU support for high-performance frame extraction")
    
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
            # Try ffprobe first, fallback to OpenCV if not available
            ffprobe_path = self.resource_manager.get_binary_path('ffprobe')
            if not ffprobe_path:
                logger.warning("FFprobe not available, using OpenCV fallback")
                return self._validate_video_with_opencv(video_path)
            
            # Ensure proper path handling for special characters
            video_path_str = str(video_path)
            
            cmd = [
                ffprobe_path,
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path_str
            ]
            
            # Hide console window on Windows
            startupinfo = None
            if os.name == 'nt':  # Windows
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            
            logger.info(f"DEBUG: Executing ffprobe command: {' '.join(cmd)}")
            logger.info(f"DEBUG: Video path: '{video_path_str}' (length: {len(video_path_str)})")
            
            # Use shell=False and ensure proper encoding on Windows
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                startupinfo=startupinfo,
                shell=False,
                encoding='utf-8'
            )
            
            logger.info(f"DEBUG: FFprobe return code: {result.returncode}")
            logger.info(f"DEBUG: FFprobe stdout length: {len(result.stdout) if result.stdout else 0}")
            logger.info(f"DEBUG: FFprobe stderr: {result.stderr}")
            
            if result.returncode != 0:
                logger.error(f"DEBUG: FFprobe failed with return code {result.returncode}")
                return {
                    'valid': False,
                    'error': f'FFprobe failed: {result.stderr}',
                    'info': None
                }
            
            # Parse JSON output
            try:
                if not result.stdout or not result.stdout.strip():
                    logger.error(f"DEBUG: FFprobe returned empty output - stdout: '{result.stdout}'")
                    return {
                        'valid': False,
                        'error': 'FFprobe returned empty output',
                        'info': None
                    }
                
                # Handle JSON parsing with multiple fallback methods
                json_output = result.stdout
                probe_data = None
                
                # Method 1: Direct parsing
                try:
                    probe_data = json.loads(json_output)
                except json.JSONDecodeError:
                    pass
                
                # Method 2: Fix common Windows path escaping issues
                if probe_data is None:
                    try:
                        # Replace backslashes that cause JSON parsing issues
                        fixed_output = json_output.replace('\\', '/')
                        # Also handle other problematic escape sequences
                        fixed_output = fixed_output.replace('\\"', '"')
                        probe_data = json.loads(fixed_output)
                        logger.info("JSON parsing successful after path fixing")
                    except json.JSONDecodeError:
                        pass
                
                # Method 3: Remove problematic characters
                if probe_data is None:
                    try:
                        # Remove non-printable characters except standard whitespace
                        clean_output = ''.join(char for char in json_output if char.isprintable() or char in '\n\r\t ')
                        probe_data = json.loads(clean_output)
                        logger.info("JSON parsing successful after character cleaning")
                    except json.JSONDecodeError:
                        pass
                
                # Method 4: Use regex to fix paths in JSON
                if probe_data is None:
                    try:
                        import re
                        # Replace Windows paths in JSON with forward slashes
                        regex_fixed = re.sub(r'"([A-Za-z]:[^"]*)"', lambda m: '"' + m.group(1).replace('\\', '/') + '"', json_output)
                        probe_data = json.loads(regex_fixed)
                        logger.info("JSON parsing successful after regex fixing")
                    except (json.JSONDecodeError, re.error):
                        pass
                
                if probe_data is None:
                    raise json.JSONDecodeError("All JSON parsing methods failed", json_output, 0)
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed for video {video_path.name}: {e}")
                logger.error(f"FFprobe stdout: {result.stdout[:1000]}")  # Log first 1000 chars
                logger.error(f"Error details: {str(e)}")
                
                # Fallback to OpenCV if JSON parsing completely fails
                logger.warning("Falling back to OpenCV for video validation")
                return self._validate_video_with_opencv(video_path)
            
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
            
            # Get accurate frame count using multiple methods
            frame_count_accurate = self._get_accurate_frame_count(video_path, video_stream, duration)
            
            info = {
                'filename': video_path.name,
                'path': str(video_path),
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'duration': duration,
                'frame_rate': eval(video_stream.get('r_frame_rate', '30/1')),
                'frame_count': frame_count_accurate,
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
    
    def _get_accurate_frame_count(self, video_path: Path, video_stream: dict, duration: float) -> int:
        """Get accurate frame count using multiple detection methods"""
        try:
            logger.info(f"DEBUG: Starting accurate frame count detection for: {video_path.name}")
            
            # Method 1: Try to get frame count directly from stream metadata
            if 'nb_frames' in video_stream and video_stream['nb_frames'] != 'N/A':
                nb_frames = int(video_stream['nb_frames'])
                if nb_frames > 0:
                    logger.info(f"DEBUG: Frame count from nb_frames: {nb_frames}")
                    return nb_frames
            
            # Method 2: Use ffprobe with frame counting
            logger.info("DEBUG: Using ffprobe frame counting method...")
            frame_count_probe = self._count_frames_with_ffprobe(video_path)
            if frame_count_probe > 0:
                logger.info(f"DEBUG: Frame count from ffprobe counting: {frame_count_probe}")
                return frame_count_probe
            
            # Method 3: Fallback to duration * frame_rate calculation
            frame_rate = eval(video_stream.get('r_frame_rate', '30/1'))
            calculated_frames = int(duration * frame_rate)
            logger.info(f"DEBUG: Frame count from duration calculation: {calculated_frames} (duration: {duration}s, fps: {frame_rate})")
            
            return calculated_frames
            
        except Exception as e:
            logger.warning(f"DEBUG: Accurate frame count detection failed: {e}")
            # Ultimate fallback
            frame_rate = eval(video_stream.get('r_frame_rate', '30/1'))
            return int(duration * frame_rate)
    
    def _count_frames_with_ffprobe(self, video_path: Path) -> int:
        """Count frames using ffprobe frame analysis"""
        try:
            ffprobe_path = self.resource_manager.get_binary_path('ffprobe')
            if not ffprobe_path:
                return 0
            
            # Use ffprobe to count frames - most accurate method
            cmd = [
                ffprobe_path,
                '-v', 'error',
                '-select_streams', 'v:0',
                '-count_frames',
                '-show_entries', 'stream=nb_read_frames',
                '-csv=p=0',
                str(video_path)
            ]
            
            logger.info(f"DEBUG: Running frame count command: {' '.join(cmd)}")
            
            # Hide console window on Windows
            startupinfo = None
            if os.name == 'nt':  # Windows
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                startupinfo=startupinfo,
                shell=False,
                encoding='utf-8'
            )
            
            if result.returncode == 0 and result.stdout and result.stdout.strip():
                frame_count = int(result.stdout.strip())
                logger.info(f"DEBUG: ffprobe frame count result: {frame_count}")
                return frame_count
            else:
                logger.warning(f"DEBUG: ffprobe frame counting failed: {result.stderr}")
                return 0
                
        except subprocess.TimeoutExpired:
            logger.warning("DEBUG: ffprobe frame counting timed out")
            return 0
        except Exception as e:
            logger.warning(f"DEBUG: ffprobe frame counting error: {e}")
            return 0
    
    def extract_frames(self, video_path: str, progress_callback: Optional[Callable] = None, progress_dialog=None) -> List[str]:
        """Extract frames from video using optimized FastFrameExtractor"""
        video_path = Path(video_path)
        
        # Add debug messages to GUI log
        if progress_dialog:
            try:
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"INFO: Starting optimized frame extraction from: {video_path.name}"))
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"INFO: Using FastFrameExtractor for improved performance"))
            except:
                pass
        
        logger.info(f"Starting optimized frame extraction from: {video_path}")
        
        # Clear previous frames
        logger.info("Clearing previous frames...")
        png_count = len(list(self.frame_dir.glob("*.png")))
        jpg_count = len(list(self.frame_dir.glob("*.jpg")))
        logger.info(f"Found {png_count} PNG files, {jpg_count} JPG files to clean")
        
        for frame_file in self.frame_dir.glob("*.png"):
            frame_file.unlink()
        for frame_file in self.frame_dir.glob("*.jpg"):
            frame_file.unlink()
        
        try:
            # Get video info for processing decision
            video_info = self.validate_video(str(video_path))
            
            if not video_info['valid']:
                if progress_dialog:
                    progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"ERROR: Video validation failed: {video_info['error']}"))
                raise RuntimeError(f"Invalid video: {video_info['error']}")
            
            total_frames = video_info['info']['frame_count']
            duration = video_info['info']['duration']
            frame_rate = video_info['info']['frame_rate']
            
            if progress_dialog:
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"INFO: Video: {total_frames} frames, {duration:.1f}s, {frame_rate:.2f} fps"))
                
                # Show estimated time
                estimated_time = self.fast_extractor.estimate_extraction_time(total_frames)
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"INFO: Estimated extraction time: {estimated_time/60:.1f} minutes"))
            
            logger.info(f"Video details - Total frames: {total_frames}, Duration: {duration}s, Frame rate: {frame_rate}")
            
            # Use FastFrameExtractor for all videos
            logger.info("Using FastFrameExtractor for optimized parallel processing")
            frame_paths = self.fast_extractor.extract_frames_parallel(
                video_path, total_frames, duration, progress_callback, progress_dialog
            )
            
            logger.info(f"FastFrameExtractor completed: {len(frame_paths)} frames extracted")
            return frame_paths
            
        except Exception as e:
            logger.error(f"Optimized frame extraction failed: {e}")
            if progress_dialog:
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"ERROR: Frame extraction failed: {str(e)}"))
            
            # Fallback to original method if FastFrameExtractor fails
            logger.info("Falling back to original frame extraction method...")
            if progress_dialog:
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message("INFO: Falling back to original extraction method"))
            
            return self._extract_frames_fallback(video_path, progress_callback, progress_dialog)
    
    def _extract_frames_fallback(self, video_path: Path, progress_callback: Optional[Callable] = None, progress_dialog=None) -> List[str]:
        """Fallback frame extraction using original single-pass method"""
        return self._extract_frames_single_pass(video_path, progress_callback, progress_dialog)
    
    def _extract_frames_single_pass(self, video_path: Path, progress_callback: Optional[Callable] = None, progress_dialog=None) -> List[str]:
        """Extract frames in a single pass for smaller videos"""
        ffmpeg_path = self.resource_manager.get_binary_path('ffmpeg')
        
        # Extract frames command - optimized PNG with memory management
        # PNG format maintained for maximum quality, with optimizations
        output_pattern = str(self.frame_dir / "frame_%06d.png")
        cmd = [
            ffmpeg_path,
            '-i', str(video_path),
            '-pix_fmt', 'rgb24',  # Optimal pixel format for waifu2x
            '-compression_level', '9',  # Maximum PNG compression (0-9) for space efficiency
            '-pred', 'mixed',  # PNG prediction method for better compression
            '-y',  # Overwrite existing files
            output_pattern
        ]
        
        logger.info("Using FFmpeg for single-pass frame extraction...")
        
        if progress_callback:
            progress_callback(10, "Starting frame extraction...")
        
        # Start FFmpeg process with real-time monitoring
        import time
        import threading
        
        # Hide console window on Windows
        startupinfo = None
        if os.name == 'nt':  # Windows
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                 text=True, bufsize=1, universal_newlines=True,
                                 startupinfo=startupinfo)
        
        # Register process with progress dialog for cancellation
        if progress_dialog:
            progress_dialog.current_process = process
        
        # Monitor progress in background thread
        def monitor_progress():
            start_time = time.time()
            last_frame_count = 0
            stagnant_count = 0
            
            while process.poll() is None:
                # Check for cancellation
                if progress_dialog and progress_dialog.cancelled:
                    try:
                        # Forcefully terminate FFmpeg process
                        process.terminate()
                        try:
                            process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=2)
                        
                        # Kill any child processes on Windows
                        import os
                        if os.name == 'nt':  # Windows
                            try:
                                import subprocess
                                subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], 
                                             capture_output=True, timeout=3)
                            except:
                                pass
                    except:
                        pass
                    break
                
                # Count current extracted frames with error handling
                try:
                    current_frames = len(list(self.frame_dir.glob("frame_*.png")))
                except:
                    current_frames = last_frame_count  # Use last known count on error
                
                if current_frames > 0:
                    elapsed = time.time() - start_time
                    
                    # Check if frames are being written
                    if current_frames == last_frame_count:
                        stagnant_count += 1
                        status_msg = f"Extracting frames... {current_frames} frames extracted (processing...)"
                    else:
                        stagnant_count = 0
                        rate = current_frames / elapsed if elapsed > 0 else 0
                        status_msg = f"Extracting frames... {current_frames} frames extracted ({rate:.1f} fps)"
                    
                    if progress_callback:
                        progress_callback(20 + min(60, (current_frames / 1000) * 60), status_msg)
                    
                    last_frame_count = current_frames
                
                time.sleep(1.5)  # Check more frequently
        
        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()
        
        # Wait for process completion without timeout
        try:
            stdout, stderr = process.communicate()
        except Exception as e:
            process.kill()
            raise RuntimeError(f"Frame extraction failed: {e}")
        
        # Check if cancelled during processing
        if progress_dialog and progress_dialog.cancelled:
            raise KeyboardInterrupt("Processing cancelled by user")
        
        if progress_callback:
            progress_callback(85, "Frame extraction completed, counting final frames...")
        
        if process.returncode != 0:
            logger.error(f"FFmpeg failed: {stderr}")
            raise RuntimeError(f"FFmpeg frame extraction failed: {stderr}")
        
        # Get final list of extracted frames
        frame_files = sorted(self.frame_dir.glob("frame_*.png"))
        frame_paths = [str(f) for f in frame_files]
        
        logger.info(f"FFmpeg successfully extracted {len(frame_paths)} frames")
        
        if progress_callback:
            progress_callback(100, f"Extracted {len(frame_paths)} frames")
        
        return frame_paths
    
    def _extract_frames_in_batches(self, video_path: Path, total_frames: int, duration: float, 
                                  batch_size: int, progress_callback: Optional[Callable] = None, 
                                  progress_dialog=None) -> List[str]:
        """Extract frames in smaller batches to prevent memory exhaustion"""
        if progress_dialog:
            progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Starting batch frame extraction"))
            progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Video path: {video_path.name}"))
            progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Total frames: {total_frames}"))
            progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Duration: {duration:.1f}s"))
            progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Batch size: {batch_size}"))
        
        logger.info(f"DEBUG: Starting batch frame extraction")
        logger.info(f"DEBUG: Video path: {video_path}")
        logger.info(f"DEBUG: Total frames: {total_frames}")
        logger.info(f"DEBUG: Duration: {duration}")
        logger.info(f"DEBUG: Batch size: {batch_size}")
        
        ffmpeg_path = self.resource_manager.get_binary_path('ffmpeg')
        if progress_dialog:
            progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: FFmpeg path for batches: {ffmpeg_path}"))
        logger.info(f"DEBUG: FFmpeg path for batches: {ffmpeg_path}")
        all_frame_paths = []
        
        # Calculate batch parameters based on accurate frame count
        total_batches = (total_frames + batch_size - 1) // batch_size
        
        # Calculate time per batch based on actual frame distribution
        # More accurate than simple duration/batches division
        frames_per_second = total_frames / duration if duration > 0 else 30
        seconds_per_batch = batch_size / frames_per_second if frames_per_second > 0 else duration / total_batches
        
        if progress_dialog:
            progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Calculated {total_batches} batches, {seconds_per_batch:.2f} seconds per batch"))
            progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Processing {total_frames} frames in {total_batches} batches of ~{batch_size} frames each"))
            progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Frame rate: {frames_per_second:.2f} fps, Time calculation method: frame-based"))
        
        logger.info(f"DEBUG: Calculated {total_batches} batches, {seconds_per_batch:.2f} seconds per batch")
        logger.info(f"DEBUG: Processing {total_frames} frames in {total_batches} batches of ~{batch_size} frames each")
        logger.info(f"DEBUG: Frame rate: {frames_per_second:.2f} fps, Time calculation method: frame-based")
        
        for batch_num in range(total_batches):
            if progress_dialog:
                progress_dialog.window.after(0, lambda b=batch_num: progress_dialog.add_log_message(f"DEBUG: Starting batch {b + 1}/{total_batches}"))
            logger.info(f"DEBUG: Starting batch {batch_num + 1}/{total_batches}")
            
            # Check for cancellation
            if progress_dialog and progress_dialog.cancelled:
                if progress_dialog:
                    progress_dialog.window.after(0, lambda b=batch_num: progress_dialog.add_log_message(f"DEBUG: Processing cancelled by user at batch {b + 1}"))
                logger.info(f"DEBUG: Processing cancelled by user at batch {batch_num + 1}")
                raise KeyboardInterrupt("Processing cancelled by user")
            
            # Calculate time range based on frame positions for accuracy
            start_frame = batch_num * batch_size
            end_frame = min(start_frame + batch_size, total_frames)
            
            # Convert frame positions to time (more accurate than time-based calculation)
            start_time = (start_frame / total_frames) * duration if total_frames > 0 else batch_num * seconds_per_batch
            end_time = (end_frame / total_frames) * duration if total_frames > 0 else min((batch_num + 1) * seconds_per_batch, duration)
            
            if progress_dialog:
                progress_dialog.window.after(0, lambda b=batch_num, s=start_time, e=end_time, sf=start_frame, ef=end_frame: progress_dialog.add_log_message(f"DEBUG: Batch {b + 1} - Frames {sf}-{ef}, Time: {s:.2f}s - {e:.2f}s"))
            logger.info(f"DEBUG: Batch {batch_num + 1} - Frames {start_frame}-{end_frame}, Time range: {start_time:.2f}s - {end_time:.2f}s")
            
            # Create batch-specific output pattern
            batch_dir = self.frame_dir / f"batch_{batch_num:03d}"
            logger.info(f"DEBUG: Creating batch directory: {batch_dir}")
            
            try:
                # Handle directory creation with retries for Windows
                if not batch_dir.exists():
                    batch_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"DEBUG: Batch directory created: {batch_dir}")
                elif batch_dir.is_dir():
                    logger.info(f"DEBUG: Batch directory already exists: {batch_dir}")
                else:
                    # If path exists but is not a directory, remove and recreate
                    logger.warning(f"DEBUG: Path exists but is not directory, removing: {batch_dir}")
                    if batch_dir.exists():
                        batch_dir.unlink()
                    batch_dir.mkdir(parents=True, exist_ok=True)
                    
                logger.info(f"DEBUG: Batch directory ready: {batch_dir.exists()}")
            except OSError as e:
                if e.winerror == 183:  # File already exists
                    logger.warning(f"DEBUG: Directory creation race condition detected, continuing: {batch_dir}")
                    # Check if directory actually exists and is usable
                    if batch_dir.exists() and batch_dir.is_dir():
                        logger.info(f"DEBUG: Directory is usable despite error: {batch_dir}")
                    else:
                        logger.error(f"DEBUG: Directory creation failed and path is not usable: {batch_dir}")
                        raise
                else:
                    logger.error(f"DEBUG: Failed to create batch directory: {e}")
                    raise
            except Exception as e:
                logger.error(f"DEBUG: Unexpected error creating batch directory: {e}")
                raise
            
            output_pattern = str(batch_dir / "frame_%06d.png")
            logger.info(f"DEBUG: Output pattern: {output_pattern}")
            
            logger.info(f"DEBUG: Processing batch {batch_num + 1}/{total_batches} (time: {start_time:.1f}s - {end_time:.1f}s)")
            
            cmd = [
                ffmpeg_path,
                '-i', str(video_path),
                '-ss', str(start_time),  # Start time
                '-t', str(end_time - start_time),  # Duration
                '-pix_fmt', 'rgb24',  # Optimal pixel format for waifu2x
                '-compression_level', '9',  # Maximum PNG compression for space efficiency
                '-pred', 'mixed',  # PNG prediction method for better compression
                '-y',
                output_pattern
            ]
            
            logger.info(f"DEBUG: FFmpeg command for batch {batch_num + 1}: {' '.join(cmd)}")
            logger.info(f"DEBUG: Command length: {len(cmd)} arguments")
            
            # Hide console window on Windows
            startupinfo = None
            if os.name == 'nt':  # Windows
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            
            # Execute batch with timeout
            try:
                logger.info(f"DEBUG: Starting FFmpeg subprocess for batch {batch_num + 1}")
                logger.info(f"DEBUG: Command: {' '.join(cmd[:5])} ... {' '.join(cmd[-3:])}")
                logger.info(f"DEBUG: Working directory: {self.frame_dir}")
                logger.info(f"DEBUG: Batch output directory: {batch_dir}")
                logger.info(f"DEBUG: Expected output pattern: {output_pattern}")
                
                result = subprocess.run(cmd, capture_output=True, text=True, 
                                      startupinfo=startupinfo)
                
                logger.info(f"DEBUG: FFmpeg batch {batch_num + 1} completed with return code: {result.returncode}")
                logger.info(f"DEBUG: FFmpeg stdout length: {len(result.stdout)} chars")
                logger.info(f"DEBUG: FFmpeg stderr length: {len(result.stderr)} chars")
                
                if result.returncode != 0:
                    logger.error(f"DEBUG: FFmpeg batch {batch_num + 1} failed!")
                    logger.error(f"DEBUG: FFmpeg stderr: {result.stderr[:500]}")
                else:
                    logger.info(f"DEBUG: FFmpeg batch {batch_num + 1} succeeded")
                    
                # Check if output files were created
                created_files = list(batch_dir.glob("frame_*.png")) if batch_dir.exists() else []
                logger.info(f"DEBUG: Files created in batch {batch_num + 1}: {len(created_files)}")
                if created_files:
                    logger.info(f"DEBUG: First few files: {[f.name for f in created_files[:3]]}")
                    logger.info(f"DEBUG: Last few files: {[f.name for f in created_files[-3:]]}")
                
                if result.returncode != 0:
                    logger.error(f"DEBUG: Batch {batch_num + 1} failed with return code {result.returncode}")
                    logger.error(f"DEBUG: FFmpeg stderr: {result.stderr}")
                    logger.error(f"DEBUG: FFmpeg stdout: {result.stdout}")
                    raise RuntimeError(f"Frame extraction batch {batch_num + 1} failed: {result.stderr}")
                
                # Collect frames from this batch
                logger.info(f"DEBUG: Collecting frames from batch {batch_num + 1}")
                batch_frames = sorted(batch_dir.glob("frame_*.png"))
                logger.info(f"DEBUG: Found {len(batch_frames)} frames in batch {batch_num + 1}")
                
                # Rename frames to maintain sequential numbering based on actual frame positions
                logger.info(f"DEBUG: Starting frame renaming for batch {batch_num + 1}")
                logger.info(f"DEBUG: Batch directory contents before renaming: {[f.name for f in batch_frames[:5]]}")
                logger.info(f"DEBUG: Total frames found in batch: {len(batch_frames)}")
                expected_frames_in_batch = min(batch_size, total_frames - batch_num * batch_size)
                logger.info(f"DEBUG: Expected frames in this batch: {expected_frames_in_batch}")
                
                for i, frame_file in enumerate(batch_frames):
                    frame_number = batch_num * batch_size + i + 1
                    new_name = self.frame_dir / f"frame_{frame_number:06d}.png"
                    logger.debug(f"DEBUG: Moving {frame_file} to {new_name}")
                    
                    # Safe file move with error handling
                    try:
                        # Check if destination already exists
                        if new_name.exists():
                            logger.warning(f"DEBUG: Destination file already exists, removing: {new_name}")
                            new_name.unlink()
                        
                        # Move file with retry mechanism
                        frame_file.rename(new_name)
                        all_frame_paths.append(str(new_name))
                        
                    except OSError as e:
                        if e.winerror == 183:  # File already exists
                            logger.warning(f"DEBUG: File move race condition detected for {frame_file}")
                            # Try alternative approach: copy then delete
                            try:
                                import shutil
                                if new_name.exists():
                                    new_name.unlink()
                                shutil.copy2(frame_file, new_name)
                                frame_file.unlink()
                                all_frame_paths.append(str(new_name))
                                logger.info(f"DEBUG: Successfully moved file using copy method: {new_name}")
                            except Exception as e2:
                                logger.error(f"DEBUG: Failed to move file {frame_file}: {e2}")
                                raise
                        else:
                            logger.error(f"DEBUG: File move failed: {e}")
                            raise
                    except Exception as e:
                        logger.error(f"DEBUG: Unexpected error moving file {frame_file}: {e}")
                        raise
                
                # Verify frame count matches expectation
                if len(batch_frames) != expected_frames_in_batch and batch_num == total_batches - 1:
                    # Last batch may have fewer frames, this is normal
                    logger.info(f"DEBUG: Last batch has {len(batch_frames)} frames (expected up to {expected_frames_in_batch})")
                elif len(batch_frames) != expected_frames_in_batch:
                    logger.warning(f"DEBUG: Batch {batch_num + 1} frame count mismatch: got {len(batch_frames)}, expected {expected_frames_in_batch}")
                
                # Clean up batch directory
                logger.info(f"DEBUG: Cleaning up batch directory {batch_dir}")
                logger.info(f"DEBUG: Batch directory exists: {batch_dir.exists()}")
                logger.info(f"DEBUG: Batch directory is empty: {len(list(batch_dir.iterdir())) == 0 if batch_dir.exists() else 'N/A'}")
                
                try:
                    if batch_dir.exists():
                        # List remaining files before cleanup
                        remaining_files = list(batch_dir.iterdir())
                        if remaining_files:
                            logger.warning(f"DEBUG: Batch directory still contains {len(remaining_files)} files: {[f.name for f in remaining_files[:5]]}")
                            # Try to remove remaining files
                            for file_path in remaining_files:
                                try:
                                    if file_path.is_file():
                                        file_path.unlink()
                                        logger.debug(f"DEBUG: Removed remaining file: {file_path.name}")
                                except Exception as cleanup_e:
                                    logger.warning(f"DEBUG: Failed to remove file {file_path.name}: {cleanup_e}")
                        
                        batch_dir.rmdir()
                        logger.info(f"DEBUG: Batch directory removed successfully")
                    else:
                        logger.info(f"DEBUG: Batch directory already removed")
                except Exception as e:
                    logger.error(f"DEBUG: Failed to remove batch directory {batch_dir}: {e}")
                    logger.error(f"DEBUG: Error type: {type(e).__name__}")
                    import traceback
                    logger.error(f"DEBUG: Traceback: {traceback.format_exc()}")
                
                # Update progress
                if progress_callback:
                    progress = 20 + (70 * (batch_num + 1) / total_batches)
                    progress_callback(progress, f"Batch {batch_num + 1}/{total_batches} complete - {len(all_frame_paths)} frames extracted")
                
                # Memory cleanup between batches
                logger.info(f"DEBUG: Performing memory cleanup after batch {batch_num + 1}")
                import gc
                gc.collect()
                
                logger.info(f"DEBUG: Batch {batch_num + 1} complete: {len(batch_frames)} frames extracted, total: {len(all_frame_paths)}/{total_frames}")
                
                # Progress verification
                expected_total_so_far = min((batch_num + 1) * batch_size, total_frames)
                if len(all_frame_paths) != expected_total_so_far:
                    logger.warning(f"DEBUG: Frame count discrepancy: extracted {len(all_frame_paths)}, expected {expected_total_so_far}")
                
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Batch {batch_num + 1} timed out after 15 minutes")
        
        logger.info(f"All batches complete: {len(all_frame_paths)} total frames extracted")
        
        if progress_callback:
            progress_callback(100, f"Extracted {len(all_frame_paths)} frames in {total_batches} batches")
        
        return all_frame_paths
    
    def combine_frames_to_video(self, frame_paths: List[str], output_path: str, 
                               original_video_path: str, fps: float = 30.0,
                               progress_callback: Optional[Callable] = None) -> bool:
        """Combine frames back into video using accurate frame rate"""
        
        if not frame_paths:
            logger.error("No frames to combine")
            return False
        
        try:
            ffmpeg_path = self.resource_manager.get_binary_path('ffmpeg')
            if not ffmpeg_path:
                raise RuntimeError("FFmpeg not available")
            
            # === DEBUG: フレーム結合開始時の詳細情報 ===
            logger.info(f"=== FRAME COMBINATION DEBUG START ===")
            logger.info(f"Input frame paths count: {len(frame_paths)}")
            logger.info(f"First frame path: {frame_paths[0] if frame_paths else 'None'}")
            logger.info(f"Last frame path: {frame_paths[-1] if frame_paths else 'None'}")
            logger.info(f"Output video path: {output_path}")
            logger.info(f"Original video path: {original_video_path}")
            
            # フレームファイルの実際の存在確認
            existing_frames = []
            for i, frame_path in enumerate(frame_paths):
                if Path(frame_path).exists():
                    existing_frames.append(frame_path)
                else:
                    logger.warning(f"DEBUG: Frame {i+1} does not exist: {frame_path}")
            
            logger.info(f"DEBUG: Total existing frames: {len(existing_frames)}/{len(frame_paths)}")
            if len(existing_frames) != len(frame_paths):
                logger.warning(f"WARNING: {len(frame_paths) - len(existing_frames)} frames are missing!")
            
            # Get accurate frame rate from original video
            logger.info(f"DEBUG: Getting accurate frame rate from original video: {Path(original_video_path).name}")
            video_info = self.validate_video(original_video_path)
            if video_info['valid']:
                accurate_fps = video_info['info']['frame_rate']
                logger.info(f"DEBUG: Using accurate frame rate: {accurate_fps} fps (instead of default {fps})")
                fps = accurate_fps
            else:
                logger.warning(f"DEBUG: Could not get accurate frame rate, using default: {fps} fps")
            
            # Create temporary frame list
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for frame_path in frame_paths:
                    f.write(f"file '{frame_path}'\n")
                    f.write(f"duration {1/fps}\n")
                frame_list_path = f.name
            
            try:
                # FFmpeg command to combine frames - optimized for high quality
                # Based on xyle-official recommendations for preserving waifu2x quality
                cmd = [
                    ffmpeg_path,
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', frame_list_path,
                    '-i', original_video_path,  # For audio track
                    '-c:v', 'libx264',
                    '-preset', 'medium',  # Balance speed and compression
                    '-crf', '18',  # High quality (lower = better quality)
                    '-c:a', 'copy',  # Copy audio from original
                    '-map', '0:v:0',
                    '-map', '1:a:0?',  # Audio is optional
                    '-r', str(fps),
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',  # Optimize for streaming
                    '-y',
                    output_path
                ]
                
                logger.info(f"Combining {len(frame_paths)} frames into video at {fps} fps...")
                logger.info(f"DEBUG: Frame count verification: {len(frame_paths)} frames to combine")
                logger.info(f"DEBUG: Output path: {output_path}")
                logger.info(f"DEBUG: Frame rate: {fps} fps")
                
                # Hide console window on Windows
                startupinfo = None
                if os.name == 'nt':  # Windows
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = subprocess.SW_HIDE
                
                result = subprocess.run(cmd, capture_output=True, text=True, startupinfo=startupinfo)
                
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
    
    def cleanup(self, preserve_frames=False):
        """Clean up video processor resources with enhanced memory management
        
        Args:
            preserve_frames: If True, preserve frame files for resume functionality
        """
        try:
            # Kill any running ffmpeg processes
            self._kill_ffmpeg_processes()
            
            # Force garbage collection before cleanup
            import gc
            gc.collect()
            
            if preserve_frames:
                logger.info("DEBUG: Cleanup with frame preservation for resume functionality")
                
                # Only clean up batch directories and temp files, not extracted frames
                if self.frame_dir.exists():
                    # Clean up batch directories if they exist
                    for batch_dir in self.frame_dir.glob("batch_*"):
                        if batch_dir.is_dir():
                            self._cleanup_directory_contents(batch_dir, "*")
                            try:
                                batch_dir.rmdir()
                            except:
                                pass
                    
                    # Clean up upscaled directory if it exists
                    upscaled_dir = self.temp_dir / "upscaled"
                    if upscaled_dir.exists():
                        self._cleanup_directory_contents(upscaled_dir, "*.png")
                        try:
                            upscaled_dir.rmdir()
                        except:
                            pass
                
                logger.info("DEBUG: Preserved frames for resume, cleaned up temp files only")
                
            else:
                logger.info("DEBUG: Full cleanup - removing all temp files including frames")
                
                # Full cleanup - remove everything
                if self.frame_dir.exists():
                    self._cleanup_directory_contents(self.frame_dir, "*.png")
                    self._cleanup_directory_contents(self.frame_dir, "*.jpg")
                    
                    # Clean up batch directories if they exist
                    for batch_dir in self.frame_dir.glob("batch_*"):
                        if batch_dir.is_dir():
                            self._cleanup_directory_contents(batch_dir, "*")
                            try:
                                batch_dir.rmdir()
                            except:
                                pass
                    
                    # Also clean up upscaled directory if it exists
                    upscaled_dir = self.temp_dir / "upscaled"
                    if upscaled_dir.exists():
                        self._cleanup_directory_contents(upscaled_dir, "*.png")
                        try:
                            upscaled_dir.rmdir()
                        except:
                            pass
                    
                    # Clean up entire temp directory
                    try:
                        import shutil
                        shutil.rmtree(self.temp_dir, ignore_errors=True)
                        logger.info(f"Cleaned up temp directory: {self.temp_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to remove temp directory: {e}")
                
            # Final garbage collection
            gc.collect()
            
            logger.info("Cleaned up video processor resources")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
    
    def _cleanup_directory_contents(self, directory: Path, pattern: str, max_files_per_batch: int = 100):
        """Clean up directory contents in small batches to prevent memory issues"""
        try:
            import gc
            
            files = list(directory.glob(pattern))
            total_files = len(files)
            
            if total_files == 0:
                return
                
            logger.info(f"Cleaning up {total_files} files from {directory}")
            
            # Process files in small batches
            for i in range(0, total_files, max_files_per_batch):
                batch_files = files[i:i + max_files_per_batch]
                
                for file_path in batch_files:
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                    except:
                        pass  # Continue even if some files can't be deleted
                
                # Force garbage collection between batches
                gc.collect()
                
                # Small delay to prevent system overload
                import time
                time.sleep(0.1)
            
            logger.info(f"Cleaned up {total_files} files from {directory}")
            
        except Exception as e:
            logger.warning(f"Failed to clean directory {directory}: {e}")
    
    def cleanup_memory_between_operations(self):
        """Force memory cleanup between major operations"""
        try:
            import gc
            import psutil
            import os
            
            logger.info("DEBUG: Starting memory cleanup between operations")
            
            # Force garbage collection
            gc.collect()
            logger.info("DEBUG: Garbage collection completed")
            
            # Log memory usage for debugging
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            virtual_mb = memory_info.vms / 1024 / 1024
            
            logger.info(f"DEBUG: Memory usage after cleanup - RSS: {memory_mb:.1f} MB, Virtual: {virtual_mb:.1f} MB")
            
            # Log system memory info
            system_memory = psutil.virtual_memory()
            logger.info(f"DEBUG: System memory - Total: {system_memory.total/1024/1024/1024:.1f} GB, Available: {system_memory.available/1024/1024/1024:.1f} GB, Used: {system_memory.percent}%")
            
            # Clean up temporary files if memory usage is high
            if memory_mb > 2000:  # If using more than 2GB
                logger.warning(f"DEBUG: High memory usage detected ({memory_mb:.1f} MB), performing aggressive cleanup")
                self._cleanup_temp_files_aggressive()
                
                # Re-check memory after aggressive cleanup
                gc.collect()
                memory_info = process.memory_info()
                memory_mb_after = memory_info.rss / 1024 / 1024
                logger.info(f"DEBUG: Memory usage after aggressive cleanup: {memory_mb_after:.1f} MB (reduced by {memory_mb - memory_mb_after:.1f} MB)")
                
        except Exception as e:
            logger.warning(f"DEBUG: Memory cleanup warning: {e}")
            import traceback
            logger.warning(f"DEBUG: Memory cleanup traceback: {traceback.format_exc()}")
    
    def _cleanup_temp_files_aggressive(self):
        """Aggressively clean up temporary files when memory is high"""
        try:
            # Clean up any temporary batch directories
            for batch_dir in self.frame_dir.glob("batch_*"):
                if batch_dir.is_dir():
                    import shutil
                    shutil.rmtree(batch_dir, ignore_errors=True)
            
            # === FIXED: Do NOT delete frame files during processing ===
            # The original code was deleting frame files needed for AI upscaling
            # Only clean up temporary batch files and other temp data
            logger.info("DEBUG: Aggressive cleanup - preserving frame files for processing")
            
            # Clean up any temporary working directories
            temp_dirs = ["temp_work", "batch_temp", "conversion_temp"]
            for temp_name in temp_dirs:
                temp_path = self.frame_dir / temp_name
                if temp_path.exists() and temp_path.is_dir():
                    import shutil
                    shutil.rmtree(temp_path, ignore_errors=True)
                    logger.info(f"DEBUG: Cleaned up temporary directory: {temp_name}")
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Aggressive cleanup warning: {e}")
    
    def _kill_ffmpeg_processes(self):
        """Kill any running ffmpeg processes"""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] and 'ffmpeg' in proc.info['name'].lower():
                    try:
                        proc.kill()
                        logger.info(f"Killed ffmpeg process: {proc.info['pid']}")
                    except:
                        pass
        except:
            pass
    
    def _validate_video_with_opencv(self, video_path: Path) -> Dict[str, Any]:
        """Validate video using OpenCV as fallback when ffprobe is not available"""
        try:
            import cv2
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {
                    'valid': False,
                    'error': 'Cannot open video file with OpenCV',
                    'info': None
                }
            
            # Get basic video info
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            # Create basic video info structure compatible with ffprobe format
            video_info = {
                'streams': [{
                    'codec_type': 'video',
                    'width': width,
                    'height': height,
                    'r_frame_rate': f"{fps}/1",
                    'nb_frames': str(frame_count)
                }],
                'format': {
                    'duration': str(duration),
                    'size': str(video_path.stat().st_size)
                }
            }
            
            return {
                'valid': True,
                'error': None,
                'info': video_info
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'OpenCV validation failed: {str(e)}',
                'info': None
            }