"""
Video Processing Module for MP4 files
Handles file validation, codec detection, and basic video operations
With GPU-accelerated frame extraction support
"""

import os
import cv2
import ffmpeg
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
from config.settings import VIDEO_SETTINGS

logger = logging.getLogger(__name__)

# GPU加速機能のインポート（利用可能な場合）
try:
    from .fast_frame_extractor import FastFrameExtractor
    from .gpu_frame_extractor import GPUFrameExtractor
    GPU_ACCELERATION_AVAILABLE = True
    logger.info("GPU acceleration modules loaded successfully")
except ImportError as e:
    GPU_ACCELERATION_AVAILABLE = False
    FastFrameExtractor = None
    GPUFrameExtractor = None
    logger.warning(f"GPU acceleration not available: {e}")


class VideoProcessor:
    """Handles MP4 video file processing operations with GPU acceleration support"""
    
    def __init__(self, resource_manager=None, temp_dir=None, gpu_info=None):
        self.supported_formats = VIDEO_SETTINGS["supported_formats"]
        self.supported_codecs = VIDEO_SETTINGS["supported_codecs"]
        self.max_file_size = VIDEO_SETTINGS["max_file_size_gb"] * 1024 * 1024 * 1024
        self.max_duration = VIDEO_SETTINGS["max_duration_minutes"] * 60
        self.max_resolution = VIDEO_SETTINGS["max_resolution"]
        
        # GPU加速サポートの初期化
        self.resource_manager = resource_manager
        self.temp_dir = temp_dir
        self.gpu_info = gpu_info or {}
        self.fast_extractor = None
        
        # GPU加速フレーム抽出の初期化（利用可能な場合）
        if GPU_ACCELERATION_AVAILABLE and resource_manager and temp_dir:
            try:
                self.fast_extractor = FastFrameExtractor(resource_manager, temp_dir, gpu_info)
                logger.info("GPU-accelerated frame extraction initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU acceleration: {e}")
                self.fast_extractor = None
        else:
            logger.info("Using standard frame extraction (no GPU acceleration)")
    
    def validate_video_file(self, file_path: str) -> Dict[str, any]:
        """
        Validate MP4 video file and return video information
        
        Args:
            file_path (str): Path to the video file
            
        Returns:
            Dict containing validation results and video info
        """
        result = {
            "valid": False,
            "error": None,
            "info": {}
        }
        
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                result["error"] = f"File does not exist: {file_path}"
                return result
            
            # Check file extension
            if file_path.suffix.lower() not in self.supported_formats:
                result["error"] = f"Unsupported format: {file_path.suffix}"
                return result
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                result["error"] = f"File too large: {file_size / (1024**3):.2f}GB > {self.max_file_size / (1024**3)}GB"
                return result
            
            # Get video information using ffprobe
            video_info = self._get_video_info(str(file_path))
            if not video_info:
                result["error"] = "Failed to read video information"
                return result
            
            # Validate duration
            duration = video_info.get("duration", 0)
            if duration > self.max_duration:
                result["error"] = f"Video too long: {duration/60:.2f}min > {self.max_duration/60}min"
                return result
            
            # Validate resolution
            width = video_info.get("width", 0)
            height = video_info.get("height", 0)
            if width > self.max_resolution[0] or height > self.max_resolution[1]:
                result["error"] = f"Resolution too high: {width}x{height} > {self.max_resolution[0]}x{self.max_resolution[1]}"
                return result
            
            # Validate codec (skip if unknown from OpenCV)
            codec = video_info.get("codec_name", "").lower()
            if codec != "unknown" and codec not in self.supported_codecs:
                result["error"] = f"Unsupported codec: {codec}"
                return result
            
            result["valid"] = True
            result["info"] = video_info
            
        except Exception as e:
            result["error"] = f"Validation error: {str(e)}"
            logger.error(f"Video validation failed: {e}")
        
        return result
    
    def _get_video_info(self, file_path: str) -> Optional[Dict]:
        """
        Extract video information using OpenCV with ffprobe fallback
        
        Args:
            file_path (str): Path to video file
            
        Returns:
            Dict containing video information or None if failed
        """
        try:
            # Try ffprobe first
            try:
                probe = ffmpeg.probe(file_path)
                video_stream = next((stream for stream in probe['streams'] 
                                   if stream['codec_type'] == 'video'), None)
                
                if video_stream:
                    info = {
                        "filename": os.path.basename(file_path),
                        "duration": float(probe['format']['duration']),
                        "size": int(probe['format']['size']),
                        "width": int(video_stream['width']),
                        "height": int(video_stream['height']),
                        "codec_name": video_stream['codec_name'],
                        "bit_rate": int(probe['format'].get('bit_rate', 0)),
                        "frame_rate": eval(video_stream.get('r_frame_rate', '0/1')),
                        "frame_count": int(video_stream.get('nb_frames', 0))
                    }
                    
                    # Calculate frame count if not available
                    if info["frame_count"] == 0:
                        info["frame_count"] = int(info["duration"] * info["frame_rate"])
                    
                    return info
            except Exception as ffmpeg_error:
                logger.warning(f"FFprobe failed, trying OpenCV: {ffmpeg_error}")
            
            # Fallback to OpenCV
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return None
            
            # Get basic information from OpenCV
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate duration
            duration = frame_count / fps if fps > 0 else 0
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            cap.release()
            
            info = {
                "filename": os.path.basename(file_path),
                "duration": duration,
                "size": file_size,
                "width": width,
                "height": height,
                "codec_name": "unknown",  # OpenCV doesn't provide codec info easily
                "bit_rate": 0,  # Not available from OpenCV
                "frame_rate": fps,
                "frame_count": frame_count
            }
            
            logger.info(f"Using OpenCV for video info: {width}x{height}, {frame_count} frames, {fps:.2f} FPS")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return None
    
    def get_upscaled_resolution(self, width: int, height: int, 
                               scale_factor: float = None) -> Tuple[int, int]:
        """
        Calculate upscaled resolution
        
        Args:
            width (int): Original width
            height (int): Original height
            scale_factor (float): Upscaling factor (default from settings)
            
        Returns:
            Tuple of (new_width, new_height)
        """
        if scale_factor is None:
            scale_factor = VIDEO_SETTINGS["default_upscale_factor"]
        
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Ensure even dimensions (required for some codecs)
        new_width = new_width + (new_width % 2)
        new_height = new_height + (new_height % 2)
        
        return new_width, new_height
    
    def estimate_processing_time(self, video_info: Dict) -> Dict[str, float]:
        """
        Estimate processing time based on video properties
        
        Args:
            video_info (Dict): Video information from validate_video_file
            
        Returns:
            Dict with time estimates in seconds
        """
        frame_count = video_info.get("frame_count", 0)
        resolution = video_info.get("width", 720) * video_info.get("height", 480)
        
        # Rough estimates (will vary greatly based on hardware)
        seconds_per_frame_cpu = 30  # Very rough estimate for CPU processing
        seconds_per_frame_gpu = 5   # GPU is much faster
        
        # Adjust based on resolution
        resolution_factor = resolution / (720 * 480)  # 480p as baseline
        
        cpu_time = frame_count * seconds_per_frame_cpu * resolution_factor
        gpu_time = frame_count * seconds_per_frame_gpu * resolution_factor
        
        return {
            "estimated_cpu_seconds": cpu_time,
            "estimated_gpu_seconds": gpu_time,
            "estimated_cpu_minutes": cpu_time / 60,
            "estimated_gpu_minutes": gpu_time / 60
        }
    
    def extract_frames_accelerated(self, video_path: str, progress_callback=None, 
                                 progress_dialog=None) -> List[str]:
        """
        GPU加速対応の高速フレーム抽出
        
        Args:
            video_path (str): 動画ファイルパス
            progress_callback: 進捗コールバック関数
            progress_dialog: 進捗ダイアログ
            
        Returns:
            List[str]: 抽出されたフレームファイルパス
        """
        if self.fast_extractor:
            try:
                # 動画情報を取得
                video_info = self._get_video_info(video_path)
                if not video_info:
                    logger.error("Failed to get video information for GPU extraction")
                    return self._fallback_frame_extraction(video_path)
                
                total_frames = video_info.get("frame_count", 0)
                duration = video_info.get("duration", 0)
                
                logger.info(f"Starting GPU-accelerated frame extraction: {total_frames} frames")
                
                # GPU加速フレーム抽出を実行
                frame_paths = self.fast_extractor.extract_frames_parallel(
                    Path(video_path), total_frames, duration, 
                    progress_callback, progress_dialog
                )
                
                if frame_paths:
                    logger.info(f"GPU extraction successful: {len(frame_paths)} frames")
                    return frame_paths
                else:
                    logger.warning("GPU extraction returned no frames, using fallback")
                    return self._fallback_frame_extraction(video_path)
                    
            except Exception as e:
                logger.error(f"GPU extraction failed: {e}")
                logger.info("Falling back to standard extraction")
                return self._fallback_frame_extraction(video_path)
        else:
            logger.info("GPU acceleration not available, using standard extraction")
            return self._fallback_frame_extraction(video_path)
    
    def _fallback_frame_extraction(self, video_path: str) -> List[str]:
        """標準フレーム抽出へのフォールバック"""
        if self.temp_dir:
            extractor = VideoFrameExtractor(str(self.temp_dir))
            return extractor.extract_frames(video_path)
        else:
            # temp_dirが指定されていない場合のデフォルト
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                extractor = VideoFrameExtractor(temp_dir)
                return extractor.extract_frames(video_path)
    
    def get_gpu_acceleration_info(self) -> Dict[str, any]:
        """GPU加速情報を取得"""
        if self.fast_extractor and hasattr(self.fast_extractor, 'gpu_extractor'):
            gpu_extractor = self.fast_extractor.gpu_extractor
            if gpu_extractor:
                return gpu_extractor.get_gpu_info()
        
        return {
            'available': False,
            'method': None,
            'estimated_speedup': 'N/A'
        }


class VideoFrameExtractor:
    """Handles frame extraction from video files"""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_frames(self, video_path: str, output_pattern: str = None) -> List[str]:
        """
        Extract all frames from video
        
        Args:
            video_path (str): Path to video file
            output_pattern (str): Pattern for output files (optional)
            
        Returns:
            List of extracted frame file paths
        """
        if output_pattern is None:
            video_name = Path(video_path).stem
            output_pattern = str(self.temp_dir / f"{video_name}_frame_%06d.png")
        
        try:
            # Try ffmpeg first
            try:
                logger.info("Using FFmpeg for frame extraction...")
                (
                    ffmpeg
                    .input(video_path)
                    .output(output_pattern, format='image2', vcodec='png')
                    .overwrite_output()
                    .run(quiet=True)
                )
                
                # Get list of extracted frames
                frame_files = sorted(self.temp_dir.glob(f"{Path(video_path).stem}_frame_*.png"))
                logger.info(f"FFmpeg successfully extracted {len(frame_files)} frames")
                return [str(f) for f in frame_files]
                
            except Exception as ffmpeg_error:
                logger.warning(f"FFmpeg frame extraction failed, trying OpenCV: {ffmpeg_error}")
            
            # Fallback to OpenCV
            logger.info("Falling back to OpenCV for frame extraction...")
            return self._extract_frames_opencv(video_path)
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return []
    
    def _extract_frames_opencv(self, video_path: str) -> List[str]:
        """
        OpenCVを使用したフレーム抽出
        
        Args:
            video_path (str): ビデオファイルパス
            
        Returns:
            List[str]: 抽出されたフレームファイルのパス
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return []
            
            frame_files = []
            frame_count = 0
            video_name = Path(video_path).stem
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # フレームファイル名を生成
                frame_filename = f"{video_name}_frame_{frame_count:06d}.png"
                frame_path = self.temp_dir / frame_filename
                
                # フレームを保存
                success = cv2.imwrite(str(frame_path), frame)
                if success:
                    frame_files.append(str(frame_path))
                
                frame_count += 1
                
                # 進行状況をログ出力（1000フレームごと）
                if frame_count % 1000 == 0:
                    logger.info(f"Extracted {frame_count} frames...")
            
            cap.release()
            logger.info(f"Extracted {len(frame_files)} frames using OpenCV")
            return frame_files
            
        except Exception as e:
            logger.error(f"OpenCV frame extraction failed: {e}")
            return []
    
    def extract_frame_at_time(self, video_path: str, timestamp: float, 
                             output_path: str) -> bool:
        """
        Extract single frame at specific timestamp
        
        Args:
            video_path (str): Path to video file
            timestamp (float): Time in seconds
            output_path (str): Output file path
            
        Returns:
            bool: Success status
        """
        try:
            (
                ffmpeg
                .input(video_path, ss=timestamp)
                .output(output_path, vframes=1, format='image2', vcodec='png')
                .overwrite_output()
                .run(quiet=True)
            )
            return True
            
        except Exception as e:
            logger.error(f"Frame extraction at {timestamp}s failed: {e}")
            return False