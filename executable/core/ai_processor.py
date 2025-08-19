"""
UpScale App - AI Processing Module  
Consolidated AI upscaling using available backends
"""

import os
import tempfile
import subprocess
import logging
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from PIL import Image
import shutil
import cv2
import numpy as np

from .utils import ProgressCallback, SimpleTimer
from .performance_monitor import OptimizedParallelProcessor, PerformanceMonitor

logger = logging.getLogger(__name__)

try:
    from waifu2x_ncnn_py import Waifu2x
    WAIFU2X_NCNN_PY_AVAILABLE = True
    logger.info("waifu2x_ncnn_py library is available")
except ImportError:
    WAIFU2X_NCNN_PY_AVAILABLE = False
    logger.warning("waifu2x_ncnn_py library not available")

class AIProcessor:
    """Handles AI-based image upscaling using available backends"""
    
    def __init__(self, resource_manager, gpu_info: Dict):
        self.resource_manager = resource_manager
        self.gpu_info = gpu_info
        self.best_backend = gpu_info.get('best_backend', 'cpu')
        
        # Initialize parallel processor settings first
        self.parallel_processor = None
        self.use_parallel_processing = True  # Enable by default for GPU backends
        
        # Initialize backend
        self.backend = None
        self._initialize_backend()
        
        logger.info(f"AIProcessor initialized - Backend: {self.best_backend}")
    
    def _initialize_backend(self):
        """Initialize the best available AI backend"""
        try:
            logger.info(f"Initializing backend for: {self.best_backend}")
            logger.info(f"WAIFU2X_NCNN_PY_AVAILABLE: {WAIFU2X_NCNN_PY_AVAILABLE}")
            
            # Always try Waifu2x first for any GPU backend
            if self.best_backend in ['nvidia', 'nvidia_cuda', 'amd', 'amd_rocm', 'vulkan', 'intel']:
                logger.info("Attempting to initialize Waifu2x backend")
                # Try Python library first, then fallback to executable
                if WAIFU2X_NCNN_PY_AVAILABLE:
                    logger.info("Using waifu2x_ncnn_py library backend")
                    try:
                        self.backend = Waifu2xPythonBackend(self.gpu_info)
                        logger.info(f"Waifu2xPythonBackend created, checking availability...")
                    except Exception as e:
                        logger.error(f"Failed to create Waifu2xPythonBackend: {e}")
                        self.backend = None
                else:
                    logger.info("waifu2x_ncnn_py not available, using waifu2x executable backend")
                    self.backend = Waifu2xExecutableBackend(self.resource_manager, self.gpu_info)
                
                if self.backend and self.backend.is_available():
                    logger.info(f"Waifu2x backend initialized successfully: {type(self.backend).__name__}")
                    # Initialize parallel processor for all backends (GPU優先だが、CPUでも並列処理を有効化)
                    if self.use_parallel_processing:
                        self.parallel_processor = OptimizedParallelProcessor(self)
                        logger.info("Parallel processing enabled for optimized processing")
                    return
                else:
                    logger.warning(f"Waifu2x backend not available (backend={self.backend}, available={self.backend.is_available() if self.backend else 'None'}), falling back to simple upscaling")
            
            # Fallback to simple upscaling
            logger.info("Using simple upscaling backend")
            self.backend = SimpleUpscalingBackend()
            # CPUモードでも並列処理を初期化（効率向上のため）
            if self.use_parallel_processing:
                self.parallel_processor = OptimizedParallelProcessor(self)
                logger.info("Parallel processing enabled for CPU mode")
                
        except Exception as e:
            logger.error(f"Failed to initialize AI backend: {e}")
            import traceback
            traceback.print_exc()
            self.backend = SimpleUpscalingBackend()
    
    def upscale_frames(self, frame_paths: List[str], output_dir: str, 
                      scale_factor: float = 2.0,
                      progress_callback: Optional[Callable] = None, progress_dialog=None) -> List[str]:
        """Upscale a list of frame images with optimized parallel processing"""
        
        if not self.backend:
            raise RuntimeError("No AI backend available")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_frames = len(frame_paths)
        
        # Add GPU debug information and initial status update
        backend_info = self.get_backend_info()
        gpu_mode_active = backend_info.get('gpu_mode', False)
        
        if progress_dialog:
            progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: AI Backend: {type(self.backend).__name__}"))
            progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: GPU Mode: {gpu_mode_active}"))
            progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Best Backend: {self.best_backend}"))
            
            # Update initial GPU status to show it's ready for processing
            if gpu_mode_active:
                progress_dialog.window.after(0, lambda: progress_dialog.update_gpu_status(
                    False, 0, f"GPU Ready: {self.best_backend.upper()} backend initialized"))
            else:
                progress_dialog.window.after(0, lambda: progress_dialog.update_gpu_status(
                    False, 0, "CPU Mode: No GPU acceleration available"))
            if isinstance(self.backend, Waifu2xExecutableBackend):
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Waifu2x GPU ID: {self.backend.gpu_id}"))
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Waifu2x Path: {self.backend.waifu2x_path}"))
        
        logger.info(f"Processing {total_frames} frames with {type(self.backend).__name__}")
        logger.info(f"DEBUG: AI Backend: {type(self.backend).__name__}")
        logger.info(f"DEBUG: GPU Mode: {backend_info.get('gpu_mode', False)}")
        logger.info(f"DEBUG: Best Backend: {self.best_backend}")
        
        # Use parallel processing if available and beneficial (降低門檻以確保使用)
        logger.info(f"DEBUG: Parallel processor available: {self.parallel_processor is not None}")
        logger.info(f"DEBUG: Total frames: {total_frames}, threshold: >3")
        logger.info(f"DEBUG: GPU mode active: {gpu_mode_active}")
        logger.info(f"DEBUG: Backend type: {type(self.backend).__name__}")
        logger.info(f"DEBUG: Parallel processing condition: {self.parallel_processor and total_frames > 3}")
        
        # 並列処理条件を緩和（3フレーム以上で並列処理を使用）
        if self.parallel_processor and total_frames > 3:
            if progress_dialog:
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message("INFO: Using optimized parallel processing"))
                # GPU処理開始を通知
                if gpu_mode_active:
                    progress_dialog.window.after(0, lambda: progress_dialog.update_gpu_status(
                        True, 75, f"Starting GPU parallel processing ({total_frames} frames)"))
                else:
                    progress_dialog.window.after(0, lambda: progress_dialog.update_gpu_status(
                        True, 50, f"Starting CPU parallel processing ({total_frames} frames)"))
            logger.info("Using optimized parallel processing for improved GPU utilization")
            
            processed_frames = self.parallel_processor.process_frames_parallel(
                frame_paths, str(output_dir), scale_factor, progress_callback, progress_dialog
            )
            
            # Log performance report
            report = self.parallel_processor.get_performance_report()
            logger.info(f"Performance Report: {report['throughput_fps']:.2f} fps, "
                       f"success rate: {report['success_rate']:.1%}")
            
            # 並列処理全体の完了を通知（すべてのフレームが処理された場合のみ）
            if progress_dialog and len(processed_frames) >= total_frames:
                progress_dialog.window.after(0, lambda: progress_dialog.update_gpu_status(
                    False, 0, f"All frames completed! ({report['throughput_fps']:.1f} fps average)"))
            elif progress_dialog:
                # 部分完了の場合は継続中表示
                progress_dialog.window.after(0, lambda: progress_dialog.update_gpu_status(
                    True, 80, f"Processing batch completed ({len(processed_frames)}/{total_frames})"))
            
            return processed_frames
        
        # Fallback to sequential processing
        logger.info("INFO: Falling back to sequential processing")
        if self.parallel_processor is None:
            logger.info("REASON: Parallel processor not initialized")
        elif total_frames <= 3:
            logger.info(f"REASON: Too few frames ({total_frames} <= 3)")
        else:
            logger.info("REASON: Unknown condition")
            
        if progress_dialog:
            progress_dialog.window.after(0, lambda: progress_dialog.add_log_message("INFO: Using sequential processing"))
            # GPU処理開始を通知
            if gpu_mode_active:
                progress_dialog.window.after(0, lambda: progress_dialog.update_gpu_status(
                    True, 60, f"Starting GPU sequential processing ({total_frames} frames)"))
            else:
                progress_dialog.window.after(0, lambda: progress_dialog.update_gpu_status(
                    True, 30, f"Starting CPU sequential processing ({total_frames} frames)"))
        logger.info("Using sequential processing")
        
        return self._process_frames_sequential(frame_paths, output_dir, scale_factor, progress_callback, progress_dialog)
    
    def _process_frames_sequential(self, frame_paths: List[str], output_dir: Path, 
                                 scale_factor: float, progress_callback: Optional[Callable], 
                                 progress_dialog) -> List[str]:
        """Sequential frame processing (original method)"""
        processed_frames = []
        total_frames = len(frame_paths)
        
        # Process frames in batches to manage memory usage
        batch_size = 100  # Process 100 frames before memory cleanup
        
        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            batch_frames = frame_paths[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}: frames {batch_start+1} to {batch_end}")
            
            for i, frame_path in enumerate(batch_frames):
                try:
                    global_index = batch_start + i
                    
                    # Generate output path
                    frame_name = Path(frame_path).stem
                    output_path = output_dir / f"{frame_name}_upscaled.png"
                    
                    # Upscale frame with debug info
                    if progress_dialog and global_index < 5:  # Debug first 5 frames
                        progress_dialog.window.after(0, lambda idx=global_index, path=frame_path: progress_dialog.add_log_message(f"DEBUG: Processing frame {idx + 1}: {Path(path).name}"))
                    
                    success = self.backend.upscale_image(frame_path, str(output_path), scale_factor, progress_dialog=progress_dialog if global_index < 3 else None)
                    
                    if success:
                        processed_frames.append(str(output_path))
                        if progress_dialog and global_index < 3:  # Debug first 3 successful frames
                            progress_dialog.window.after(0, lambda idx=global_index: progress_dialog.add_log_message(f"DEBUG: Frame {idx + 1} upscaled successfully"))
                    else:
                        logger.warning(f"Failed to upscale frame: {frame_path}")
                        if progress_dialog and global_index < 3:
                            progress_dialog.window.after(0, lambda idx=global_index: progress_dialog.add_log_message(f"DEBUG: Frame {idx + 1} upscaling failed"))
                    
                    # Update progress
                    if progress_callback:
                        progress = (global_index + 1) / total_frames * 100
                        progress_callback(progress, f"Processed frame {global_index+1}/{total_frames}")
                        
                except Exception as e:
                    logger.error(f"Error processing frame {frame_path}: {e}")
            
            # Memory cleanup between batches
            self._cleanup_memory_between_batches()
        
        logger.info(f"Successfully processed {len(processed_frames)}/{total_frames} frames")
        
        # 逐次処理完了の通知
        if progress_dialog:
            if len(processed_frames) >= total_frames:
                progress_dialog.window.after(0, lambda: progress_dialog.update_gpu_status(
                    False, 0, f"Sequential processing completed! ({len(processed_frames)} frames)"))
            else:
                progress_dialog.window.after(0, lambda: progress_dialog.update_gpu_status(
                    False, 0, f"Processing finished ({len(processed_frames)}/{total_frames} successful)"))
        
        return processed_frames
    
    def _cleanup_memory_between_batches(self):
        """Clean up memory between frame processing batches"""
        try:
            import gc
            import psutil
            import os
            
            # Force garbage collection
            gc.collect()
            
            # Log memory usage
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            logger.info(f"Memory usage between batches: {memory_mb:.1f} MB")
            
            # Kill any stray waifu2x processes that might be consuming memory
            self._kill_waifu2x_processes()
            
            # Small delay to allow system to stabilize
            import time
            time.sleep(0.5)
            
        except Exception as e:
            logger.warning(f"Memory cleanup between batches warning: {e}")
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the active backend"""
        if self.backend:
            return self.backend.get_info()
        return {"backend": "none", "available": False}
    
    def cleanup(self):
        """Clean up AI processor resources"""
        try:
            # Kill any running waifu2x processes
            self._kill_waifu2x_processes()
            
            if self.backend:
                self.backend.cleanup()
        except Exception as e:
            logger.warning(f"AI processor cleanup error: {e}")
    
    def _kill_waifu2x_processes(self):
        """Kill any running waifu2x processes"""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] and 'waifu2x' in proc.info['name'].lower():
                    try:
                        proc.kill()
                        logger.info(f"Killed waifu2x process: {proc.info['pid']}")
                    except:
                        pass
        except:
            pass

class Waifu2xExecutableBackend:
    """Waifu2x backend using executable binary"""
    
    def __init__(self, resource_manager, gpu_info: Dict):
        self.resource_manager = resource_manager
        self.gpu_info = gpu_info
        self.waifu2x_path = resource_manager.get_binary_path('waifu2x')
        self.gpu_id = self._determine_gpu_id()
        
    def _determine_gpu_id(self) -> int:
        """Determine best GPU ID to use"""
        best_backend = self.gpu_info.get('best_backend', 'cpu')
        
        # Force GPU usage for better performance
        # Prefer Radeon RX Vega (GPU 0) over Intel HD Graphics (GPU 1)
        if self.gpu_info.get('amd', {}).get('available'):
            logger.info("Using AMD Radeon RX Vega (GPU 0) for waifu2x")
            return 0  # AMD GPU (Radeon RX Vega)
        elif self.gpu_info.get('nvidia', {}).get('available'):
            logger.info("Using NVIDIA GPU (GPU 0) for waifu2x")
            return 0  # NVIDIA GPU
        else:
            # Even if no GPU info, try GPU 0 (Radeon RX Vega)
            logger.info("Attempting to use GPU 0 (Radeon RX Vega) for waifu2x")
            return 0  # Default to first GPU
        
        # Only fallback to CPU if explicitly needed
        # return -1  # CPU mode
    
    def _get_optimal_tile_size(self) -> int:
        """Get optimal tile size for GPU based on xyle-official recommendations"""
        # Based on https://xyle-official.com/2020/03/23/waifu2x_upscale/
        # Recommended tile size: 400 for optimal GPU utilization
        gpu_name = "unknown"
        try:
            gpu_name = self.gpu_info.get('amd', {}).get('gpus', [{}])[0].get('name', 'unknown').lower()
        except:
            pass
        
        if 'vega' in gpu_name:
            # Radeon RX Vega: Use recommended 400 for better GPU utilization
            logger.info("Using optimized tile size 400 for Radeon RX Vega (xyle-official recommended)")
            return 400
        elif 'rx' in gpu_name or self.gpu_info.get('nvidia', {}).get('available'):
            # Other discrete GPUs: use recommended size
            return 400
        else:
            # Conservative for integrated GPUs
            return 256
    
    def is_available(self) -> bool:
        """Check if Waifu2x executable is available"""
        logger.info(f"Checking Waifu2x availability...")
        logger.info(f"Waifu2x path from resource manager: {self.waifu2x_path}")
        
        if not self.waifu2x_path:
            logger.warning("Waifu2x executable path not found")
            # Try to debug the resource manager
            logger.info("Available binaries in resource manager:")
            availability = self.resource_manager.check_binary_availability()
            for bin_name, available in availability.items():
                logger.info(f"  {bin_name}: {available}")
            return False
        
        try:
            # Test if the executable actually works
            from pathlib import Path
            if not Path(self.waifu2x_path).exists():
                logger.warning(f"Waifu2x executable not found at: {self.waifu2x_path}")
                return False
            
            logger.info(f"Waifu2x executable found at: {self.waifu2x_path}")
            
            # Quick test execution to verify it works
            logger.info("Testing waifu2x executable with -h flag...")
            result = self.resource_manager.run_binary('waifu2x', ['-h'], timeout=10, check=False, hide_window=True)
            if result.returncode == 0 or 'waifu2x' in result.stdout.lower() or 'usage' in result.stdout.lower():
                logger.info("Waifu2x executable is available and working")
                return True
            else:
                logger.warning(f"Waifu2x executable test failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.warning(f"Waifu2x availability check failed: {e}")
            return False
    
    def upscale_image(self, input_path: str, output_path: str, scale_factor: float = 2.0, progress_dialog=None) -> bool:
        """Upscale single image using Waifu2x"""
        try:
            if not self.waifu2x_path:
                return False
            
            # Convert scale factor to integer (Waifu2x supports 1, 2, 4, 8, 16, 32)
            scale_int = max(1, min(32, int(scale_factor)))
            
            # Optimize based on xyle-official recommendations
            # https://xyle-official.com/2020/03/23/waifu2x_upscale/
            tile_size = self._get_optimal_tile_size()
            
            # Get Waifu2x executable directory
            waifu2x_dir = Path(self.waifu2x_path).parent
            
            # Try different model directory structures
            model_candidates = [
                # Method 1: Bundle structure (resources/models/models-cunet)
                (waifu2x_dir.parent.parent / 'models' / 'models-cunet', 'cunet'),
                (waifu2x_dir.parent.parent / 'models' / 'models-upconv_7_anime_style_art_rgb', 'anime'),
                # Method 2: Same directory structure (models-cunet)
                (waifu2x_dir / 'models-cunet', 'cunet_default'),
                (waifu2x_dir / 'models-upconv_7_anime_style_art_rgb', 'anime_default'),
                # Method 3: Alternative bundle paths
                (waifu2x_dir.parent / 'models' / 'models-cunet', 'cunet_alt'),
                (waifu2x_dir.parent / 'models' / 'models-upconv_7_anime_style_art_rgb', 'anime_alt'),
            ]
            
            model_path = None
            model_name = None
            
            for candidate_path, candidate_name in model_candidates:
                if candidate_path.exists():
                    # Verify it contains model files
                    model_files = list(candidate_path.glob("*.param"))
                    if model_files:
                        model_path = candidate_path
                        model_name = candidate_name
                        break
            
            # Final fallback - use relative path and hope for the best
            if model_path is None:
                model_path = Path("models-cunet")
                model_name = 'fallback_cunet'
            
            cmd = [
                self.waifu2x_path,
                '-i', input_path,
                '-o', output_path,
                '-s', str(scale_int),
                '-n', '1',  # Denoising level from article recommendation
                '-g', '0',  # Force GPU 0 (Radeon RX Vega) explicitly
                '-m', str(model_path),
                '-f', 'png',  # Output format
                '-t', '400',  # Default tile size from article (optimal performance)
                '-j', f'2:{min(4, max(2, self.gpu_info.get("amd", {}).get("gpus", [{}])[0].get("memory_mb", 4000) // 1000))}:2',  # Optimized for GPU memory
                '-v'  # Verbose output for debugging
            ]
            
            # Debug waifu2x command for first few frames
            if progress_dialog:
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Using {model_name} model at: {model_path}"))
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Waifu2x command: {' '.join(cmd)}"))
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: GPU ID: {self.gpu_id}, Tile size: 400"))
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Waifu2x executable exists: {Path(self.waifu2x_path).exists()}"))
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Model directory exists: {model_path.exists()}"))
                if model_path.exists():
                    model_files = list(model_path.glob("*.param"))[:3]
                    all_files = list(model_path.glob("*"))[:10]
                    progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Model files found: {[f.name for f in model_files]}"))
                    progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: All files in model dir: {[f.name for f in all_files]}"))
                    progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Model directory path: {model_path}"))
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Input file exists: {Path(input_path).exists()}"))
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Output directory exists: {Path(output_path).parent.exists()}"))
            
            # Hide console window on Windows
            startupinfo = None
            if os.name == 'nt':  # Windows
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            
            # Run waifu2x
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=120,  # 2 minute timeout per image
                startupinfo=startupinfo
            )
            
            # Debug waifu2x output for first few frames
            if progress_dialog:
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Waifu2x return code: {result.returncode}"))
                
                # Decode specific error codes
                if result.returncode == 3221226505:  # 0xC0000409
                    progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Error: Stack overflow/Memory access violation"))
                elif result.returncode == -1073741819:  # 0xC0000005
                    progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Error: Access violation"))
                elif result.returncode != 0:
                    progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Error: Unknown error code {result.returncode} (0x{result.returncode:08X})"))
                
                if result.stderr:
                    progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Waifu2x stderr: {result.stderr[:300]}"))
                if result.stdout:
                    progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Waifu2x stdout: {result.stdout[:200]}"))
                
                # Check if output file was created despite error
                output_exists = Path(output_path).exists()
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Output file created: {output_exists}"))
            
            if result.returncode != 0:
                logger.error(f"Waifu2x GPU failed with return code {result.returncode}")
                logger.error(f"Waifu2x stderr: {result.stderr}")
                logger.error(f"Waifu2x stdout: {result.stdout}")
                logger.error(f"Waifu2x command: {' '.join(cmd)}")
                
                # Try CPU fallback if GPU failed
                if progress_dialog:
                    progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"ERROR: Waifu2x GPU failed with code {result.returncode}"))
                    progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"ERROR: {result.stderr}"))
                    progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Trying CPU fallback..."))
                
                # Modify command for CPU processing
                cpu_cmd = cmd.copy()
                for i, arg in enumerate(cpu_cmd):
                    if arg == '-g':
                        cpu_cmd[i+1] = '-1'  # Use CPU
                        break
                
                try:
                    cpu_result = subprocess.run(
                        cpu_cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=240,  # Longer timeout for CPU
                        startupinfo=startupinfo
                    )
                    
                    if progress_dialog:
                        progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: CPU fallback return code: {cpu_result.returncode}"))
                    
                    if cpu_result.returncode == 0 and Path(output_path).exists():
                        if progress_dialog:
                            progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: CPU fallback successful"))
                        return True
                        
                except subprocess.TimeoutExpired:
                    logger.warning(f"CPU fallback timed out for: {input_path}")
                except Exception as e:
                    logger.debug(f"CPU fallback error: {e}")
                
                # Final fallback: OpenCV upscaling
                if progress_dialog:
                    progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Trying OpenCV fallback..."))
                
                return self._opencv_upscale_fallback(input_path, output_path, scale_factor, progress_dialog)
            
            # Verify output file exists
            return Path(output_path).exists()
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Waifu2x timed out for image: {input_path}")
            return False
        except Exception as e:
            logger.debug(f"Waifu2x processing error: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get backend information"""
        return {
            "backend": "waifu2x_executable",
            "available": self.is_available(),
            "executable_path": self.waifu2x_path,
            "gpu_id": self.gpu_id,
            "gpu_mode": self.gpu_id >= 0
        }
    
    def _opencv_upscale_fallback(self, input_path: str, output_path: str, scale_factor: float = 2.0, progress_dialog=None) -> bool:
        """OpenCV-based upscaling fallback when Waifu2x fails"""
        try:
            if progress_dialog:
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"INFO: Using OpenCV INTER_CUBIC upscaling"))
            
            # Read image
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                logger.error(f"Failed to read image: {input_path}")
                return False
            
            # Calculate new dimensions
            height, width = img.shape[:2]
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Upscale using INTER_CUBIC (better quality than LANCZOS for some cases)
            upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Save result
            success = cv2.imwrite(output_path, upscaled)
            
            if success and progress_dialog:
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"INFO: OpenCV fallback successful"))
            
            return success
            
        except Exception as e:
            logger.error(f"OpenCV fallback failed: {e}")
            if progress_dialog:
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"ERROR: OpenCV fallback failed: {e}"))
            return False

    def cleanup(self):
        """Cleanup backend resources"""
        pass  # Nothing to cleanup for executable backend

class SimpleUpscalingBackend:
    """Fallback simple upscaling using PIL"""
    
    def __init__(self):
        self.method = Image.Resampling.LANCZOS
        
    def is_available(self) -> bool:
        """Simple upscaling is always available"""
        return True
    
    def upscale_image(self, input_path: str, output_path: str, scale_factor: float = 2.0, progress_dialog=None) -> bool:
        """Simple upscaling using PIL"""
        try:
            with Image.open(input_path) as img:
                # Convert RGBA to RGB if necessary
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate new size
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)
                
                # Upscale image
                upscaled = img.resize((new_width, new_height), self.method)
                
                # Save result
                upscaled.save(output_path, 'PNG', quality=95)
                
                return True
                
        except Exception as e:
            logger.error(f"Simple upscaling failed: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get backend information"""
        return {
            "backend": "simple_upscaling",
            "available": True,
            "method": "PIL Lanczos",
            "gpu_mode": False
        }
    
    def cleanup(self):
        """Cleanup backend resources"""
        pass  # Nothing to cleanup for simple backend


class Waifu2xPythonBackend:
    """Waifu2x backend using waifu2x_ncnn_py library"""
    
    def __init__(self, gpu_info: Dict):
        self.gpu_info = gpu_info
        self.gpu_id = self._determine_gpu_id()
        self.waifu2x = None
        self.current_scale = 2.0
        self._processing_lock = threading.Lock()  # GPU処理の同期用
        self._initialize_waifu2x()
        
    def _determine_gpu_id(self) -> int:
        """Determine best GPU ID to use"""
        # Prefer AMD Radeon RX Vega (GPU 0) for Vega 56
        if self.gpu_info.get('amd', {}).get('available'):
            logger.info("Using AMD Radeon RX Vega (GPU 0) for waifu2x_ncnn_py")
            return 0  # AMD GPU (Radeon RX Vega)
        elif self.gpu_info.get('nvidia', {}).get('available'):
            logger.info("Using NVIDIA GPU (GPU 0) for waifu2x_ncnn_py")
            return 0  # NVIDIA GPU
        else:
            logger.info("Attempting to use GPU 0 (Radeon RX Vega) for waifu2x_ncnn_py")
            return 0  # Default to GPU 0
    
    def _initialize_waifu2x(self):
        """Initialize waifu2x instance"""
        try:
            logger.info(f"Initializing waifu2x with GPU ID: {self.gpu_id}")
            # Initialize with optimal settings for Vega 56
            # Based on Vega architecture optimizations
            self.current_scale = 2.0  # Store scale separately
            self.waifu2x = Waifu2x(
                gpuid=self.gpu_id,
                scale=int(self.current_scale),  # Ensure integer scale
                noise=3,  # High quality noise reduction
                tilesize=400,  # Optimal for Vega 56's 8GB memory
                model="models-cunet",  # Best quality model
                tta_mode=False  # Disable for performance
            )
            logger.info(f"SUCCESS: Waifu2x_ncnn_py initialized with GPU {self.gpu_id}, scale={self.current_scale}")
        except Exception as e:
            logger.error(f"FAILED: Failed to initialize waifu2x_ncnn_py: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.waifu2x = None
            self.current_scale = 2.0
    
    def is_available(self) -> bool:
        """Check if Waifu2x backend is available"""
        return self.waifu2x is not None and WAIFU2X_NCNN_PY_AVAILABLE
    
    def upscale_image(self, input_path: str, output_path: str, scale_factor: float = 2.0, progress_dialog=None) -> bool:
        """Upscale image using waifu2x_ncnn_py"""
        if not self.is_available():
            return False
            
        # GPU処理を同期化（並列処理でのリソース競合を防止）
        with self._processing_lock:
            try:
                # GPU処理開始を通知
                if progress_dialog:
                    progress_dialog.window.after(0, lambda: progress_dialog.update_gpu_status(
                        True, 75, f"Processing with waifu2x_ncnn_py (GPU 0)"))
                
                # Check if scale changed - need to reinitialize if different
                if scale_factor != self.current_scale:
                    logger.info(f"Scale changed from {self.current_scale} to {scale_factor}, reinitializing waifu2x...")
                    self.current_scale = int(scale_factor)
                    self.waifu2x = Waifu2x(
                        gpuid=self.gpu_id,
                        scale=int(self.current_scale),  # Ensure integer scale
                        noise=3,
                        tilesize=400,
                        model="models-cunet",
                        tta_mode=False
                    )
                
                # Process image
                with Image.open(input_path) as image:
                    # Convert RGBA to RGB if necessary
                    if image.mode == 'RGBA':
                        background = Image.new('RGB', image.size, (255, 255, 255))
                        background.paste(image, mask=image.split()[-1])
                        image = background
                    elif image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Process with waifu2x
                    upscaled_image = self.waifu2x.process_pil(image)
                    
                    # Save result
                    upscaled_image.save(output_path, 'PNG', quality=95)
                    
                    # GPU処理継続中の通知（個別フレーム完了は全体完了ではない）
                    if progress_dialog:
                        progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"INFO: Waifu2x_ncnn_py frame processing successful"))
                        # まだ処理中なのでアクティブ状態を継続
                        progress_dialog.window.after(0, lambda: progress_dialog.update_gpu_status(
                            True, 60, "Waifu2x frame processed, continuing..."))
                    
                    return True
                    
            except Exception as e:
                logger.error(f"Waifu2x_ncnn_py processing error: {e}")
                if progress_dialog:
                    progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"ERROR: Waifu2x_ncnn_py processing failed: {e}"))
                return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get backend information"""
        return {
            "backend": "waifu2x_ncnn_py",
            "available": self.is_available(),
            "gpu_id": self.gpu_id,
            "gpu_mode": self.gpu_id >= 0,
            "library_version": "waifu2x_ncnn_py 2.0.0+"
        }
    
    def cleanup(self):
        """Cleanup backend resources"""
        if self.waifu2x:
            del self.waifu2x
            self.waifu2x = None