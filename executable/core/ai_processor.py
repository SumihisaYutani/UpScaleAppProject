"""
UpScale App - AI Processing Module  
Consolidated AI upscaling using available backends
"""

import os
import tempfile
import subprocess
import logging
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from PIL import Image
import shutil

from .utils import ProgressCallback, SimpleTimer

logger = logging.getLogger(__name__)

class AIProcessor:
    """Handles AI-based image upscaling using available backends"""
    
    def __init__(self, resource_manager, gpu_info: Dict):
        self.resource_manager = resource_manager
        self.gpu_info = gpu_info
        self.best_backend = gpu_info.get('best_backend', 'cpu')
        
        # Initialize backend
        self.backend = None
        self._initialize_backend()
        
        logger.info(f"AIProcessor initialized - Backend: {self.best_backend}")
    
    def _initialize_backend(self):
        """Initialize the best available AI backend"""
        try:
            logger.info(f"Initializing backend for: {self.best_backend}")
            
            # Always try Waifu2x first for any GPU backend
            if self.best_backend in ['nvidia', 'nvidia_cuda', 'amd', 'amd_rocm', 'vulkan', 'intel']:
                logger.info("Attempting to initialize Waifu2x backend")
                self.backend = Waifu2xExecutableBackend(self.resource_manager, self.gpu_info)
                
                if self.backend.is_available():
                    logger.info(f"Waifu2x backend initialized successfully: {type(self.backend).__name__}")
                    return
                else:
                    logger.warning("Waifu2x backend not available, falling back to simple upscaling")
            
            # Fallback to simple upscaling
            logger.info("Using simple upscaling backend")
            self.backend = SimpleUpscalingBackend()
                
        except Exception as e:
            logger.error(f"Failed to initialize AI backend: {e}")
            import traceback
            traceback.print_exc()
            self.backend = SimpleUpscalingBackend()
    
    def upscale_frames(self, frame_paths: List[str], output_dir: str, 
                      scale_factor: float = 2.0,
                      progress_callback: Optional[Callable] = None, progress_dialog=None) -> List[str]:
        """Upscale a list of frame images with memory management"""
        
        if not self.backend:
            raise RuntimeError("No AI backend available")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_frames = []
        total_frames = len(frame_paths)
        
        # Add GPU debug information
        backend_info = self.get_backend_info()
        if progress_dialog:
            progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: AI Backend: {type(self.backend).__name__}"))
            progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: GPU Mode: {backend_info.get('gpu_mode', False)}"))
            progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Best Backend: {self.best_backend}"))
            if isinstance(self.backend, Waifu2xExecutableBackend):
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Waifu2x GPU ID: {self.backend.gpu_id}"))
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Waifu2x Path: {self.backend.waifu2x_path}"))
        
        logger.info(f"Processing {total_frames} frames with {type(self.backend).__name__}")
        logger.info(f"DEBUG: AI Backend: {type(self.backend).__name__}")
        logger.info(f"DEBUG: GPU Mode: {backend_info.get('gpu_mode', False)}")
        logger.info(f"DEBUG: Best Backend: {self.best_backend}")
        
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
            
            cmd = [
                self.waifu2x_path,
                '-i', input_path,
                '-o', output_path,
                '-s', str(scale_int),
                '-n', '3',  # Strong noise reduction (xyle-official recommended)
                '-g', '0',  # Force GPU 0 (Radeon RX Vega) explicitly
                '-m', str(Path(self.waifu2x_path).parent / 'models-cunet'),  # Full model path
                '-f', 'png',  # Output format
                '-t', str(tile_size),  # Optimized tile size (400 recommended)
                '-j', '1:8:4',  # Increased processing threads for better GPU utilization
                '-v'  # Verbose output for debugging
            ]
            
            # Debug waifu2x command for first few frames
            if progress_dialog:
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Waifu2x command: {' '.join(cmd)}"))
                progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: GPU ID: {self.gpu_id}, Tile size: {tile_size}"))
            
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
                if result.stderr:
                    progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Waifu2x stderr: {result.stderr[:200]}"))
                if result.stdout:
                    progress_dialog.window.after(0, lambda: progress_dialog.add_log_message(f"DEBUG: Waifu2x stdout: {result.stdout[:200]}"))
            
            if result.returncode != 0:
                logger.debug(f"Waifu2x failed: {result.stderr}")
                return False
            
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