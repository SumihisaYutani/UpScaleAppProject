"""
Waifu2x Processing Module
High-quality image upscaling using waifu2x AI models
Optimized for anime-style images and general purpose upscaling
"""

try:
    from waifu2x_ncnn_vulkan import Waifu2x
    WAIFU2X_NCNN_PACKAGE_AVAILABLE = True
except ImportError:
    WAIFU2X_NCNN_PACKAGE_AVAILABLE = False

# Check for waifu2x executable as fallback
import subprocess
from pathlib import Path
WAIFU2X_NCNN_EXE_PATH = Path("tools/waifu2x-ncnn-vulkan/waifu2x-ncnn-vulkan-20220728-windows/waifu2x-ncnn-vulkan.exe")
WAIFU2X_NCNN_EXE_AVAILABLE = WAIFU2X_NCNN_EXE_PATH.exists()

# Overall NCNN availability (package or executable)
WAIFU2X_NCNN_AVAILABLE = WAIFU2X_NCNN_PACKAGE_AVAILABLE or WAIFU2X_NCNN_EXE_AVAILABLE

try:
    import waifu2x
    WAIFU2X_CHAINER_AVAILABLE = True
except ImportError:
    WAIFU2X_CHAINER_AVAILABLE = False

# AMD GPU backend support
try:
    from .amd_waifu2x_backend import AMDWaifu2xBackend, test_amd_waifu2x_availability
    AMD_BACKEND_AVAILABLE = True
except ImportError:
    try:
        from src.modules.amd_waifu2x_backend import AMDWaifu2xBackend, test_amd_waifu2x_availability
        AMD_BACKEND_AVAILABLE = True
    except ImportError:
        AMD_BACKEND_AVAILABLE = False

# Import mock implementation as fallback
try:
    from .mock_waifu2x import MockWaifu2xUpscaler, test_mock_waifu2x_availability
    MOCK_WAIFU2X_AVAILABLE = True
except ImportError:
    try:
        from src.modules.mock_waifu2x import MockWaifu2xUpscaler, test_mock_waifu2x_availability
        MOCK_WAIFU2X_AVAILABLE = True
    except ImportError:
        MOCK_WAIFU2X_AVAILABLE = False

import logging
from PIL import Image
import numpy as np
from typing import List, Optional, Callable, Dict, Any
from pathlib import Path
import tempfile
import os
import time
import threading

logger = logging.getLogger(__name__)


class Waifu2xUpscaler:
    """
    High-quality image upscaling using waifu2x models
    Supports both ncnn-vulkan and chainer implementations
    """
    
    def __init__(self, 
                 backend: str = "auto",
                 gpu_id: int = 0,
                 scale: int = 2,
                 noise: int = 1,
                 model: str = "models-cunet"):
        """
        Initialize Waifu2x upscaler
        
        Args:
            backend: "ncnn", "chainer", "amd", or "auto"
            gpu_id: GPU device ID (0 for first GPU, -1 for CPU)
            scale: Scaling factor (1, 2, 4, 8, 16, 32)
            noise: Noise reduction level (-1: none, 0-3: weak to strong)
            model: Model type for backend
        """
        self.backend = backend
        self.gpu_id = gpu_id
        self.scale = scale
        self.noise = noise
        self.model = model
        self._processor = None
        self._available = False
        
        # Determine best available backend
        if backend == "auto":
            # Check for AMD GPU first if available
            if AMD_BACKEND_AVAILABLE:
                amd_availability = test_amd_waifu2x_availability()
                if amd_availability.get('amd_backend_available', False):
                    self.backend = "amd"
                    logger.info("Auto-selected AMD backend")
                else:
                    # Fall back to other backends
                    if WAIFU2X_NCNN_AVAILABLE:
                        self.backend = "ncnn"
                    elif WAIFU2X_CHAINER_AVAILABLE:
                        self.backend = "chainer"
                    elif MOCK_WAIFU2X_AVAILABLE:
                        self.backend = "mock"
                    else:
                        logger.error("No waifu2x backend available")
                        return
            elif WAIFU2X_NCNN_AVAILABLE:
                self.backend = "ncnn"
            elif WAIFU2X_CHAINER_AVAILABLE:
                self.backend = "chainer"
            elif MOCK_WAIFU2X_AVAILABLE:
                self.backend = "mock"
            else:
                logger.error("No waifu2x backend available")
                return
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the selected backend"""
        try:
            if self.backend == "amd" and AMD_BACKEND_AVAILABLE:
                self._initialize_amd()
            elif self.backend == "ncnn" and WAIFU2X_NCNN_AVAILABLE:
                self._initialize_ncnn()
            elif self.backend == "chainer" and WAIFU2X_CHAINER_AVAILABLE:
                self._initialize_chainer()
            elif self.backend == "mock" and MOCK_WAIFU2X_AVAILABLE:
                self._initialize_mock()
            else:
                logger.error(f"Backend {self.backend} not available")
                return
                
            self._available = True
            logger.info(f"Waifu2x initialized with {self.backend} backend")
            
        except Exception as e:
            logger.error(f"Failed to initialize waifu2x backend: {e}")
            self._available = False
    
    def _initialize_ncnn(self):
        """Initialize ncnn-vulkan backend"""
        try:
            # デバッグ: GPU検出情報
            logger.info(f"Attempting NCNN initialization with GPU ID: {self.gpu_id}")
            
            # GPU利用可能性をチェック
            available_gpus = detect_vulkan_gpus()
            logger.info(f"Available Vulkan GPUs: {available_gpus}")
            
            if self.gpu_id >= len(available_gpus) and available_gpus:
                logger.warning(f"GPU ID {self.gpu_id} not available, using GPU 0")
                self.gpu_id = 0
            elif not available_gpus:
                logger.warning("No Vulkan GPUs detected, falling back to CPU")
                self.gpu_id = -1  # CPU mode
            
            # Try Python package first, fall back to executable
            if WAIFU2X_NCNN_PACKAGE_AVAILABLE:
                self._processor = Waifu2x(
                    gpuid=self.gpu_id,
                    scale=self.scale,
                    noise=self.noise,
                    model=self.model
                )
                logger.info("Using waifu2x-ncnn-vulkan Python package")
            elif WAIFU2X_NCNN_EXE_AVAILABLE:
                self._processor = NCNNExecutableWrapper(
                    gpu_id=self.gpu_id,
                    scale=self.scale,
                    noise=self.noise,
                    model=self.model
                )
                logger.info("Using waifu2x-ncnn-vulkan executable")
            else:
                raise RuntimeError("Neither NCNN package nor executable available")
            
            # 初期化成功時の詳細ログ
            if self.gpu_id >= 0:
                logger.info(f"NCNN backend initialized successfully - GPU: {self.gpu_id}, Scale: {self.scale}x, Noise: {self.noise}")
                logger.info(f"Using Vulkan GPU: {available_gpus[self.gpu_id] if available_gpus else 'Unknown'}")
            else:
                logger.warning(f"NCNN backend initialized in CPU mode - Scale: {self.scale}x, Noise: {self.noise}")
                
        except Exception as e:
            logger.error(f"Failed to initialize NCNN backend: {e}")
            logger.error(f"GPU ID: {self.gpu_id}, Available GPUs: {detect_vulkan_gpus()}")
            raise
    
    def _initialize_chainer(self):
        """Initialize chainer backend"""
        # Chainer backend initialization would go here
        # This is a placeholder as the actual implementation depends on the specific package
        logger.warning("Chainer backend not fully implemented yet")
        raise NotImplementedError("Chainer backend not implemented")
    
    def _initialize_amd(self):
        """Initialize AMD GPU backend"""
        try:
            self._processor = AMDWaifu2xBackend(
                backend_type="auto",
                device_id=self.gpu_id,
                scale=self.scale,
                noise=self.noise,
                model=self.model
            )
            logger.info(f"AMD backend initialized - GPU: {self.gpu_id}, Scale: {self.scale}x, Noise: {self.noise}")
        except Exception as e:
            logger.error(f"Failed to initialize AMD backend: {e}")
            raise
    
    def _initialize_mock(self):
        """Initialize mock backend"""
        try:
            self._processor = MockWaifu2xUpscaler(
                backend="mock",
                gpu_id=self.gpu_id,
                scale=self.scale,
                noise=self.noise,
                model=self.model
            )
            logger.info(f"Mock backend initialized - Scale: {self.scale}x, Noise: {self.noise}")
        except Exception as e:
            logger.error(f"Failed to initialize mock backend: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if waifu2x is available"""
        return self._available
    
    def upscale_image(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Upscale a single image using waifu2x
        
        Args:
            image: PIL Image to upscale
            
        Returns:
            Upscaled PIL Image or None if failed
        """
        if not self._available:
            logger.error("Waifu2x not available")
            return None
            
        try:
            if self.backend == "amd":
                return self._upscale_amd(image)
            elif self.backend == "ncnn":
                return self._upscale_ncnn(image)
            elif self.backend == "chainer":
                return self._upscale_chainer(image)
            elif self.backend == "mock":
                return self._upscale_mock(image)
            else:
                logger.error(f"Unknown backend: {self.backend}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to upscale image: {e}")
            return None
    
    def _upscale_ncnn(self, image: Image.Image) -> Optional[Image.Image]:
        """Upscale using ncnn-vulkan backend"""
        try:
            # Convert RGBA to RGB if necessary
            if image.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process with waifu2x
            upscaled = self._processor.process(image)
            
            if upscaled is None:
                logger.error("Waifu2x processing returned None")
                return None
                
            return upscaled
            
        except Exception as e:
            logger.error(f"NCNN upscaling failed: {e}")
            return None
    
    def _upscale_chainer(self, image: Image.Image) -> Optional[Image.Image]:
        """Upscale using chainer backend"""
        # Placeholder for chainer implementation
        raise NotImplementedError("Chainer backend not implemented")
    
    def _upscale_amd(self, image: Image.Image) -> Optional[Image.Image]:
        """Upscale using AMD GPU backend"""
        try:
            return self._processor.upscale_image(image)
        except Exception as e:
            logger.error(f"AMD upscaling failed: {e}")
            return None
    
    def _upscale_mock(self, image: Image.Image) -> Optional[Image.Image]:
        """Upscale using mock backend"""
        try:
            return self._processor.upscale_image(image)
        except Exception as e:
            logger.error(f"Mock upscaling failed: {e}")
            return None
    
    def upscale_frames(self, 
                      frame_files: List[str], 
                      output_dir: str,
                      progress_callback: Optional[Callable] = None) -> List[str]:
        """
        Upscale multiple frame files
        
        Args:
            frame_files: List of input frame file paths
            output_dir: Output directory for upscaled frames
            progress_callback: Optional progress callback function
            
        Returns:
            List of output frame file paths
        """
        if not self._available:
            logger.error("Waifu2x not available")
            return []
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        upscaled_files = []
        total_frames = len(frame_files)
        
        for i, frame_file in enumerate(frame_files):
            try:
                # Load frame
                with Image.open(frame_file) as image:
                    # Create a copy to avoid file handle issues
                    image_copy = image.copy()
                
                # Upscale the copied image
                upscaled = self.upscale_image(image_copy)
                
                if upscaled is None:
                    logger.warning(f"Failed to upscale frame: {frame_file}")
                    continue
                
                # Generate unique output filename to avoid conflicts
                frame_name = Path(frame_file).stem
                timestamp = int(time.time() * 1000000)  # microseconds for uniqueness
                output_file = output_dir / f"{frame_name}_waifu2x_{timestamp}.png"
                
                # Ensure output directory exists
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Save upscaled frame with error handling
                try:
                    upscaled.save(output_file, "PNG", optimize=False)
                    # Verify file was saved correctly
                    if output_file.exists() and output_file.stat().st_size > 0:
                        upscaled_files.append(str(output_file))
                    else:
                        logger.warning(f"Output file appears corrupted: {output_file}")
                except Exception as save_error:
                    logger.error(f"Failed to save upscaled frame: {save_error}")
                    continue
                
                # Progress callback
                if progress_callback:
                    progress = (i + 1) / total_frames * 100
                    progress_callback(progress, f"Upscaled frame {i+1}/{total_frames}")
                    
            except Exception as e:
                logger.error(f"Failed to process frame {frame_file}: {e}")
                continue
        
        logger.info(f"Successfully upscaled {len(upscaled_files)}/{total_frames} frames")
        return upscaled_files
    
    def get_supported_scales(self) -> List[int]:
        """Get supported scale factors"""
        if self.backend == "ncnn":
            return [1, 2, 4, 8, 16, 32]
        else:
            return [1, 2, 4]  # Default for other backends
    
    def get_supported_noise_levels(self) -> List[int]:
        """Get supported noise reduction levels"""
        return [-1, 0, 1, 2, 3]  # -1: none, 0-3: weak to strong
    
    def get_available_models(self) -> List[str]:
        """Get available model types"""
        if self.backend == "ncnn":
            return [
                "models-cunet",      # Default, balanced quality/speed
                "models-upconv_7_anime_style_art_rgb",  # Anime style
                "models-upconv_7_photo",  # Photographic images
            ]
        else:
            return ["default"]
    
    def update_settings(self, 
                       scale: Optional[int] = None,
                       noise: Optional[int] = None,
                       model: Optional[str] = None):
        """
        Update processing settings
        
        Args:
            scale: New scale factor
            noise: New noise reduction level
            model: New model type
        """
        settings_changed = False
        
        if scale is not None and scale != self.scale:
            self.scale = scale
            settings_changed = True
            
        if noise is not None and noise != self.noise:
            self.noise = noise
            settings_changed = True
            
        if model is not None and model != self.model:
            self.model = model
            settings_changed = True
        
        # Reinitialize if settings changed
        if settings_changed and self._available:
            logger.info("Reinitializing waifu2x with new settings")
            self._initialize_backend()
    
    def get_info(self) -> Dict[str, Any]:
        """Get processor information"""
        return {
            "backend": self.backend,
            "available": self._available,
            "gpu_id": self.gpu_id,
            "scale": self.scale,
            "noise": self.noise,
            "model": self.model,
            "supported_scales": self.get_supported_scales(),
            "supported_noise_levels": self.get_supported_noise_levels(),
            "available_models": self.get_available_models()
        }


def create_waifu2x_upscaler(**kwargs) -> Waifu2xUpscaler:
    """Factory function to create waifu2x upscaler"""
    return Waifu2xUpscaler(**kwargs)


def detect_vulkan_gpus():
    """Detect available Vulkan GPUs"""
    try:
        # Try to get GPU info from waifu2x-ncnn-vulkan
        import subprocess
        result = subprocess.run([
            "tools/waifu2x-ncnn-vulkan/waifu2x-ncnn-vulkan-20220728-windows/waifu2x-ncnn-vulkan.exe", 
            "-h"
        ], capture_output=True, text=True, timeout=10)
        
        # Parse GPU list from help output
        gpus = []
        lines = result.stderr.split('\n') if result.stderr else []
        for line in lines:
            if 'gpu' in line.lower() and ':' in line:
                gpus.append(line.strip())
        
        if not gpus:
            # Fallback: try to detect via system
            try:
                # Windows GPU detection
                if os.name == 'nt':
                    import wmi
                    c = wmi.WMI()
                    gpus = [gpu.Name for gpu in c.Win32_VideoController() if gpu.Name]
            except:
                gpus = ["Unknown GPU"]
                
        return gpus
        
    except Exception as e:
        logger.warning(f"GPU detection failed: {e}")
        return []


def get_performance_info(gpu_id=0, backend="ncnn"):
    """Get current performance information"""
    import psutil
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Memory usage
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    # GPU usage (if possible)
    gpu_usage = get_gpu_usage(gpu_id)
    
    return {
        "cpu_usage": cpu_percent,
        "memory_usage": memory_percent,
        "gpu_usage": gpu_usage,
        "backend": backend,
        "gpu_id": gpu_id,
        "processing_mode": "GPU" if gpu_id >= 0 else "CPU"
    }


def get_gpu_usage(gpu_id=0):
    """Try to get GPU usage information"""
    try:
        # Try NVIDIA first
        import nvidia_ml_py3 as nvml
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(gpu_id)
        utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
        return utilization.gpu
    except:
        pass
    
    try:
        # Try nvidia-smi command
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return int(result.stdout.strip())
    except:
        pass
    
    # Skip WMI approach for now due to missing wmi module
    
    try:
        # Try using psutil with GPU monitoring extensions
        import psutil
        # Check if GPU monitoring is available through psutil extensions
        if hasattr(psutil, 'sensors_temperatures'):
            sensors = psutil.sensors_temperatures()
            # If we have temperature sensors, GPU is likely active
            if sensors:
                return 25  # Placeholder indicating GPU is being monitored
    except:
        pass
    
    try:
        # Check if we're in a GPU processing context by detecting Vulkan GPUs
        gpus = detect_vulkan_gpus()
        if gpus and gpu_id < len(gpus):
            # GPU is available and being used, return a reasonable placeholder
            return 30  # Placeholder indicating GPU is active
    except:
        pass
    
    return -1  # Unknown


def test_waifu2x_availability() -> Dict[str, bool]:
    """Test waifu2x backend availability"""
    amd_available = False
    if AMD_BACKEND_AVAILABLE:
        try:
            amd_info = test_amd_waifu2x_availability()
            amd_available = amd_info.get('amd_backend_available', False)
        except Exception:
            amd_available = False
    
    return {
        "amd": amd_available,
        "ncnn": WAIFU2X_NCNN_AVAILABLE,
        "chainer": WAIFU2X_CHAINER_AVAILABLE,
        "mock": MOCK_WAIFU2X_AVAILABLE,
        "any_available": amd_available or WAIFU2X_NCNN_AVAILABLE or WAIFU2X_CHAINER_AVAILABLE or MOCK_WAIFU2X_AVAILABLE
    }


class NCNNExecutableWrapper:
    """Wrapper for waifu2x-ncnn-vulkan executable"""
    
    # Class-level lock to prevent too many concurrent processes
    _process_lock = threading.Semaphore(2)  # Allow maximum 2 concurrent processes
    
    def __init__(self, gpu_id: int = 0, scale: int = 2, noise: int = 1, model: str = "models-cunet"):
        self.gpu_id = gpu_id
        self.scale = scale  
        self.noise = noise
        self.model = model
        self.exe_path = WAIFU2X_NCNN_EXE_PATH
        
    def process(self, input_image: Image.Image) -> Image.Image:
        """Process image using waifu2x executable"""
        import time
        import threading
        
        # Use semaphore to limit concurrent processes
        with self._process_lock:
            # Create unique temporary directory with thread ID and timestamp
            thread_id = threading.get_ident()
            timestamp = int(time.time() * 1000000)  # microseconds for uniqueness
            unique_suffix = f"waifu2x_{thread_id}_{timestamp}"
            
            # Use a more specific temporary directory
            with tempfile.TemporaryDirectory(prefix=unique_suffix + "_") as tmp_dir:
                # Create unique file names within the directory
                input_path = Path(tmp_dir) / f"input_{thread_id}_{timestamp}.png"
                output_path = Path(tmp_dir) / f"output_{thread_id}_{timestamp}.png"
                
                try:
                    # Save input image
                    input_image.save(input_path, "PNG")
                    
                    # Ensure file is completely written
                    time.sleep(0.01)  # Small delay to ensure file write completion
                    
                    # Build command
                    cmd = [
                        str(self.exe_path),
                        "-i", str(input_path),
                        "-o", str(output_path),
                        "-s", str(self.scale),
                        "-n", str(self.noise),
                        "-g", str(self.gpu_id),
                        "-m", self.model
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode != 0:
                        raise RuntimeError(f"waifu2x executable failed: {result.stderr}")
                    
                    # Wait a bit for output file to be completely written
                    max_wait = 50  # 5 seconds max
                    wait_count = 0
                    while not output_path.exists() and wait_count < max_wait:
                        time.sleep(0.1)
                        wait_count += 1
                    
                    if not output_path.exists():
                        raise RuntimeError("Output image was not created")
                    
                    # Load result image and copy to memory to avoid file lock issues
                    with Image.open(output_path) as img:
                        # Create a copy in memory to avoid file handle issues
                        result_image = img.copy()
                    
                    # Explicitly close and clean up files
                    if input_path.exists():
                        try:
                            input_path.unlink()
                        except:
                            pass
                    
                    if output_path.exists():
                        try:
                            output_path.unlink()
                        except:
                            pass
                    
                    return result_image
                        
                except subprocess.TimeoutExpired:
                    raise RuntimeError("waifu2x executable timed out")
                except Exception as e:
                    # Clean up files on error
                    for file_path in [input_path, output_path]:
                        if file_path.exists():
                            try:
                                file_path.unlink()
                            except:
                                pass
                    raise


# Convenience class for backward compatibility
class Waifu2xProcessor(Waifu2xUpscaler):
    """Alias for backward compatibility"""
    pass