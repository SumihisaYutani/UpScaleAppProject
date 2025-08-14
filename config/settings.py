"""
UpScaleApp Configuration Settings
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Video processing settings
VIDEO_SETTINGS = {
    "supported_formats": [".mp4"],
    "supported_codecs": ["h264", "h265", "hevc", "avc"],
    "max_file_size_gb": 2,
    "max_duration_minutes": 60,
    "default_upscale_factor": 1.5,
    "max_resolution": (1920, 1080)
}

# AI processing settings
AI_SETTINGS = {
    "model_name": "stabilityai/stable-diffusion-2-1",
    "batch_size": 4,
    "guidance_scale": 7.5,
    "num_inference_steps": 20,
    "device": "cuda" if os.environ.get("CUDA_AVAILABLE") else "cpu",
    "preferred_method": "auto"  # "auto", "waifu2x", "stable_diffusion", "simple"
}

# Waifu2x specific settings
WAIFU2X_SETTINGS = {
    "backend": "auto",  # "auto", "amd", "ncnn", "chainer"
    "gpu_id": 0,  # GPU device ID (0 for first GPU, -1 for CPU)
    "scale": 2,  # Scaling factor (1, 2, 4, 8, 16, 32 for ncnn)
    "noise": 1,  # Noise reduction level (-1: none, 0-3: weak to strong)
    "model": "models-cunet",  # Model type for backend
    "tile_size": 512,  # Tile size for processing large images
    "tile_pad": 10,  # Padding for tiles
    # AMD GPU specific settings
    "amd_backend_type": "auto",  # "auto", "rocm", "vulkan"
    "amd_optimize": True,  # Apply AMD-specific optimizations
    "amd_memory_fraction": 0.9  # Memory fraction for ROCm backend
}

# File paths
PATHS = {
    "temp_dir": BASE_DIR / "temp",
    "output_dir": BASE_DIR / "output",
    "models_dir": BASE_DIR / "models",
    "logs_dir": BASE_DIR / "logs"
}

# Performance settings
PERFORMANCE = {
    "max_memory_gb": 8,
    "max_concurrent_frames": 10,
    "cleanup_temp_files": True
}

# Logging settings
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

# Dependency checking
DEPENDENCIES = {
    "torch": False,
    "cv2": False,
    "diffusers": False,
    "ffmpeg": False,
    "PIL": False,
    "numpy": False,
    "waifu2x_ncnn": False,
    "waifu2x_chainer": False,
    "waifu2x_amd": False
}

# Check available dependencies
try:
    import torch
    DEPENDENCIES["torch"] = True
except ImportError:
    pass

try:
    import cv2
    DEPENDENCIES["cv2"] = True
except ImportError:
    pass

try:
    import diffusers
    DEPENDENCIES["diffusers"] = True
except ImportError:
    pass

try:
    import ffmpeg
    DEPENDENCIES["ffmpeg"] = True
except ImportError:
    pass

try:
    from PIL import Image
    DEPENDENCIES["PIL"] = True
except ImportError:
    pass

try:
    import numpy
    DEPENDENCIES["numpy"] = True
except ImportError:
    pass

try:
    from waifu2x_ncnn_vulkan import Waifu2x
    DEPENDENCIES["waifu2x_ncnn"] = True
except ImportError:
    pass

try:
    import waifu2x
    DEPENDENCIES["waifu2x_chainer"] = True
except ImportError:
    pass

# Check AMD backend availability
try:
    from src.modules.amd_waifu2x_backend import test_amd_waifu2x_availability
    amd_info = test_amd_waifu2x_availability()
    DEPENDENCIES["waifu2x_amd"] = amd_info.get('amd_backend_available', False)
except ImportError:
    try:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
        from modules.amd_waifu2x_backend import test_amd_waifu2x_availability
        amd_info = test_amd_waifu2x_availability()
        DEPENDENCIES["waifu2x_amd"] = amd_info.get('amd_backend_available', False)
    except ImportError:
        DEPENDENCIES["waifu2x_amd"] = False

# Check mock waifu2x availability
try:
    from src.modules.mock_waifu2x import MockWaifu2xUpscaler
    DEPENDENCIES["waifu2x_mock"] = True
except ImportError:
    try:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
        from modules.mock_waifu2x import MockWaifu2xUpscaler
        DEPENDENCIES["waifu2x_mock"] = True
    except ImportError:
        DEPENDENCIES["waifu2x_mock"] = False

# Adjust AI settings based on available dependencies
if not DEPENDENCIES["torch"] or not DEPENDENCIES["diffusers"]:
    AI_SETTINGS["device"] = "cpu"
    AI_SETTINGS["model_name"] = "fallback"

# Create necessary directories with error handling
for name, path in PATHS.items():
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directory {name} at {path}: {e}")

# Environment status
ENVIRONMENT_STATUS = {
    "ai_available": DEPENDENCIES["torch"] and DEPENDENCIES["diffusers"],
    "waifu2x_available": DEPENDENCIES["waifu2x_amd"] or DEPENDENCIES["waifu2x_ncnn"] or DEPENDENCIES["waifu2x_chainer"] or DEPENDENCIES["waifu2x_mock"],
    "waifu2x_amd_available": DEPENDENCIES["waifu2x_amd"],
    "waifu2x_ncnn_available": DEPENDENCIES["waifu2x_ncnn"],
    "waifu2x_chainer_available": DEPENDENCIES["waifu2x_chainer"],
    "waifu2x_mock_available": DEPENDENCIES["waifu2x_mock"],
    "video_processing_available": DEPENDENCIES["cv2"] and DEPENDENCIES["ffmpeg"],
    "basic_functionality_available": DEPENDENCIES["PIL"] and DEPENDENCIES["numpy"],
    "any_ai_available": (DEPENDENCIES["torch"] and DEPENDENCIES["diffusers"]) or 
                       DEPENDENCIES["waifu2x_amd"] or DEPENDENCIES["waifu2x_ncnn"] or DEPENDENCIES["waifu2x_chainer"] or DEPENDENCIES["waifu2x_mock"],
    "dependencies": DEPENDENCIES
}