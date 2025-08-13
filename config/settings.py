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
    "max_duration_minutes": 30,
    "default_upscale_factor": 1.5,
    "max_resolution": (1920, 1080)
}

# AI processing settings
AI_SETTINGS = {
    "model_name": "stabilityai/stable-diffusion-2-1",
    "batch_size": 4,
    "guidance_scale": 7.5,
    "num_inference_steps": 20,
    "device": "cuda" if os.environ.get("CUDA_AVAILABLE") else "cpu"
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

# Create necessary directories
for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)