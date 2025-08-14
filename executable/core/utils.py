"""
UpScale App - Utility Functions
Resource management and helper functions for executable
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class ResourceManager:
    """Manages bundled resources and external binaries"""
    
    def __init__(self, resource_dir: Path):
        self.resource_dir = Path(resource_dir)
        self.binaries_dir = self.resource_dir / "binaries"
        self.assets_dir = self.resource_dir / "assets"
        
        # Binary paths
        self.binaries = {
            'ffmpeg': self.binaries_dir / 'ffmpeg.exe',
            'ffprobe': self.binaries_dir / 'ffprobe.exe',
            'waifu2x': self.binaries_dir / 'waifu2x-ncnn-vulkan.exe',
            'vulkaninfo': self.binaries_dir / 'vulkaninfo.exe'
        }
        
        logger.info(f"ResourceManager initialized - Resource dir: {self.resource_dir}")
    
    def get_binary_path(self, name: str) -> Optional[str]:
        """Get path to a binary executable"""
        if name in self.binaries:
            binary_path = self.binaries[name]
            if binary_path.exists():
                logger.debug(f"Found bundled binary: {name} at {binary_path}")
                return str(binary_path)
        
        # Fallback to system PATH
        system_path = shutil.which(name)
        if system_path:
            logger.debug(f"Found system binary: {name} at {system_path}")
            return system_path
        
        # Try with .exe extension
        if not name.endswith('.exe'):
            exe_name = f"{name}.exe"
            if exe_name in self.binaries:
                binary_path = self.binaries[exe_name]
                if binary_path.exists():
                    return str(binary_path)
            
            system_path = shutil.which(exe_name)
            if system_path:
                return system_path
        
        logger.warning(f"Binary not found: {name}")
        return None
    
    def check_binary_availability(self) -> Dict[str, bool]:
        """Check availability of all required binaries"""
        availability = {}
        
        for name in self.binaries.keys():
            availability[name] = self.get_binary_path(name) is not None
        
        return availability
    
    def get_asset_path(self, asset_name: str) -> Optional[Path]:
        """Get path to an asset file"""
        asset_path = self.assets_dir / asset_name
        if asset_path.exists():
            return asset_path
        return None
    
    def run_binary(self, binary_name: str, args: List[str], 
                   timeout: int = 300, check: bool = True) -> subprocess.CompletedProcess:
        """Run a binary with given arguments"""
        binary_path = self.get_binary_path(binary_name)
        if not binary_path:
            raise FileNotFoundError(f"Binary not found: {binary_name}")
        
        cmd = [binary_path] + args
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=check
            )
            return result
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Command timed out after {timeout} seconds: {binary_name}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command failed: {binary_name} - {e.stderr}")

def ensure_directory(path: Path) -> Path:
    """Ensure directory exists and return path"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def safe_remove_file(file_path: Path) -> bool:
    """Safely remove a file"""
    try:
        if file_path.exists():
            file_path.unlink()
            return True
    except Exception as e:
        logger.warning(f"Failed to remove file {file_path}: {e}")
    return False

def safe_remove_directory(dir_path: Path) -> bool:
    """Safely remove a directory"""
    try:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            return True
    except Exception as e:
        logger.warning(f"Failed to remove directory {dir_path}: {e}")
    return False

def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB"""
    if file_path.exists():
        return file_path.stat().st_size / (1024 * 1024)
    return 0.0

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def get_available_memory_gb() -> float:
    """Get available system memory in GB"""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        # Fallback estimation
        return 4.0  # Conservative estimate

def get_disk_space_gb(path: Path) -> float:
    """Get available disk space in GB"""
    try:
        return shutil.disk_usage(path).free / (1024**3)
    except Exception:
        return 0.0

class ProgressCallback:
    """Helper class for progress tracking"""
    
    def __init__(self, callback=None):
        self.callback = callback
        self.current = 0.0
        self.message = ""
    
    def update(self, progress: float, message: str = ""):
        """Update progress"""
        self.current = max(0, min(100, progress))
        self.message = message
        
        if self.callback:
            try:
                self.callback(self.current, self.message)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    def increment(self, amount: float, message: str = ""):
        """Increment progress by amount"""
        self.update(self.current + amount, message)

class SimpleTimer:
    """Simple timer for performance measurement"""
    
    def __init__(self):
        import time
        self.start_time = time.time()
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        import time
        return time.time() - self.start_time
    
    def elapsed_str(self) -> str:
        """Get elapsed time as formatted string"""
        return format_duration(self.elapsed())