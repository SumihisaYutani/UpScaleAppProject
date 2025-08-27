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
        
        # Detect if running as PyInstaller bundle
        self.is_bundle = getattr(sys, 'frozen', False)
        if self.is_bundle:
            # PyInstaller extracts to sys._MEIPASS
            self.bundle_dir = Path(sys._MEIPASS)
            logger.info(f"Running as PyInstaller bundle - MEIPASS: {self.bundle_dir}")
        else:
            self.bundle_dir = self.resource_dir
        
        # Binary paths - check both bundle and resource locations
        self.binaries = {
            'ffmpeg': 'ffmpeg.exe',
            'ffprobe': 'ffprobe.exe', 
            'waifu2x': 'waifu2x-ncnn-vulkan.exe',
            'realcugan-ncnn-vulkan': 'realcugan-ncnn-vulkan.exe',
            'realesrgan-ncnn-vulkan': 'realesrgan-ncnn-vulkan.exe',
            'vulkaninfo': 'vulkaninfo.exe'
        }
        
        logger.info(f"ResourceManager initialized - Resource dir: {self.resource_dir}")
        logger.info(f"Bundle mode: {self.is_bundle}, Bundle dir: {self.bundle_dir}")
    
    def get_binary_path(self, name: str) -> Optional[str]:
        """Get path to a binary executable - prioritize system installations for ffmpeg/ffprobe"""
        if name not in self.binaries:
            logger.warning(f"Unknown binary requested: {name}")
            return None
        
        binary_filename = self.binaries[name]
        
        # Method 1: For ffmpeg/ffprobe, check system paths first (more reliable)
        if name in ['ffmpeg', 'ffprobe']:
            # Check system PATH first
            import shutil
            system_binary = shutil.which(binary_filename)
            if system_binary:
                logger.info(f"Found system binary in PATH: {name} at {system_binary}")
                return system_binary
            
            # Check common install locations
            system_locations = [
                Path("C:/ffmpeg/bin") / binary_filename,
                Path("C:/Program Files/ffmpeg/bin") / binary_filename,
                Path("C:/Program Files (x86)/ffmpeg/bin") / binary_filename,
            ]
            
            for system_path in system_locations:
                if system_path.exists():
                    logger.info(f"Found system binary: {name} at {system_path}")
                    return str(system_path)
        
        # Method 2: Check PyInstaller bundle directory for other binaries
        if self.is_bundle:
            # Check in root of bundle
            bundle_path = self.bundle_dir / binary_filename
            if bundle_path.exists():
                logger.info(f"Found bundled binary: {name} at {bundle_path}")
                return str(bundle_path)
            
            # Check in resources/binaries within bundle
            bundle_res_path = self.bundle_dir / "resources" / "binaries" / binary_filename
            if bundle_res_path.exists():
                logger.info(f"Found bundled binary in resources: {name} at {bundle_res_path}")
                return str(bundle_res_path)
            
            # Search recursively in bundle directory
            for search_path in self.bundle_dir.rglob(binary_filename):
                if search_path.is_file():
                    logger.info(f"Found bundled binary recursively: {name} at {search_path}")
                    return str(search_path)
        
        # Method 2: Check regular resource directory
        resource_binary_path = self.binaries_dir / binary_filename
        if resource_binary_path.exists():
            logger.info(f"Found resource binary: {name} at {resource_binary_path}")
            return str(resource_binary_path)
        
        # Method 3: Search in all subdirectories of binaries_dir
        for search_path in self.binaries_dir.rglob(f"{name}*"):
            if search_path.is_file() and search_path.suffix.lower() in ['.exe', '']:
                logger.info(f"Found binary in subdirectory: {name} at {search_path}")
                return str(search_path)
        
        # Method 4: Check project tools directory (for development)
        project_root = Path(__file__).parent.parent.parent  # Go up from executable/core/
        if name == 'waifu2x':
            tools_path = project_root / "tools" / "waifu2x-ncnn-vulkan" / "waifu2x-ncnn-vulkan-20220728-windows" / "waifu2x-ncnn-vulkan.exe"
            if tools_path.exists():
                logger.info(f"Found project tools binary: {name} at {tools_path}")
                return str(tools_path)
        elif name in ['ffmpeg', 'ffprobe']:
            # Check common FFmpeg installation paths
            ffmpeg_paths = [
                Path("C:/ffmpeg/bin") / binary_filename,
                Path("C:/Program Files/ffmpeg/bin") / binary_filename,
                Path("C:/ffmpeg") / binary_filename,
            ]
            for ffmpeg_path in ffmpeg_paths:
                if ffmpeg_path.exists():
                    logger.info(f"Found system FFmpeg binary: {name} at {ffmpeg_path}")
                    return str(ffmpeg_path)
        
        # Method 5: Fallback to system PATH
        system_path = shutil.which(binary_filename)
        if system_path:
            logger.info(f"Found system binary: {name} at {system_path}")
            return system_path
        
        # Method 5: Try alternative names
        if name == 'waifu2x':
            for alt_name in ['waifu2x-ncnn-vulkan.exe', 'waifu2x.exe', 'waifu2x-ncnn.exe']:
                if self.is_bundle:
                    alt_path = self.bundle_dir / alt_name
                    if alt_path.exists():
                        logger.info(f"Found alternative bundled binary: {alt_name} at {alt_path}")
                        return str(alt_path)
                    
                    # Search recursively for alternative names
                    for search_path in self.bundle_dir.rglob(alt_name):
                        if search_path.is_file():
                            logger.info(f"Found alternative bundled binary recursively: {alt_name} at {search_path}")
                            return str(search_path)
                
                alt_resource_path = self.binaries_dir / alt_name
                if alt_resource_path.exists():
                    logger.info(f"Found alternative resource binary: {alt_name} at {alt_resource_path}")
                    return str(alt_resource_path)
        
        elif name == 'realcugan-ncnn-vulkan':
            for alt_name in ['realcugan-ncnn-vulkan.exe', 'realcugan.exe', 'real-cugan-ncnn-vulkan.exe']:
                if self.is_bundle:
                    alt_path = self.bundle_dir / alt_name
                    if alt_path.exists():
                        logger.info(f"Found alternative bundled Real-CUGAN binary: {alt_name} at {alt_path}")
                        return str(alt_path)
                    
                    # Search recursively for alternative names
                    for search_path in self.bundle_dir.rglob(alt_name):
                        if search_path.is_file():
                            logger.info(f"Found alternative bundled Real-CUGAN binary recursively: {alt_name} at {search_path}")
                            return str(search_path)
                
                alt_resource_path = self.binaries_dir / alt_name
                if alt_resource_path.exists():
                    logger.info(f"Found alternative resource Real-CUGAN binary: {alt_name} at {alt_resource_path}")
                    return str(alt_resource_path)
        
        elif name == 'realesrgan-ncnn-vulkan':
            for alt_name in ['realesrgan-ncnn-vulkan.exe', 'realesrgan.exe', 'real-esrgan-ncnn-vulkan.exe']:
                if self.is_bundle:
                    alt_path = self.bundle_dir / alt_name
                    if alt_path.exists():
                        logger.info(f"Found alternative bundled Real-ESRGAN binary: {alt_name} at {alt_path}")
                        return str(alt_path)
                    
                    # Search recursively for alternative names
                    for search_path in self.bundle_dir.rglob(alt_name):
                        if search_path.is_file():
                            logger.info(f"Found alternative bundled Real-ESRGAN binary recursively: {alt_name} at {search_path}")
                            return str(search_path)
                
                alt_resource_path = self.binaries_dir / alt_name
                if alt_resource_path.exists():
                    logger.info(f"Found alternative resource Real-ESRGAN binary: {alt_name} at {alt_resource_path}")
                    return str(alt_resource_path)
        
        logger.warning(f"Binary not found: {name} (searched: bundle={self.is_bundle}, resource_dir={self.binaries_dir}, system paths)")
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
                   timeout: int = 300, check: bool = True, hide_window: bool = False) -> subprocess.CompletedProcess:
        """Run a binary with given arguments"""
        binary_path = self.get_binary_path(binary_name)
        if not binary_path:
            raise FileNotFoundError(f"Binary not found: {binary_name}")
        
        cmd = [binary_path] + args
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        # Setup to hide console window on Windows
        startupinfo = None
        if hide_window and os.name == 'nt':  # Windows
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=check,
                startupinfo=startupinfo
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