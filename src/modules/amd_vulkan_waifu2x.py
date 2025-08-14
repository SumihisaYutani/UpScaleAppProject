"""
AMD Vulkan-optimized Waifu2x Processor
Implements waifu2x processing optimized for AMD GPUs using Vulkan backend
"""

import logging
import os
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from PIL import Image
import platform

logger = logging.getLogger(__name__)


class AMDVulkanWaifu2xProcessor:
    """
    AMD GPU-optimized Vulkan waifu2x processor
    Uses waifu2x-ncnn-vulkan with AMD-specific optimizations
    """
    
    def __init__(self,
                 gpu_id: int = 0,
                 scale: int = 2,
                 noise: int = 1,
                 model: str = "models-cunet"):
        """
        Initialize AMD Vulkan waifu2x processor
        
        Args:
            gpu_id: Vulkan device ID for AMD GPU
            scale: Upscaling factor (1, 2, 4, 8, 16, 32)
            noise: Noise reduction level (-1: none, 0-3: levels)
            model: Model type for processing
        """
        self.gpu_id = gpu_id
        self.scale = scale
        self.noise = noise
        self.model = model
        
        self.platform = platform.system().lower()
        self._executable_path = None
        self._models_path = None
        self._available = False
        
        self._setup_vulkan_environment()
        self._find_waifu2x_executable()
        self._apply_amd_optimizations()
    
    def _setup_vulkan_environment(self):
        """Set up Vulkan environment for AMD GPUs"""
        # AMD-specific Vulkan environment variables
        if self.platform == "linux":
            # Use RADV (AMD's open-source Vulkan driver) for better performance
            os.environ['AMD_VULKAN_ICD'] = 'RADV'
            os.environ['RADV_PERFTEST'] = 'aco,llvm'  # Enable ACO compiler and LLVM
            os.environ['VK_LAYER_PATH'] = ''  # Disable validation layers for performance
            
            # AMD GPU memory optimizations
            os.environ['AMD_DEBUG'] = 'nodma'  # Disable DMA for stability
            
        elif self.platform == "windows":
            # Windows AMD Vulkan optimizations
            os.environ['AMD_VULKAN_ICD'] = 'AMDVLK'
            
        # General Vulkan optimizations
        os.environ['VK_INSTANCE_LAYERS'] = ''  # No validation layers
        os.environ['MESA_VK_DEVICE_SELECT'] = str(self.gpu_id)  # Select specific GPU
        
        logger.info(f"Vulkan environment configured for AMD GPU {self.gpu_id}")
    
    def _find_waifu2x_executable(self):
        """Find waifu2x-ncnn-vulkan executable"""
        possible_names = [
            'waifu2x-ncnn-vulkan',
            'waifu2x-ncnn-vulkan.exe',
            'waifu2x_ncnn_vulkan',
            'waifu2x_ncnn_vulkan.exe'
        ]
        
        possible_paths = [
            # Current directory and subdirectories
            Path.cwd(),
            Path.cwd() / 'bin',
            Path.cwd() / 'tools',
            Path.cwd() / 'waifu2x-ncnn-vulkan',
        ]
        
        # Add system PATH locations
        system_path = os.environ.get('PATH', '').split(os.pathsep)
        possible_paths.extend([Path(p) for p in system_path if p])
        
        # Search for executable
        for path in possible_paths:
            if not path.exists():
                continue
                
            for name in possible_names:
                executable = path / name
                if executable.exists() and executable.is_file():
                    self._executable_path = executable
                    logger.info(f"Found waifu2x executable: {executable}")
                    break
            
            if self._executable_path:
                break
        
        if not self._executable_path:
            # Try to download or install
            self._download_waifu2x_ncnn_vulkan()
    
    def _download_waifu2x_ncnn_vulkan(self):
        """Download waifu2x-ncnn-vulkan for AMD GPUs"""
        try:
            import requests
            import zipfile
            
            # Create tools directory
            tools_dir = Path.cwd() / 'tools' / 'waifu2x-ncnn-vulkan'
            tools_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine download URL based on platform
            if self.platform == "windows":
                url = "https://github.com/nihui/waifu2x-ncnn-vulkan/releases/download/20220728/waifu2x-ncnn-vulkan-20220728-windows.zip"
                exe_name = "waifu2x-ncnn-vulkan.exe"
            elif self.platform == "linux":
                url = "https://github.com/nihui/waifu2x-ncnn-vulkan/releases/download/20220728/waifu2x-ncnn-vulkan-20220728-ubuntu.zip"
                exe_name = "waifu2x-ncnn-vulkan"
            else:
                logger.error(f"Unsupported platform for download: {self.platform}")
                return
            
            # Download
            logger.info(f"Downloading waifu2x-ncnn-vulkan from {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Save and extract
            zip_file = tools_dir / "waifu2x-ncnn-vulkan.zip"
            with open(zip_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(tools_dir)
            
            # Find extracted executable
            for item in tools_dir.rglob(exe_name):
                if item.is_file():
                    self._executable_path = item
                    # Make executable on Linux
                    if self.platform == "linux":
                        os.chmod(self._executable_path, 0o755)
                    logger.info(f"Found extracted executable: {self._executable_path}")
                    break
            
            # Clean up zip file
            zip_file.unlink()
            
            if self._executable_path:
                logger.info(f"Downloaded waifu2x executable: {self._executable_path}")
            else:
                logger.error("Failed to find executable after download")
                
        except Exception as e:
            logger.error(f"Failed to download waifu2x-ncnn-vulkan: {e}")
    
    def _apply_amd_optimizations(self):
        """Apply AMD-specific optimizations"""
        if not self._executable_path:
            return
        
        try:
            # Test if executable works with AMD GPU
            cmd = [
                str(self._executable_path),
                '-g', str(self.gpu_id),
                '-s', '1',  # Scale 1x for testing
                '-n', '0',  # No noise reduction for testing
                '-h'        # Help to test if it runs
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 or "Usage" in result.stderr:
                self._available = True
                logger.info("AMD Vulkan waifu2x processor initialized successfully")
            else:
                logger.error(f"waifu2x test failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Failed to test waifu2x executable: {e}")
    
    def is_available(self) -> bool:
        """Check if AMD Vulkan processor is available"""
        return self._available and self._executable_path is not None
    
    def process(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Process image using AMD Vulkan backend
        
        Args:
            image: PIL Image to process
            
        Returns:
            Processed PIL Image or None if failed
        """
        if not self.is_available():
            logger.error("AMD Vulkan processor not available")
            return None
        
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "input.png"
            output_file = temp_path / "output.png"
            
            try:
                # Save input image
                image.save(input_file, "PNG")
                
                # Build command with AMD optimizations
                cmd = [
                    str(self._executable_path),
                    '-i', str(input_file),
                    '-o', str(output_file),
                    '-g', str(self.gpu_id),  # Use specified GPU
                    '-s', str(self.scale),   # Scale factor
                    '-n', str(self.noise),   # Noise reduction
                    '-m', self.model,        # Model type
                    '-j', '1:2:2',          # Thread configuration (1:2:2 good for AMD)
                    '-f', 'png'             # Output format
                ]
                
                # Add AMD-specific parameters
                if self.platform == "linux":
                    # Use more threads for AMD GPUs
                    cmd.extend(['-j', '2:4:4'])
                
                # Execute waifu2x
                logger.debug(f"Executing: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode != 0:
                    logger.error(f"waifu2x processing failed: {result.stderr}")
                    return None
                
                # Load and return result
                if output_file.exists():
                    processed_image = Image.open(output_file)
                    # Convert to RGB if needed
                    if processed_image.mode != 'RGB':
                        processed_image = processed_image.convert('RGB')
                    return processed_image.copy()  # Copy to ensure file is closed
                else:
                    logger.error("Output file not created")
                    return None
                    
            except subprocess.TimeoutExpired:
                logger.error("waifu2x processing timed out")
                return None
            except Exception as e:
                logger.error(f"Processing failed: {e}")
                return None
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get Vulkan device information"""
        if not self.is_available():
            return {}
        
        try:
            # Get device list
            cmd = [str(self._executable_path), '-h']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            return {
                'executable_path': str(self._executable_path),
                'gpu_id': self.gpu_id,
                'scale': self.scale,
                'noise': self.noise,
                'model': self.model,
                'platform': self.platform,
                'available': self._available
            }
            
        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            return {}
    
    def benchmark_performance(self, test_image_size: tuple = (512, 512)) -> Dict[str, Any]:
        """Benchmark AMD GPU performance"""
        if not self.is_available():
            return {'error': 'Processor not available'}
        
        try:
            import time
            
            # Create test image
            test_image = Image.new('RGB', test_image_size, (128, 128, 128))
            
            # Warm up
            self.process(test_image)
            
            # Benchmark
            start_time = time.time()
            result = self.process(test_image)
            end_time = time.time()
            
            if result:
                processing_time = end_time - start_time
                pixels_per_second = (test_image_size[0] * test_image_size[1]) / processing_time
                
                return {
                    'processing_time': processing_time,
                    'input_size': test_image_size,
                    'output_size': result.size,
                    'pixels_per_second': pixels_per_second,
                    'scale_factor': self.scale,
                    'success': True
                }
            else:
                return {'error': 'Processing failed', 'success': False}
                
        except Exception as e:
            return {'error': str(e), 'success': False}


def create_amd_vulkan_processor(**kwargs) -> AMDVulkanWaifu2xProcessor:
    """Factory function to create AMD Vulkan processor"""
    return AMDVulkanWaifu2xProcessor(**kwargs)


def test_amd_vulkan_availability() -> Dict:
    """Test AMD Vulkan waifu2x availability"""
    try:
        processor = AMDVulkanWaifu2xProcessor()
        return {
            'available': processor.is_available(),
            'executable_found': processor._executable_path is not None,
            'device_info': processor.get_device_info()
        }
    except Exception as e:
        return {
            'available': False,
            'executable_found': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # Test AMD Vulkan processor
    print("AMD Vulkan Waifu2x Processor Test")
    print("=" * 40)
    
    availability = test_amd_vulkan_availability()
    print(f"Available: {availability['available']}")
    print(f"Executable Found: {availability['executable_found']}")
    
    if availability.get('device_info'):
        print("Device Info:")
        for key, value in availability['device_info'].items():
            print(f"  {key}: {value}")
    
    if availability.get('error'):
        print(f"Error: {availability['error']}")