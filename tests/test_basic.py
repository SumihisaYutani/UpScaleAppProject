"""
Basic tests for UpScale App modules
"""

import pytest
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from modules.video_processor import VideoProcessor
from modules.ai_processor import SimpleUpscaler
from upscale_app import UpScaleApp


class TestVideoProcessor:
    """Test VideoProcessor functionality"""
    
    def setup_method(self):
        self.processor = VideoProcessor()
    
    def test_supported_formats(self):
        """Test supported format validation"""
        assert ".mp4" in self.processor.supported_formats
        assert ".avi" not in self.processor.supported_formats
    
    def test_upscaled_resolution_calculation(self):
        """Test resolution calculation"""
        width, height = self.processor.get_upscaled_resolution(640, 480, 1.5)
        assert width == 960  # 640 * 1.5
        assert height == 720  # 480 * 1.5
        
        # Test even number adjustment
        width, height = self.processor.get_upscaled_resolution(641, 481, 1.5)
        assert width % 2 == 0  # Should be even
        assert height % 2 == 0  # Should be even
    
    def test_file_validation_nonexistent(self):
        """Test validation of non-existent file"""
        result = self.processor.validate_video_file("nonexistent.mp4")
        assert not result["valid"]
        assert "does not exist" in result["error"]


class TestSimpleUpscaler:
    """Test SimpleUpscaler functionality"""
    
    def test_simple_upscaling_parameters(self):
        """Test that simple upscaler accepts correct parameters"""
        # This test just verifies the method signature
        # Real testing would require actual image files
        upscaler = SimpleUpscaler()
        assert hasattr(upscaler, 'upscale_image_simple')
        
        # Test method is static
        assert callable(SimpleUpscaler.upscale_image_simple)


class TestUpScaleApp:
    """Test main UpScaleApp functionality"""
    
    def setup_method(self):
        self.app = UpScaleApp(use_ai=False)  # Disable AI for testing
    
    def test_app_initialization(self):
        """Test app initializes correctly"""
        assert self.app.video_processor is not None
        assert self.app.frame_extractor is not None
        assert self.app.video_builder is not None
        assert not self.app.use_ai  # We disabled AI
    
    def test_system_info(self):
        """Test system info collection"""
        info = self.app.get_system_info()
        
        required_keys = [
            "platform", "python_version", "cuda_available",
            "temp_dir", "output_dir", "max_memory_gb"
        ]
        
        for key in required_keys:
            assert key in info
        
        assert isinstance(info["cuda_available"], bool)
        assert isinstance(info["max_memory_gb"], (int, float))


def test_imports():
    """Test that all modules can be imported"""
    try:
        from modules.video_processor import VideoProcessor, VideoFrameExtractor
        from modules.video_builder import VideoBuilder
        from modules.ai_processor import AIUpscaler, SimpleUpscaler
        from upscale_app import UpScaleApp
        from config.settings import VIDEO_SETTINGS, AI_SETTINGS
        assert True  # If we get here, imports worked
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])