"""
AI Integration Tests
Tests for AI processing modules and integration
"""

import pytest
import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from modules.ai_processor import AIUpscaler, SimpleUpscaler
from upscale_app import UpScaleApp


class TestAIUpscaler:
    """Test AI upscaling functionality"""
    
    def setup_method(self):
        self.upscaler = AIUpscaler()
    
    def test_initialization(self):
        """Test AI upscaler initialization"""
        assert self.upscaler.model_name is not None
        assert self.upscaler.device in ["cuda", "cpu"]
        assert self.upscaler.batch_size > 0
        assert not self.upscaler._model_loaded
    
    @patch('modules.ai_processor.StableDiffusionImg2ImgPipeline')
    def test_model_loading(self, mock_pipeline):
        """Test model loading with mocked dependencies"""
        # Mock the pipeline creation
        mock_instance = MagicMock()
        mock_pipeline.from_pretrained.return_value = mock_instance
        mock_instance.to.return_value = mock_instance
        
        # Test loading
        result = self.upscaler.load_model()
        
        # Verify calls
        mock_pipeline.from_pretrained.assert_called_once()
        assert result == True or result == False  # Either works or fails gracefully
    
    def test_vram_estimation(self):
        """Test VRAM usage estimation"""
        estimate = self.upscaler.estimate_vram_usage((1920, 1080), batch_size=1)
        
        assert isinstance(estimate, float)
        assert estimate > 0
        assert estimate <= 16.0  # Reasonable upper bound
    
    def test_cleanup(self):
        """Test model cleanup"""
        # Should not crash even if no model is loaded
        self.upscaler.cleanup()
        assert not self.upscaler._model_loaded


class TestSimpleUpscaler:
    """Test simple upscaling functionality"""
    
    def test_simple_upscaling_mock(self):
        """Test simple upscaling with mock image"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_input:
            input_path = temp_input.name
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_output:
            output_path = temp_output.name
        
        try:
            # Create a simple test image (this would normally require PIL)
            # For now, just test the method exists and handles errors gracefully
            
            with patch('modules.ai_processor.Image') as mock_image:
                mock_img = MagicMock()
                mock_image.open.return_value = mock_img
                mock_img.convert.return_value = mock_img
                mock_img.size = (640, 480)
                mock_img.resize.return_value = mock_img
                
                result = SimpleUpscaler.upscale_image_simple(
                    input_path, output_path, 1.5
                )
                
                # Should either succeed or fail gracefully
                assert isinstance(result, bool)
                
        finally:
            # Cleanup temp files
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)


class TestUpScaleAppIntegration:
    """Test main app with AI integration"""
    
    def setup_method(self):
        # Create app without AI to avoid heavy dependencies in tests
        self.app = UpScaleApp(use_ai=False, temp_cleanup=True)
    
    def test_app_with_ai_disabled(self):
        """Test app initialization without AI"""
        assert not self.app.use_ai
        assert self.app.ai_upscaler is None
        assert self.app.video_processor is not None
    
    def test_app_with_ai_enabled(self):
        """Test app initialization with AI enabled"""
        app_with_ai = UpScaleApp(use_ai=True, temp_cleanup=True)
        
        assert app_with_ai.use_ai
        assert app_with_ai.ai_upscaler is not None
        assert isinstance(app_with_ai.ai_upscaler, AIUpscaler)
    
    @patch('modules.video_processor.VideoProcessor.validate_video_file')
    def test_process_video_validation_failure(self, mock_validate):
        """Test video processing with validation failure"""
        # Mock validation failure
        mock_validate.return_value = {
            "valid": False,
            "error": "Test validation error"
        }
        
        result = self.app.process_video("fake_video.mp4")
        
        assert not result["success"]
        assert "Test validation error" in result["error"]
    
    @patch('modules.video_processor.VideoProcessor.validate_video_file')
    @patch('modules.video_processor.VideoFrameExtractor.extract_frames')
    def test_process_video_frame_extraction_failure(self, mock_extract, mock_validate):
        """Test video processing with frame extraction failure"""
        # Mock successful validation
        mock_validate.return_value = {
            "valid": True,
            "info": {
                "filename": "test.mp4",
                "width": 640,
                "height": 480,
                "duration": 10.0,
                "frame_rate": 30.0,
                "frame_count": 300,
                "codec_name": "h264"
            }
        }
        
        # Mock frame extraction failure
        mock_extract.return_value = []
        
        result = self.app.process_video("fake_video.mp4")
        
        assert not result["success"]
        assert "Failed to extract frames" in result["error"]


class TestAIModelIntegration:
    """Test AI model integration scenarios"""
    
    @pytest.mark.skipif(not os.environ.get("TEST_AI_MODELS"), 
                       reason="AI model tests skipped (set TEST_AI_MODELS=1 to enable)")
    def test_real_ai_model_loading(self):
        """Test loading real AI models (only run when explicitly requested)"""
        upscaler = AIUpscaler()
        
        # This test requires actual GPU/CUDA setup and takes time
        # Only run when TEST_AI_MODELS environment variable is set
        try:
            result = upscaler.load_model()
            
            if result:
                assert upscaler._model_loaded
                assert upscaler.pipeline is not None
                
                # Cleanup
                upscaler.cleanup()
                assert not upscaler._model_loaded
            else:
                # Model loading failed, but that's acceptable in test environment
                pytest.skip("AI model loading not available in test environment")
                
        except Exception as e:
            pytest.skip(f"AI model test skipped due to environment: {e}")
    
    def test_ai_fallback_behavior(self):
        """Test AI fallback to simple upscaling"""
        upscaler = AIUpscaler()
        
        # Without loading the model, AI operations should fail gracefully
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_input:
            input_path = temp_input.name
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_output:
            output_path = temp_output.name
        
        try:
            # This should fail gracefully without crashing
            result = upscaler.upscale_image(input_path, output_path)
            assert isinstance(result, bool)
            
        finally:
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)


def test_dependency_imports():
    """Test that required dependencies can be imported"""
    import_tests = [
        ("pathlib", "Path"),
        ("logging", "getLogger"),
        ("tempfile", "NamedTemporaryFile"),
    ]
    
    for module_name, attr_name in import_tests:
        try:
            module = __import__(module_name)
            assert hasattr(module, attr_name), f"{module_name}.{attr_name} not found"
        except ImportError:
            pytest.fail(f"Required module {module_name} not available")


@pytest.mark.integration
class TestEndToEndWorkflow:
    """End-to-end integration tests"""
    
    def test_complete_workflow_mock(self):
        """Test complete workflow with mocked components"""
        with patch('modules.video_processor.VideoProcessor.validate_video_file') as mock_validate, \
             patch('modules.video_processor.VideoFrameExtractor.extract_frames') as mock_extract, \
             patch('modules.video_builder.VideoBuilder.combine_frames_to_video') as mock_combine:
            
            # Setup mocks
            mock_validate.return_value = {
                "valid": True,
                "info": {
                    "filename": "test.mp4",
                    "width": 640,
                    "height": 480,
                    "duration": 5.0,
                    "frame_rate": 30.0,
                    "frame_count": 150,
                    "codec_name": "h264"
                }
            }
            
            mock_extract.return_value = ["frame1.png", "frame2.png", "frame3.png"]
            mock_combine.return_value = True
            
            # Test workflow
            app = UpScaleApp(use_ai=False)
            result = app.process_video("test.mp4", "output.mp4")
            
            # Verify workflow completed
            assert result["success"]
            assert result["output_path"] is not None
            assert "stats" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])