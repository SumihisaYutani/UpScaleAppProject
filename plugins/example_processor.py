"""
Example Processor Plugin
Demonstrates how to create a custom processing plugin
"""

import numpy as np
from typing import Any, List, Dict
import cv2
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from plugins.plugin_system import ProcessorPlugin, PluginInfo, PluginType


class NoiseReductionProcessor(ProcessorPlugin):
    """Noise reduction processor plugin"""
    
    def get_info(self) -> PluginInfo:
        """Get plugin information"""
        return PluginInfo(
            id="noise_reduction_processor",
            name="Noise Reduction Processor",
            version="1.0.0",
            description="Reduces noise in images using advanced filtering techniques",
            author="UpScale Team",
            plugin_type=PluginType.PROCESSOR,
            requirements=["opencv-python"],
            settings_schema={
                "method": {
                    "type": "choice",
                    "choices": ["gaussian", "bilateral", "non_local_means"],
                    "default": "bilateral",
                    "description": "Noise reduction method"
                },
                "strength": {
                    "type": "float",
                    "min": 0.1,
                    "max": 2.0,
                    "default": 1.0,
                    "description": "Processing strength"
                }
            }
        )
    
    def initialize(self, settings: Dict[str, Any] = None):
        """Initialize the plugin"""
        super().initialize(settings)
        
        self.method = self.settings.get("method", "bilateral")
        self.strength = self.settings.get("strength", 1.0)
        
        print(f"Noise Reduction Processor initialized: method={self.method}, strength={self.strength}")
    
    def process_frame(self, frame_data: Any, **kwargs) -> Any:
        """Process a single frame"""
        
        if isinstance(frame_data, str):
            # Load image from path
            image = cv2.imread(frame_data)
        elif isinstance(frame_data, np.ndarray):
            image = frame_data.copy()
        else:
            raise ValueError("Unsupported frame data type")
        
        if image is None:
            raise ValueError("Could not load image")
        
        # Apply noise reduction based on selected method
        if self.method == "gaussian":
            # Gaussian blur with strength-based kernel size
            kernel_size = int(3 + self.strength * 4)
            if kernel_size % 2 == 0:
                kernel_size += 1
            result = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
        elif self.method == "bilateral":
            # Bilateral filter
            d = int(5 + self.strength * 10)  # Diameter
            sigma_color = self.strength * 80
            sigma_space = self.strength * 80
            result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            
        elif self.method == "non_local_means":
            # Non-local means denoising
            h = self.strength * 10  # Filter strength
            template_window_size = 7
            search_window_size = 21
            
            # Apply to each channel separately for color images
            if len(image.shape) == 3:
                result = cv2.fastNlMeansDenoisingColored(
                    image, None, h, h, template_window_size, search_window_size
                )
            else:
                result = cv2.fastNlMeansDenoising(
                    image, None, h, template_window_size, search_window_size
                )
        else:
            # Fallback to original
            result = image
        
        return result
    
    def process_batch(self, frames: List[Any], **kwargs) -> List[Any]:
        """Process multiple frames with optimizations"""
        
        results = []
        total_frames = len(frames)
        
        print(f"Processing batch of {total_frames} frames with {self.method} method")
        
        start_time = time.time()
        
        for i, frame in enumerate(frames):
            try:
                processed_frame = self.process_frame(frame, **kwargs)
                results.append(processed_frame)
                
                # Progress reporting
                if i % 10 == 0:
                    progress = (i + 1) / total_frames * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / (i + 1)) * (total_frames - i - 1)
                    print(f"Progress: {progress:.1f}% ({i+1}/{total_frames}), ETA: {eta:.1f}s")
                    
            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                # Use original frame if processing fails
                results.append(frame)
        
        processing_time = time.time() - start_time
        print(f"Batch processing completed in {processing_time:.2f}s")
        
        return results
    
    def get_supported_formats(self) -> List[str]:
        """Get supported input formats"""
        return [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
    
    def get_settings_ui(self) -> Dict[str, Any]:
        """Get settings UI definition"""
        return {
            "type": "form",
            "fields": [
                {
                    "name": "method",
                    "label": "Noise Reduction Method",
                    "type": "select",
                    "options": [
                        {"value": "gaussian", "label": "Gaussian Blur"},
                        {"value": "bilateral", "label": "Bilateral Filter"},
                        {"value": "non_local_means", "label": "Non-Local Means"}
                    ],
                    "default": "bilateral"
                },
                {
                    "name": "strength",
                    "label": "Processing Strength",
                    "type": "slider",
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "default": 1.0
                }
            ]
        }
    
    def validate_settings(self, settings: Dict[str, Any]) -> tuple[bool, str]:
        """Validate plugin settings"""
        
        # Check method
        method = settings.get("method", "bilateral")
        valid_methods = ["gaussian", "bilateral", "non_local_means"]
        if method not in valid_methods:
            return False, f"Method must be one of: {', '.join(valid_methods)}"
        
        # Check strength
        strength = settings.get("strength", 1.0)
        if not isinstance(strength, (int, float)) or strength < 0.1 or strength > 2.0:
            return False, "Strength must be between 0.1 and 2.0"
        
        return True, ""
    
    def cleanup(self):
        """Cleanup plugin resources"""
        super().cleanup()
        print("Noise Reduction Processor cleaned up")