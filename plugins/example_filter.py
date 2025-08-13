"""
Example Filter Plugin
Demonstrates how to create a custom filter plugin
"""

import numpy as np
from typing import Any, Dict
import cv2

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from plugins.plugin_system import FilterPlugin, PluginInfo, PluginType


class SharpnessFilterPlugin(FilterPlugin):
    """Sharpness enhancement filter plugin"""
    
    def get_info(self) -> PluginInfo:
        """Get plugin information"""
        return PluginInfo(
            id="sharpness_filter",
            name="Sharpness Filter",
            version="1.0.0",
            description="Enhances image sharpness using unsharp masking",
            author="UpScale Team",
            plugin_type=PluginType.FILTER,
            settings_schema={
                "strength": {
                    "type": "float",
                    "min": 0.0,
                    "max": 2.0,
                    "default": 0.5,
                    "description": "Filter strength"
                },
                "radius": {
                    "type": "float", 
                    "min": 0.5,
                    "max": 5.0,
                    "default": 1.0,
                    "description": "Blur radius for unsharp mask"
                },
                "threshold": {
                    "type": "int",
                    "min": 0,
                    "max": 255,
                    "default": 3,
                    "description": "Threshold for edge detection"
                }
            }
        )
    
    def initialize(self, settings: Dict[str, Any] = None):
        """Initialize the plugin"""
        super().initialize(settings)
        
        # Set default values
        self.strength = self.settings.get("strength", 0.5)
        self.radius = self.settings.get("radius", 1.0)
        self.threshold = self.settings.get("threshold", 3)
        
        print(f"Sharpness Filter initialized: strength={self.strength}, radius={self.radius}")
    
    def apply_filter(self, image_data: Any, **kwargs) -> Any:
        """Apply sharpness filter to image data"""
        
        if isinstance(image_data, str):
            # Load image from path
            image = cv2.imread(image_data)
        elif isinstance(image_data, np.ndarray):
            image = image_data.copy()
        else:
            raise ValueError("Unsupported image data type")
        
        if image is None:
            raise ValueError("Could not load image")
        
        # Apply unsharp masking
        # Create Gaussian blur
        gaussian = cv2.GaussianBlur(image, (0, 0), self.radius)
        
        # Create unsharp mask
        unsharp_mask = cv2.addWeighted(image, 1.0 + self.strength, gaussian, -self.strength, 0)
        
        # Apply threshold to avoid noise amplification
        if self.threshold > 0:
            diff = cv2.absdiff(image, gaussian)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            mask = cv2.threshold(gray_diff, self.threshold, 255, cv2.THRESH_BINARY)[1]
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            
            # Blend based on mask
            result = image * (1 - mask) + unsharp_mask * mask
            result = np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = np.clip(unsharp_mask, 0, 255).astype(np.uint8)
        
        return result
    
    def get_filter_parameters(self) -> Dict[str, Any]:
        """Get filter parameters schema"""
        return {
            "strength": {
                "type": "float",
                "min": 0.0,
                "max": 2.0,
                "default": 0.5,
                "step": 0.1,
                "description": "Sharpening strength"
            },
            "radius": {
                "type": "float",
                "min": 0.5,
                "max": 5.0,
                "default": 1.0,
                "step": 0.1,
                "description": "Gaussian blur radius"
            },
            "threshold": {
                "type": "int",
                "min": 0,
                "max": 255,
                "default": 3,
                "step": 1,
                "description": "Edge detection threshold"
            }
        }
    
    def get_settings_ui(self) -> Dict[str, Any]:
        """Get settings UI definition"""
        return {
            "type": "form",
            "fields": [
                {
                    "name": "strength",
                    "label": "Strength",
                    "type": "slider",
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "default": 0.5
                },
                {
                    "name": "radius", 
                    "label": "Radius",
                    "type": "slider",
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "default": 1.0
                },
                {
                    "name": "threshold",
                    "label": "Threshold",
                    "type": "slider",
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "default": 3
                }
            ]
        }
    
    def validate_settings(self, settings: Dict[str, Any]) -> tuple[bool, str]:
        """Validate plugin settings"""
        
        # Check strength
        strength = settings.get("strength", 0.5)
        if not isinstance(strength, (int, float)) or strength < 0 or strength > 2.0:
            return False, "Strength must be between 0.0 and 2.0"
        
        # Check radius
        radius = settings.get("radius", 1.0)
        if not isinstance(radius, (int, float)) or radius < 0.5 or radius > 5.0:
            return False, "Radius must be between 0.5 and 5.0"
        
        # Check threshold
        threshold = settings.get("threshold", 3)
        if not isinstance(threshold, int) or threshold < 0 or threshold > 255:
            return False, "Threshold must be between 0 and 255"
        
        return True, ""