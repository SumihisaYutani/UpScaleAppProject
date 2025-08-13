"""
Plugin System for UpScale App
Extensible architecture for adding custom processing capabilities
"""

import os
import sys
import json
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from enum import Enum

from config.settings import PATHS

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Types of plugins"""
    PROCESSOR = "processor"      # Custom processing algorithms
    FILTER = "filter"           # Video/image filters
    EXPORT = "export"           # Export format handlers
    AI_MODEL = "ai_model"       # AI model integrations
    UI_COMPONENT = "ui_component"  # UI extensions


@dataclass
class PluginInfo:
    """Plugin metadata and information"""
    id: str
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    settings_schema: Dict = field(default_factory=dict)
    enabled: bool = True
    file_path: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "plugin_type": self.plugin_type.value,
            "dependencies": self.dependencies,
            "requirements": self.requirements,
            "settings_schema": self.settings_schema,
            "enabled": self.enabled,
            "file_path": self.file_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginInfo':
        """Create from dictionary"""
        return cls(
            id=data["id"],
            name=data["name"],
            version=data["version"],
            description=data["description"],
            author=data["author"],
            plugin_type=PluginType(data["plugin_type"]),
            dependencies=data.get("dependencies", []),
            requirements=data.get("requirements", []),
            settings_schema=data.get("settings_schema", {}),
            enabled=data.get("enabled", True),
            file_path=data.get("file_path", "")
        )


class PluginInterface(ABC):
    """Base interface for all plugins"""
    
    def __init__(self):
        self.info: Optional[PluginInfo] = None
        self.settings: Dict[str, Any] = {}
        self.enabled: bool = True
    
    @abstractmethod
    def get_info(self) -> PluginInfo:
        """Get plugin information"""
        pass
    
    def initialize(self, settings: Dict[str, Any] = None):
        """Initialize the plugin with settings"""
        if settings:
            self.settings.update(settings)
        logger.info(f"Initialized plugin: {self.info.name if self.info else 'Unknown'}")
    
    def cleanup(self):
        """Cleanup plugin resources"""
        logger.info(f"Cleaned up plugin: {self.info.name if self.info else 'Unknown'}")
    
    def get_settings_ui(self) -> Dict[str, Any]:
        """Get settings UI definition"""
        return {}
    
    def validate_settings(self, settings: Dict[str, Any]) -> tuple[bool, str]:
        """Validate plugin settings"""
        return True, ""


class ProcessorPlugin(PluginInterface):
    """Base class for processing plugins"""
    
    @abstractmethod
    def process_frame(self, frame_data: Any, **kwargs) -> Any:
        """Process a single frame"""
        pass
    
    def process_batch(self, frames: List[Any], **kwargs) -> List[Any]:
        """Process multiple frames (default implementation)"""
        return [self.process_frame(frame, **kwargs) for frame in frames]
    
    def get_supported_formats(self) -> List[str]:
        """Get supported input formats"""
        return [".png", ".jpg", ".jpeg"]


class FilterPlugin(PluginInterface):
    """Base class for filter plugins"""
    
    @abstractmethod
    def apply_filter(self, image_data: Any, **kwargs) -> Any:
        """Apply filter to image data"""
        pass
    
    def get_filter_parameters(self) -> Dict[str, Any]:
        """Get filter parameters schema"""
        return {}


class ExportPlugin(PluginInterface):
    """Base class for export plugins"""
    
    @abstractmethod
    def export(self, data: Any, output_path: str, **kwargs) -> bool:
        """Export data to specified format"""
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions"""
        pass
    
    def get_export_options(self) -> Dict[str, Any]:
        """Get export options schema"""
        return {}


class AIModelPlugin(PluginInterface):
    """Base class for AI model plugins"""
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the AI model"""
        pass
    
    @abstractmethod
    def unload_model(self):
        """Unload the AI model"""
        pass
    
    @abstractmethod
    def upscale_image(self, image_data: Any, scale_factor: float, **kwargs) -> Any:
        """Upscale image using AI model"""
        pass
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {}


class PluginManager:
    """Manages plugin loading, registration, and execution"""
    
    def __init__(self):
        self.plugins_dir = Path("plugins")
        self.plugins_dir.mkdir(exist_ok=True)
        
        # Plugin registry
        self.registered_plugins: Dict[str, PluginInterface] = {}
        self.plugin_info: Dict[str, PluginInfo] = {}
        self.enabled_plugins: Dict[str, bool] = {}
        
        # Plugin hooks
        self.hooks: Dict[str, List[Callable]] = {}
        
        # Settings
        self.plugin_settings_file = PATHS["logs_dir"] / "plugin_settings.json"
        self.plugin_settings: Dict[str, Dict[str, Any]] = {}
        
        self.load_plugin_settings()
    
    def scan_plugins(self) -> List[PluginInfo]:
        """Scan for available plugins"""
        found_plugins = []
        
        # Scan plugins directory
        for plugin_file in self.plugins_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
                
            try:
                plugin_info = self._load_plugin_info(plugin_file)
                if plugin_info:
                    found_plugins.append(plugin_info)
            except Exception as e:
                logger.warning(f"Failed to scan plugin {plugin_file}: {e}")
        
        # Scan subdirectories
        for plugin_dir in self.plugins_dir.iterdir():
            if not plugin_dir.is_dir() or plugin_dir.name.startswith("_"):
                continue
                
            plugin_file = plugin_dir / "plugin.py"
            if plugin_file.exists():
                try:
                    plugin_info = self._load_plugin_info(plugin_file)
                    if plugin_info:
                        found_plugins.append(plugin_info)
                except Exception as e:
                    logger.warning(f"Failed to scan plugin {plugin_dir}: {e}")
        
        return found_plugins
    
    def _load_plugin_info(self, plugin_file: Path) -> Optional[PluginInfo]:
        """Load plugin info from file"""
        try:
            # Try to load plugin metadata first
            metadata_file = plugin_file.parent / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    info = PluginInfo.from_dict(metadata)
                    info.file_path = str(plugin_file)
                    return info
            
            # Fallback: load plugin module and extract info
            spec = importlib.util.spec_from_file_location(
                f"plugin_{plugin_file.stem}", plugin_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, PluginInterface) and 
                    obj != PluginInterface):
                    plugin_class = obj
                    break
            
            if plugin_class:
                # Instantiate temporarily to get info
                temp_plugin = plugin_class()
                info = temp_plugin.get_info()
                info.file_path = str(plugin_file)
                temp_plugin.cleanup()
                return info
                
        except Exception as e:
            logger.error(f"Failed to load plugin info from {plugin_file}: {e}")
        
        return None
    
    def load_plugin(self, plugin_id: str) -> bool:
        """Load a specific plugin"""
        if plugin_id in self.registered_plugins:
            logger.info(f"Plugin {plugin_id} already loaded")
            return True
        
        # Find plugin info
        if plugin_id not in self.plugin_info:
            # Rescan for plugins
            available_plugins = self.scan_plugins()
            for plugin_info in available_plugins:
                self.plugin_info[plugin_info.id] = plugin_info
            
            if plugin_id not in self.plugin_info:
                logger.error(f"Plugin {plugin_id} not found")
                return False
        
        plugin_info = self.plugin_info[plugin_id]
        
        try:
            # Check dependencies
            if not self._check_dependencies(plugin_info):
                logger.error(f"Plugin {plugin_id} dependencies not met")
                return False
            
            # Load plugin module
            spec = importlib.util.spec_from_file_location(
                f"plugin_{plugin_id}", plugin_info.file_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find and instantiate plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, PluginInterface) and 
                    obj != PluginInterface):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                logger.error(f"No valid plugin class found in {plugin_info.file_path}")
                return False
            
            # Instantiate plugin
            plugin = plugin_class()
            plugin.info = plugin_info
            
            # Initialize with settings
            plugin_settings = self.plugin_settings.get(plugin_id, {})
            plugin.initialize(plugin_settings)
            
            # Register plugin
            self.registered_plugins[plugin_id] = plugin
            self.enabled_plugins[plugin_id] = plugin_info.enabled
            
            logger.info(f"Successfully loaded plugin: {plugin_info.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_id}: {e}")
            return False
    
    def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a specific plugin"""
        if plugin_id not in self.registered_plugins:
            return False
        
        try:
            plugin = self.registered_plugins[plugin_id]
            plugin.cleanup()
            del self.registered_plugins[plugin_id]
            
            if plugin_id in self.enabled_plugins:
                del self.enabled_plugins[plugin_id]
            
            logger.info(f"Unloaded plugin: {plugin_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_id}: {e}")
            return False
    
    def enable_plugin(self, plugin_id: str) -> bool:
        """Enable a plugin"""
        if plugin_id not in self.registered_plugins:
            if not self.load_plugin(plugin_id):
                return False
        
        self.enabled_plugins[plugin_id] = True
        if plugin_id in self.plugin_info:
            self.plugin_info[plugin_id].enabled = True
        
        self.save_plugin_settings()
        logger.info(f"Enabled plugin: {plugin_id}")
        return True
    
    def disable_plugin(self, plugin_id: str) -> bool:
        """Disable a plugin"""
        self.enabled_plugins[plugin_id] = False
        if plugin_id in self.plugin_info:
            self.plugin_info[plugin_id].enabled = False
        
        self.save_plugin_settings()
        logger.info(f"Disabled plugin: {plugin_id}")
        return True
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInterface]:
        """Get all enabled plugins of a specific type"""
        plugins = []
        for plugin_id, plugin in self.registered_plugins.items():
            if (self.enabled_plugins.get(plugin_id, False) and
                plugin.info and 
                plugin.info.plugin_type == plugin_type):
                plugins.append(plugin)
        return plugins
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginInterface]:
        """Get a specific plugin"""
        return self.registered_plugins.get(plugin_id)
    
    def is_plugin_enabled(self, plugin_id: str) -> bool:
        """Check if a plugin is enabled"""
        return self.enabled_plugins.get(plugin_id, False)
    
    def get_available_plugins(self) -> List[PluginInfo]:
        """Get list of all available plugins"""
        return list(self.plugin_info.values())
    
    def get_loaded_plugins(self) -> List[PluginInfo]:
        """Get list of loaded plugins"""
        return [plugin.info for plugin in self.registered_plugins.values() 
                if plugin.info]
    
    def _check_dependencies(self, plugin_info: PluginInfo) -> bool:
        """Check if plugin dependencies are met"""
        for dependency in plugin_info.dependencies:
            if dependency not in self.registered_plugins:
                logger.warning(f"Plugin dependency not met: {dependency}")
                return False
        
        # Check Python package requirements
        for requirement in plugin_info.requirements:
            try:
                __import__(requirement)
            except ImportError:
                logger.warning(f"Plugin requirement not met: {requirement}")
                return False
        
        return True
    
    def register_hook(self, hook_name: str, callback: Callable):
        """Register a hook callback"""
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)
    
    def unregister_hook(self, hook_name: str, callback: Callable):
        """Unregister a hook callback"""
        if hook_name in self.hooks and callback in self.hooks[hook_name]:
            self.hooks[hook_name].remove(callback)
    
    def call_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Call all callbacks for a hook"""
        results = []
        for callback in self.hooks.get(hook_name, []):
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook callback error in {hook_name}: {e}")
        return results
    
    def set_plugin_setting(self, plugin_id: str, key: str, value: Any):
        """Set a plugin setting"""
        if plugin_id not in self.plugin_settings:
            self.plugin_settings[plugin_id] = {}
        
        self.plugin_settings[plugin_id][key] = value
        
        # Update plugin if loaded
        if plugin_id in self.registered_plugins:
            plugin = self.registered_plugins[plugin_id]
            plugin.settings[key] = value
        
        self.save_plugin_settings()
    
    def get_plugin_setting(self, plugin_id: str, key: str, default: Any = None) -> Any:
        """Get a plugin setting"""
        return self.plugin_settings.get(plugin_id, {}).get(key, default)
    
    def load_plugin_settings(self):
        """Load plugin settings from file"""
        try:
            if self.plugin_settings_file.exists():
                with open(self.plugin_settings_file, "r") as f:
                    self.plugin_settings = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load plugin settings: {e}")
            self.plugin_settings = {}
    
    def save_plugin_settings(self):
        """Save plugin settings to file"""
        try:
            self.plugin_settings_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.plugin_settings_file, "w") as f:
                json.dump(self.plugin_settings, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save plugin settings: {e}")
    
    def create_plugin_template(self, plugin_name: str, plugin_type: PluginType,
                              output_dir: Path = None) -> str:
        """Create a plugin template"""
        if output_dir is None:
            output_dir = self.plugins_dir
        
        plugin_dir = output_dir / plugin_name.lower().replace(" ", "_")
        plugin_dir.mkdir(exist_ok=True)
        
        # Create plugin.py
        template_code = self._get_plugin_template(plugin_name, plugin_type)
        with open(plugin_dir / "plugin.py", "w") as f:
            f.write(template_code)
        
        # Create metadata.json
        metadata = {
            "id": plugin_name.lower().replace(" ", "_"),
            "name": plugin_name,
            "version": "1.0.0",
            "description": f"Custom {plugin_type.value} plugin",
            "author": "User",
            "plugin_type": plugin_type.value,
            "dependencies": [],
            "requirements": [],
            "settings_schema": {},
            "enabled": True
        }
        
        with open(plugin_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Create README.md
        readme_content = f"""# {plugin_name} Plugin

## Description
{metadata['description']}

## Installation
1. Copy this folder to the plugins directory
2. Restart the application
3. Enable the plugin in settings

## Configuration
Edit the settings in the plugin manager.

## Development
Modify plugin.py to implement your custom functionality.
"""
        
        with open(plugin_dir / "README.md", "w") as f:
            f.write(readme_content)
        
        return str(plugin_dir)
    
    def _get_plugin_template(self, plugin_name: str, plugin_type: PluginType) -> str:
        """Get plugin template code"""
        base_class = {
            PluginType.PROCESSOR: "ProcessorPlugin",
            PluginType.FILTER: "FilterPlugin", 
            PluginType.EXPORT: "ExportPlugin",
            PluginType.AI_MODEL: "AIModelPlugin"
        }.get(plugin_type, "PluginInterface")
        
        template = f'''"""
{plugin_name} Plugin
Custom {plugin_type.value} implementation
"""

import numpy as np
from typing import Any, Dict, List
from plugins.plugin_system import {base_class}, PluginInfo, PluginType


class {plugin_name.replace(" ", "")}Plugin({base_class}):
    """Custom {plugin_name} plugin"""
    
    def get_info(self) -> PluginInfo:
        """Get plugin information"""
        return PluginInfo(
            id="{plugin_name.lower().replace(' ', '_')}",
            name="{plugin_name}",
            version="1.0.0",
            description="Custom {plugin_type.value} plugin",
            author="User",
            plugin_type=PluginType.{plugin_type.name}
        )
    
    def initialize(self, settings: Dict[str, Any] = None):
        """Initialize the plugin"""
        super().initialize(settings)
        # Add your initialization code here
    
    def cleanup(self):
        """Cleanup plugin resources"""
        super().cleanup()
        # Add your cleanup code here
'''
        
        if plugin_type == PluginType.PROCESSOR:
            template += '''
    def process_frame(self, frame_data: Any, **kwargs) -> Any:
        """Process a single frame"""
        # Implement your frame processing logic here
        # Example: apply some transformation to the frame
        return frame_data
'''
        elif plugin_type == PluginType.FILTER:
            template += '''
    def apply_filter(self, image_data: Any, **kwargs) -> Any:
        """Apply filter to image data"""
        # Implement your filter logic here
        return image_data
    
    def get_filter_parameters(self) -> Dict[str, Any]:
        """Get filter parameters schema"""
        return {
            "strength": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "mode": {"type": "choice", "choices": ["mode1", "mode2"], "default": "mode1"}
        }
'''
        elif plugin_type == PluginType.EXPORT:
            template += '''
    def export(self, data: Any, output_path: str, **kwargs) -> bool:
        """Export data to specified format"""
        # Implement your export logic here
        try:
            # Your export code here
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions"""
        return [".custom"]
    
    def get_export_options(self) -> Dict[str, Any]:
        """Get export options schema"""
        return {
            "quality": {"type": "int", "min": 1, "max": 100, "default": 85}
        }
'''
        elif plugin_type == PluginType.AI_MODEL:
            template += '''
    def load_model(self) -> bool:
        """Load the AI model"""
        try:
            # Your model loading code here
            return True
        except Exception as e:
            print(f"Model loading failed: {e}")
            return False
    
    def unload_model(self):
        """Unload the AI model"""
        # Your model unloading code here
        pass
    
    def upscale_image(self, image_data: Any, scale_factor: float, **kwargs) -> Any:
        """Upscale image using AI model"""
        # Your AI upscaling logic here
        return image_data
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        # Return True if your model is loaded
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": "Custom Model",
            "version": "1.0",
            "input_size": "any",
            "output_size": "scaled"
        }
'''
        
        return template
    
    def shutdown(self):
        """Shutdown plugin manager"""
        logger.info("Shutting down plugin manager...")
        
        # Unload all plugins
        for plugin_id in list(self.registered_plugins.keys()):
            self.unload_plugin(plugin_id)
        
        # Save settings
        self.save_plugin_settings()
        
        logger.info("Plugin manager shutdown complete")


# Global plugin manager instance
plugin_manager = PluginManager()