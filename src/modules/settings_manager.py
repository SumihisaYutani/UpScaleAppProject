"""
Advanced Settings Management System
Handles user preferences, profiles, and configuration management
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from copy import deepcopy

from config.settings import PATHS, VIDEO_SETTINGS, AI_SETTINGS, PERFORMANCE

logger = logging.getLogger(__name__)


class SettingType(Enum):
    """Types of settings"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    CHOICE = "choice"
    RANGE = "range"
    FILE_PATH = "file_path"
    FOLDER_PATH = "folder_path"
    COLOR = "color"


@dataclass
class SettingDefinition:
    """Definition of a setting parameter"""
    key: str
    name: str
    description: str
    setting_type: SettingType
    default_value: Any
    category: str
    subcategory: str = ""
    choices: List[Any] = field(default_factory=list)
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    step: Optional[Union[int, float]] = None
    validation_func: Optional[Callable] = None
    requires_restart: bool = False
    advanced: bool = False
    
    def validate(self, value: Any) -> tuple[bool, str]:
        """Validate a value for this setting"""
        try:
            # Type validation
            if self.setting_type == SettingType.BOOLEAN:
                if not isinstance(value, bool):
                    return False, "Must be a boolean value"
            elif self.setting_type == SettingType.INTEGER:
                if not isinstance(value, int):
                    return False, "Must be an integer"
                if self.min_value is not None and value < self.min_value:
                    return False, f"Must be >= {self.min_value}"
                if self.max_value is not None and value > self.max_value:
                    return False, f"Must be <= {self.max_value}"
            elif self.setting_type == SettingType.FLOAT:
                if not isinstance(value, (int, float)):
                    return False, "Must be a number"
                if self.min_value is not None and value < self.min_value:
                    return False, f"Must be >= {self.min_value}"
                if self.max_value is not None and value > self.max_value:
                    return False, f"Must be <= {self.max_value}"
            elif self.setting_type == SettingType.STRING:
                if not isinstance(value, str):
                    return False, "Must be a string"
            elif self.setting_type == SettingType.CHOICE:
                if value not in self.choices:
                    return False, f"Must be one of: {', '.join(map(str, self.choices))}"
            elif self.setting_type in [SettingType.FILE_PATH, SettingType.FOLDER_PATH]:
                if not isinstance(value, str):
                    return False, "Must be a valid path"
                path = Path(value)
                if self.setting_type == SettingType.FILE_PATH and value and not path.exists():
                    return False, "File does not exist"
                elif self.setting_type == SettingType.FOLDER_PATH and value and not path.is_dir():
                    return False, "Directory does not exist"
            
            # Custom validation
            if self.validation_func:
                is_valid, msg = self.validation_func(value)
                if not is_valid:
                    return False, msg
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"


@dataclass
class SettingProfile:
    """A collection of settings (profile/preset)"""
    name: str
    description: str
    settings: Dict[str, Any] = field(default_factory=dict)
    is_builtin: bool = False
    created_at: float = field(default_factory=lambda: __import__('time').time())
    modified_at: float = field(default_factory=lambda: __import__('time').time())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SettingProfile':
        """Create from dictionary"""
        return cls(**data)


class AdvancedSettingsManager:
    """Advanced settings management system"""
    
    def __init__(self):
        self.settings_file = PATHS["logs_dir"] / "user_settings.json"
        self.profiles_file = PATHS["logs_dir"] / "settings_profiles.json"
        
        # Setting definitions
        self._setting_definitions: Dict[str, SettingDefinition] = {}
        self._initialize_setting_definitions()
        
        # Current settings and profiles
        self.current_settings: Dict[str, Any] = {}
        self.profiles: Dict[str, SettingProfile] = {}
        self.current_profile: str = "default"
        
        # Change tracking
        self.change_callbacks: List[Callable] = []
        self.pending_restart_settings: set = set()
        
        # Load settings
        self.load_settings()
        self.load_profiles()
        self._create_builtin_profiles()
    
    def _initialize_setting_definitions(self):
        """Initialize all setting definitions"""
        
        # Video Processing Settings
        self._add_setting_definition(SettingDefinition(
            key="video.supported_formats",
            name="Supported Formats",
            description="List of supported video formats",
            setting_type=SettingType.CHOICE,
            default_value=VIDEO_SETTINGS["supported_formats"],
            category="Video Processing",
            choices=[[".mp4"], [".mp4", ".avi"], [".mp4", ".avi", ".mov"]],
            advanced=True
        ))
        
        self._add_setting_definition(SettingDefinition(
            key="video.max_file_size_gb",
            name="Maximum File Size (GB)",
            description="Maximum allowed file size for processing",
            setting_type=SettingType.FLOAT,
            default_value=VIDEO_SETTINGS["max_file_size_gb"],
            category="Video Processing",
            min_value=0.1,
            max_value=50.0,
            step=0.1
        ))
        
        self._add_setting_definition(SettingDefinition(
            key="video.max_duration_minutes",
            name="Maximum Duration (minutes)",
            description="Maximum allowed video duration",
            setting_type=SettingType.INTEGER,
            default_value=VIDEO_SETTINGS["max_duration_minutes"],
            category="Video Processing",
            min_value=1,
            max_value=180,
            step=1
        ))
        
        self._add_setting_definition(SettingDefinition(
            key="video.default_upscale_factor",
            name="Default Upscale Factor",
            description="Default scaling factor for video upscaling",
            setting_type=SettingType.CHOICE,
            default_value=VIDEO_SETTINGS["default_upscale_factor"],
            category="Video Processing",
            choices=[1.2, 1.5, 2.0, 2.5, 3.0]
        ))
        
        # AI Processing Settings
        self._add_setting_definition(SettingDefinition(
            key="ai.model_name",
            name="AI Model",
            description="Stable Diffusion model to use for processing",
            setting_type=SettingType.CHOICE,
            default_value=AI_SETTINGS["model_name"],
            category="AI Processing",
            choices=[
                "stabilityai/stable-diffusion-2-1",
                "runwayml/stable-diffusion-v1-5",
                "stabilityai/stable-diffusion-xl-base-1.0"
            ],
            requires_restart=True
        ))
        
        self._add_setting_definition(SettingDefinition(
            key="ai.batch_size",
            name="Batch Size",
            description="Number of frames processed simultaneously",
            setting_type=SettingType.INTEGER,
            default_value=AI_SETTINGS["batch_size"],
            category="AI Processing",
            min_value=1,
            max_value=16,
            step=1
        ))
        
        self._add_setting_definition(SettingDefinition(
            key="ai.guidance_scale",
            name="Guidance Scale",
            description="How closely to follow the text prompt",
            setting_type=SettingType.FLOAT,
            default_value=AI_SETTINGS["guidance_scale"],
            category="AI Processing",
            min_value=1.0,
            max_value=20.0,
            step=0.5
        ))
        
        self._add_setting_definition(SettingDefinition(
            key="ai.num_inference_steps",
            name="Inference Steps",
            description="Number of denoising steps (higher = better quality)",
            setting_type=SettingType.INTEGER,
            default_value=AI_SETTINGS["num_inference_steps"],
            category="AI Processing",
            min_value=5,
            max_value=50,
            step=1
        ))
        
        self._add_setting_definition(SettingDefinition(
            key="ai.device",
            name="Processing Device",
            description="Device to use for AI processing",
            setting_type=SettingType.CHOICE,
            default_value=AI_SETTINGS["device"],
            category="AI Processing",
            choices=["auto", "cpu", "cuda"],
            requires_restart=True
        ))
        
        # Performance Settings
        self._add_setting_definition(SettingDefinition(
            key="performance.max_memory_gb",
            name="Maximum Memory Usage (GB)",
            description="Maximum RAM usage for processing",
            setting_type=SettingType.FLOAT,
            default_value=PERFORMANCE["max_memory_gb"],
            category="Performance",
            min_value=2.0,
            max_value=64.0,
            step=1.0
        ))
        
        self._add_setting_definition(SettingDefinition(
            key="performance.max_concurrent_frames",
            name="Max Concurrent Frames",
            description="Maximum frames processed concurrently",
            setting_type=SettingType.INTEGER,
            default_value=PERFORMANCE["max_concurrent_frames"],
            category="Performance",
            min_value=1,
            max_value=50,
            step=1
        ))
        
        self._add_setting_definition(SettingDefinition(
            key="performance.cleanup_temp_files",
            name="Cleanup Temporary Files",
            description="Automatically clean up temporary files after processing",
            setting_type=SettingType.BOOLEAN,
            default_value=PERFORMANCE["cleanup_temp_files"],
            category="Performance"
        ))
        
        # UI/UX Settings
        self._add_setting_definition(SettingDefinition(
            key="ui.theme",
            name="Theme",
            description="Application color theme",
            setting_type=SettingType.CHOICE,
            default_value="dark",
            category="User Interface",
            choices=["light", "dark", "system"],
            requires_restart=True
        ))
        
        self._add_setting_definition(SettingDefinition(
            key="ui.language",
            name="Language",
            description="Application language",
            setting_type=SettingType.CHOICE,
            default_value="en",
            category="User Interface",
            choices=["en", "ja", "es", "fr", "de"],
            requires_restart=True
        ))
        
        self._add_setting_definition(SettingDefinition(
            key="ui.auto_save_settings",
            name="Auto-save Settings",
            description="Automatically save settings when changed",
            setting_type=SettingType.BOOLEAN,
            default_value=True,
            category="User Interface"
        ))
        
        self._add_setting_definition(SettingDefinition(
            key="ui.show_advanced_options",
            name="Show Advanced Options",
            description="Show advanced settings in the UI",
            setting_type=SettingType.BOOLEAN,
            default_value=False,
            category="User Interface"
        ))
        
        # File Paths
        self._add_setting_definition(SettingDefinition(
            key="paths.default_output_folder",
            name="Default Output Folder",
            description="Default folder for processed videos",
            setting_type=SettingType.FOLDER_PATH,
            default_value=str(PATHS["output_dir"]),
            category="File Paths"
        ))
        
        self._add_setting_definition(SettingDefinition(
            key="paths.temp_folder",
            name="Temporary Folder",
            description="Folder for temporary processing files",
            setting_type=SettingType.FOLDER_PATH,
            default_value=str(PATHS["temp_dir"]),
            category="File Paths",
            requires_restart=True
        ))
        
        # Advanced/Expert Settings
        self._add_setting_definition(SettingDefinition(
            key="expert.enable_debug_logging",
            name="Enable Debug Logging",
            description="Enable detailed debug logging",
            setting_type=SettingType.BOOLEAN,
            default_value=False,
            category="Expert Settings",
            advanced=True,
            requires_restart=True
        ))
        
        self._add_setting_definition(SettingDefinition(
            key="expert.custom_ffmpeg_args",
            name="Custom FFmpeg Arguments",
            description="Additional FFmpeg arguments for video processing",
            setting_type=SettingType.STRING,
            default_value="",
            category="Expert Settings",
            advanced=True
        ))
    
    def _add_setting_definition(self, definition: SettingDefinition):
        """Add a setting definition"""
        self._setting_definitions[definition.key] = definition
    
    def get_setting_definitions(self, category: str = None, 
                               advanced: bool = None) -> List[SettingDefinition]:
        """Get setting definitions, optionally filtered"""
        definitions = list(self._setting_definitions.values())
        
        if category:
            definitions = [d for d in definitions if d.category == category]
        
        if advanced is not None:
            definitions = [d for d in definitions if d.advanced == advanced]
        
        return sorted(definitions, key=lambda x: (x.category, x.subcategory, x.name))
    
    def get_categories(self) -> List[str]:
        """Get all setting categories"""
        categories = set(d.category for d in self._setting_definitions.values())
        return sorted(list(categories))
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value"""
        return self.current_settings.get(key, default)
    
    def set_setting(self, key: str, value: Any, validate: bool = True) -> bool:
        """Set a setting value"""
        
        if key not in self._setting_definitions:
            logger.warning(f"Unknown setting key: {key}")
            return False
        
        definition = self._setting_definitions[key]
        
        # Validate if requested
        if validate:
            is_valid, error_msg = definition.validate(value)
            if not is_valid:
                logger.error(f"Invalid value for {key}: {error_msg}")
                return False
        
        # Set the value
        old_value = self.current_settings.get(key)
        self.current_settings[key] = value
        
        # Track restart requirements
        if definition.requires_restart and old_value != value:
            self.pending_restart_settings.add(key)
        
        # Auto-save if enabled
        if self.get_setting("ui.auto_save_settings", True):
            self.save_settings()
        
        # Notify callbacks
        self._notify_change_callbacks(key, old_value, value)
        
        logger.debug(f"Setting {key} changed: {old_value} -> {value}")
        return True
    
    def reset_setting(self, key: str) -> bool:
        """Reset a setting to its default value"""
        if key not in self._setting_definitions:
            return False
        
        default_value = self._setting_definitions[key].default_value
        return self.set_setting(key, default_value, validate=False)
    
    def reset_category(self, category: str):
        """Reset all settings in a category to defaults"""
        for definition in self._setting_definitions.values():
            if definition.category == category:
                self.reset_setting(definition.key)
    
    def reset_all_settings(self):
        """Reset all settings to defaults"""
        for key in self._setting_definitions.keys():
            self.reset_setting(key)
    
    def get_profile(self, name: str) -> Optional[SettingProfile]:
        """Get a settings profile"""
        return self.profiles.get(name)
    
    def create_profile(self, name: str, description: str, 
                      base_profile: str = None) -> SettingProfile:
        """Create a new settings profile"""
        
        if base_profile and base_profile in self.profiles:
            base_settings = self.profiles[base_profile].settings.copy()
        else:
            base_settings = self.current_settings.copy()
        
        profile = SettingProfile(
            name=name,
            description=description,
            settings=base_settings
        )
        
        self.profiles[name] = profile
        self.save_profiles()
        
        logger.info(f"Created settings profile: {name}")
        return profile
    
    def delete_profile(self, name: str) -> bool:
        """Delete a settings profile"""
        if name not in self.profiles:
            return False
        
        profile = self.profiles[name]
        if profile.is_builtin:
            logger.warning(f"Cannot delete built-in profile: {name}")
            return False
        
        del self.profiles[name]
        
        # Switch to default if this was current profile
        if self.current_profile == name:
            self.load_profile("default")
        
        self.save_profiles()
        logger.info(f"Deleted settings profile: {name}")
        return True
    
    def load_profile(self, name: str) -> bool:
        """Load a settings profile"""
        if name not in self.profiles:
            logger.warning(f"Profile not found: {name}")
            return False
        
        profile = self.profiles[name]
        
        # Load settings from profile
        for key, value in profile.settings.items():
            if key in self._setting_definitions:
                self.set_setting(key, value, validate=False)
        
        self.current_profile = name
        logger.info(f"Loaded settings profile: {name}")
        return True
    
    def save_current_as_profile(self, name: str, description: str):
        """Save current settings as a new profile"""
        profile = SettingProfile(
            name=name,
            description=description,
            settings=self.current_settings.copy()
        )
        
        self.profiles[name] = profile
        self.save_profiles()
        
        logger.info(f"Saved current settings as profile: {name}")
    
    def _create_builtin_profiles(self):
        """Create built-in profiles"""
        
        # Default profile
        if "default" not in self.profiles:
            default_settings = {}
            for definition in self._setting_definitions.values():
                default_settings[definition.key] = definition.default_value
            
            self.profiles["default"] = SettingProfile(
                name="default",
                description="Default settings",
                settings=default_settings,
                is_builtin=True
            )
        
        # Performance profiles
        if "performance" not in self.profiles:
            perf_settings = self.profiles["default"].settings.copy()
            perf_settings.update({
                "ai.batch_size": 8,
                "ai.num_inference_steps": 15,
                "performance.max_concurrent_frames": 20,
                "performance.cleanup_temp_files": True
            })
            
            self.profiles["performance"] = SettingProfile(
                name="performance",
                description="Optimized for speed",
                settings=perf_settings,
                is_builtin=True
            )
        
        # Quality profile
        if "quality" not in self.profiles:
            quality_settings = self.profiles["default"].settings.copy()
            quality_settings.update({
                "ai.batch_size": 2,
                "ai.num_inference_steps": 30,
                "ai.guidance_scale": 10.0,
                "performance.max_concurrent_frames": 5
            })
            
            self.profiles["quality"] = SettingProfile(
                name="quality",
                description="Optimized for quality",
                settings=quality_settings,
                is_builtin=True
            )
    
    def add_change_callback(self, callback: Callable[[str, Any, Any], None]):
        """Add a callback for setting changes"""
        self.change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable):
        """Remove a change callback"""
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
    
    def _notify_change_callbacks(self, key: str, old_value: Any, new_value: Any):
        """Notify all change callbacks"""
        for callback in self.change_callbacks:
            try:
                callback(key, old_value, new_value)
            except Exception as e:
                logger.warning(f"Settings change callback error: {e}")
    
    def requires_restart(self) -> bool:
        """Check if any settings require application restart"""
        return len(self.pending_restart_settings) > 0
    
    def get_restart_settings(self) -> List[str]:
        """Get list of settings that require restart"""
        return list(self.pending_restart_settings)
    
    def clear_restart_flags(self):
        """Clear restart requirement flags"""
        self.pending_restart_settings.clear()
    
    def save_settings(self):
        """Save current settings to file"""
        try:
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            
            settings_data = {
                "settings": self.current_settings,
                "current_profile": self.current_profile,
                "version": "1.0"
            }
            
            with open(self.settings_file, "w") as f:
                json.dump(settings_data, f, indent=2)
                
            logger.debug("Settings saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
    
    def load_settings(self):
        """Load settings from file"""
        try:
            if not self.settings_file.exists():
                # Initialize with defaults
                for definition in self._setting_definitions.values():
                    self.current_settings[definition.key] = definition.default_value
                return
            
            with open(self.settings_file, "r") as f:
                settings_data = json.load(f)
            
            # Load settings
            loaded_settings = settings_data.get("settings", {})
            for key, value in loaded_settings.items():
                if key in self._setting_definitions:
                    self.current_settings[key] = value
            
            # Load current profile
            self.current_profile = settings_data.get("current_profile", "default")
            
            logger.debug("Settings loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load settings: {e}")
            # Initialize with defaults
            for definition in self._setting_definitions.values():
                self.current_settings[definition.key] = definition.default_value
    
    def save_profiles(self):
        """Save profiles to file"""
        try:
            self.profiles_file.parent.mkdir(parents=True, exist_ok=True)
            
            profiles_data = {
                "profiles": {name: profile.to_dict() 
                           for name, profile in self.profiles.items()},
                "version": "1.0"
            }
            
            with open(self.profiles_file, "w") as f:
                json.dump(profiles_data, f, indent=2)
                
            logger.debug("Profiles saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save profiles: {e}")
    
    def load_profiles(self):
        """Load profiles from file"""
        try:
            if not self.profiles_file.exists():
                return
            
            with open(self.profiles_file, "r") as f:
                profiles_data = json.load(f)
            
            # Load profiles
            for name, profile_data in profiles_data.get("profiles", {}).items():
                self.profiles[name] = SettingProfile.from_dict(profile_data)
            
            logger.debug("Profiles loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load profiles: {e}")
    
    def export_settings(self, file_path: str, include_profiles: bool = True):
        """Export settings to file"""
        try:
            export_data = {
                "settings": self.current_settings,
                "current_profile": self.current_profile,
                "export_timestamp": __import__('time').time(),
                "version": "1.0"
            }
            
            if include_profiles:
                export_data["profiles"] = {
                    name: profile.to_dict() 
                    for name, profile in self.profiles.items()
                    if not profile.is_builtin
                }
            
            with open(file_path, "w") as f:
                json.dump(export_data, f, indent=2)
                
            logger.info(f"Settings exported to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export settings: {e}")
            raise
    
    def import_settings(self, file_path: str, 
                       import_profiles: bool = True) -> Dict[str, Any]:
        """Import settings from file"""
        try:
            with open(file_path, "r") as f:
                import_data = json.load(f)
            
            # Import settings
            imported_settings = import_data.get("settings", {})
            valid_settings = {}
            invalid_settings = {}
            
            for key, value in imported_settings.items():
                if key in self._setting_definitions:
                    is_valid, error_msg = self._setting_definitions[key].validate(value)
                    if is_valid:
                        valid_settings[key] = value
                    else:
                        invalid_settings[key] = error_msg
                else:
                    invalid_settings[key] = "Unknown setting"
            
            # Apply valid settings
            for key, value in valid_settings.items():
                self.set_setting(key, value, validate=False)
            
            # Import profiles if requested
            imported_profiles = []
            if import_profiles and "profiles" in import_data:
                for name, profile_data in import_data["profiles"].items():
                    if name not in self.profiles or not self.profiles[name].is_builtin:
                        profile = SettingProfile.from_dict(profile_data)
                        self.profiles[name] = profile
                        imported_profiles.append(name)
                
                self.save_profiles()
            
            # Save settings
            self.save_settings()
            
            result = {
                "valid_settings": len(valid_settings),
                "invalid_settings": invalid_settings,
                "imported_profiles": imported_profiles
            }
            
            logger.info(f"Settings imported: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to import settings: {e}")
            raise