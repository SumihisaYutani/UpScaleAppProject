"""
Plugin System Test Script
Tests the plugin system functionality with example plugins
"""

import sys
import os
import time
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from plugins.plugin_system import PluginManager, PluginType


def test_plugin_discovery():
    """Test plugin discovery and loading"""
    print("=== Plugin Discovery Test ===")
    
    plugin_manager = PluginManager()
    
    # Scan for plugins
    available_plugins = plugin_manager.scan_plugins()
    print(f"Found {len(available_plugins)} plugins:")
    
    for plugin in available_plugins:
        print(f"  - {plugin.name} ({plugin.id}) v{plugin.version}")
        print(f"    Type: {plugin.plugin_type.value}")
        print(f"    Description: {plugin.description}")
        print(f"    Author: {plugin.author}")
        print()
    
    return available_plugins


def test_plugin_loading():
    """Test plugin loading and initialization"""
    print("=== Plugin Loading Test ===")
    
    plugin_manager = PluginManager()
    
    # Load example plugins
    test_plugins = ["sharpness_filter", "noise_reduction_processor", "processing_report_exporter"]
    
    for plugin_id in test_plugins:
        print(f"Loading plugin: {plugin_id}")
        
        success = plugin_manager.load_plugin(plugin_id)
        if success:
            plugin = plugin_manager.get_plugin(plugin_id)
            if plugin:
                print(f"  ✓ Loaded: {plugin.info.name}")
                print(f"    Status: {'Enabled' if plugin_manager.is_plugin_enabled(plugin_id) else 'Disabled'}")
            else:
                print(f"  ✗ Plugin not found after loading")
        else:
            print(f"  ✗ Failed to load plugin")
        print()


def test_plugin_functionality():
    """Test plugin functionality"""
    print("=== Plugin Functionality Test ===")
    
    plugin_manager = PluginManager()
    
    # Test filter plugin
    filter_plugin = plugin_manager.get_plugin("sharpness_filter")
    if filter_plugin:
        print("Testing Sharpness Filter Plugin:")
        
        # Test settings validation
        valid_settings = {"strength": 0.7, "radius": 1.5, "threshold": 5}
        invalid_settings = {"strength": 3.0, "radius": -1.0}  # Invalid values
        
        is_valid, msg = filter_plugin.validate_settings(valid_settings)
        print(f"  Valid settings test: {'✓' if is_valid else '✗'} {msg}")
        
        is_valid, msg = filter_plugin.validate_settings(invalid_settings)
        print(f"  Invalid settings test: {'✓' if not is_valid else '✗'} {msg}")
        
        # Test UI schema
        ui_schema = filter_plugin.get_settings_ui()
        print(f"  UI schema fields: {len(ui_schema.get('fields', []))}")
        print()
    
    # Test processor plugin
    processor_plugin = plugin_manager.get_plugin("noise_reduction_processor")
    if processor_plugin:
        print("Testing Noise Reduction Processor Plugin:")
        
        # Test supported formats
        formats = processor_plugin.get_supported_formats()
        print(f"  Supported formats: {', '.join(formats)}")
        
        # Test settings validation
        valid_settings = {"method": "bilateral", "strength": 1.2}
        is_valid, msg = processor_plugin.validate_settings(valid_settings)
        print(f"  Settings validation: {'✓' if is_valid else '✗'} {msg}")
        print()
    
    # Test export plugin
    export_plugin = plugin_manager.get_plugin("processing_report_exporter")
    if export_plugin:
        print("Testing Processing Report Exporter Plugin:")
        
        # Test supported extensions
        extensions = export_plugin.get_supported_extensions()
        print(f"  Supported extensions: {', '.join(extensions)}")
        
        # Test export functionality with sample data
        test_data = {
            "processing_stats": {
                "files_processed": 10,
                "success_count": 8,
                "fail_count": 2,
                "average_time": 45.2
            },
            "settings_used": {
                "scale_factor": 1.5,
                "quality": "high",
                "ai_enabled": True
            }
        }
        
        test_output_path = Path("test_output")
        test_output_path.mkdir(exist_ok=True)
        
        # Test different formats
        for format_type in ["json", "csv", "txt"]:
            export_plugin.settings["format"] = format_type
            success = export_plugin.export(
                test_data,
                str(test_output_path / f"test_report.{format_type}")
            )
            print(f"  Export to {format_type}: {'✓' if success else '✗'}")
        print()


def test_plugin_management():
    """Test plugin management operations"""
    print("=== Plugin Management Test ===")
    
    plugin_manager = PluginManager()
    
    # Test enabling/disabling plugins
    test_plugin = "sharpness_filter"
    
    print(f"Plugin {test_plugin} initial status: {'Enabled' if plugin_manager.is_plugin_enabled(test_plugin) else 'Disabled'}")
    
    # Disable plugin
    plugin_manager.disable_plugin(test_plugin)
    print(f"After disable: {'Enabled' if plugin_manager.is_plugin_enabled(test_plugin) else 'Disabled'}")
    
    # Enable plugin
    plugin_manager.enable_plugin(test_plugin)
    print(f"After enable: {'Enabled' if plugin_manager.is_plugin_enabled(test_plugin) else 'Disabled'}")
    
    # Test getting plugins by type
    filter_plugins = plugin_manager.get_plugins_by_type(PluginType.FILTER)
    processor_plugins = plugin_manager.get_plugins_by_type(PluginType.PROCESSOR)
    export_plugins = plugin_manager.get_plugins_by_type(PluginType.EXPORT)
    
    print(f"\nPlugins by type:")
    print(f"  Filter plugins: {len(filter_plugins)}")
    print(f"  Processor plugins: {len(processor_plugins)}")
    print(f"  Export plugins: {len(export_plugins)}")
    print()


def test_plugin_settings():
    """Test plugin settings management"""
    print("=== Plugin Settings Test ===")
    
    plugin_manager = PluginManager()
    
    test_plugin = "sharpness_filter"
    
    # Set plugin settings
    plugin_manager.set_plugin_setting(test_plugin, "strength", 0.8)
    plugin_manager.set_plugin_setting(test_plugin, "radius", 2.0)
    
    # Get plugin settings
    strength = plugin_manager.get_plugin_setting(test_plugin, "strength")
    radius = plugin_manager.get_plugin_setting(test_plugin, "radius")
    
    print(f"Plugin settings for {test_plugin}:")
    print(f"  Strength: {strength}")
    print(f"  Radius: {radius}")
    
    # Test settings persistence
    plugin_manager.save_plugin_settings()
    
    # Create new plugin manager to test loading
    new_plugin_manager = PluginManager()
    loaded_strength = new_plugin_manager.get_plugin_setting(test_plugin, "strength", "not found")
    
    print(f"  Settings persistence test: {'✓' if loaded_strength == 0.8 else '✗'}")
    print()


def test_plugin_template_creation():
    """Test plugin template creation"""
    print("=== Plugin Template Creation Test ===")
    
    plugin_manager = PluginManager()
    
    # Test creating templates for different plugin types
    template_types = [
        ("Test Filter", PluginType.FILTER),
        ("Test Processor", PluginType.PROCESSOR),
        ("Test Exporter", PluginType.EXPORT),
        ("Test AI Model", PluginType.AI_MODEL)
    ]
    
    template_dir = Path("test_templates")
    
    for plugin_name, plugin_type in template_types:
        try:
            template_path = plugin_manager.create_plugin_template(
                plugin_name, plugin_type, template_dir
            )
            print(f"  ✓ Created {plugin_type.value} template: {template_path}")
        except Exception as e:
            print(f"  ✗ Failed to create {plugin_type.value} template: {e}")
    
    print()


def main():
    """Main test function"""
    print("Plugin System Test Suite")
    print("=" * 50)
    
    try:
        # Run all tests
        test_plugin_discovery()
        test_plugin_loading()
        test_plugin_functionality()
        test_plugin_management()
        test_plugin_settings()
        test_plugin_template_creation()
        
        print("=" * 50)
        print("All tests completed!")
        
    except Exception as e:
        print(f"Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()