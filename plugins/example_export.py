"""
Example Export Plugin
Demonstrates how to create a custom export plugin
"""

import json
import csv
from typing import Any, List, Dict
from pathlib import Path
import time

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from plugins.plugin_system import ExportPlugin, PluginInfo, PluginType


class ProcessingReportExporter(ExportPlugin):
    """Export processing reports and statistics"""
    
    def get_info(self) -> PluginInfo:
        """Get plugin information"""
        return PluginInfo(
            id="processing_report_exporter",
            name="Processing Report Exporter",
            version="1.0.0",
            description="Exports processing reports and statistics to various formats",
            author="UpScale Team",
            plugin_type=PluginType.EXPORT,
            settings_schema={
                "format": {
                    "type": "choice",
                    "choices": ["json", "csv", "txt"],
                    "default": "json",
                    "description": "Export format"
                },
                "include_metadata": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include processing metadata"
                },
                "include_timestamps": {
                    "type": "boolean", 
                    "default": True,
                    "description": "Include timestamps in report"
                }
            }
        )
    
    def initialize(self, settings: Dict[str, Any] = None):
        """Initialize the plugin"""
        super().initialize(settings)
        
        self.format = self.settings.get("format", "json")
        self.include_metadata = self.settings.get("include_metadata", True)
        self.include_timestamps = self.settings.get("include_timestamps", True)
        
        print(f"Report Exporter initialized: format={self.format}")
    
    def export(self, data: Any, output_path: str, **kwargs) -> bool:
        """Export data to specified format"""
        
        try:
            # Prepare export data
            export_data = self._prepare_data(data)
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if self.format == "json":
                return self._export_json(export_data, output_file)
            elif self.format == "csv":
                return self._export_csv(export_data, output_file)
            elif self.format == "txt":
                return self._export_text(export_data, output_file)
            else:
                print(f"Unsupported format: {self.format}")
                return False
                
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    def _prepare_data(self, data: Any) -> Dict:
        """Prepare data for export"""
        
        if isinstance(data, dict):
            processed_data = data.copy()
        elif isinstance(data, list):
            processed_data = {"items": data}
        else:
            processed_data = {"data": str(data)}
        
        # Add metadata if requested
        if self.include_metadata:
            processed_data["export_metadata"] = {
                "export_format": self.format,
                "exported_by": "Processing Report Exporter v1.0.0",
                "settings": self.settings.copy()
            }
        
        # Add timestamp if requested
        if self.include_timestamps:
            processed_data["export_timestamp"] = time.time()
            processed_data["export_time_readable"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        return processed_data
    
    def _export_json(self, data: Dict, output_file: Path) -> bool:
        """Export as JSON"""
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"Exported to JSON: {output_file}")
            return True
        except Exception as e:
            print(f"JSON export failed: {e}")
            return False
    
    def _export_csv(self, data: Dict, output_file: Path) -> bool:
        """Export as CSV"""
        try:
            # Handle different data structures
            if "items" in data and isinstance(data["items"], list):
                items = data["items"]
                if items and isinstance(items[0], dict):
                    # List of dictionaries - standard CSV
                    with open(output_file, "w", newline="", encoding="utf-8") as f:
                        if items:
                            fieldnames = items[0].keys()
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(items)
                else:
                    # List of simple values
                    with open(output_file, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        for item in items:
                            writer.writerow([item])
            else:
                # Single object - convert to key-value pairs
                with open(output_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Key", "Value"])
                    for key, value in data.items():
                        writer.writerow([key, str(value)])
            
            print(f"Exported to CSV: {output_file}")
            return True
        except Exception as e:
            print(f"CSV export failed: {e}")
            return False
    
    def _export_text(self, data: Dict, output_file: Path) -> bool:
        """Export as plain text"""
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("Processing Report\n")
                f.write("=" * 50 + "\n\n")
                
                # Write timestamp if available
                if "export_time_readable" in data:
                    f.write(f"Exported: {data['export_time_readable']}\n\n")
                
                # Write main data
                for key, value in data.items():
                    if key.startswith("export_"):
                        continue  # Skip metadata in main section
                    
                    f.write(f"{key}:\n")
                    if isinstance(value, (dict, list)):
                        f.write(f"  {json.dumps(value, indent=2, default=str)}\n")
                    else:
                        f.write(f"  {value}\n")
                    f.write("\n")
                
                # Write metadata section
                if "export_metadata" in data:
                    f.write("\nExport Information\n")
                    f.write("-" * 30 + "\n")
                    for key, value in data["export_metadata"].items():
                        f.write(f"{key}: {value}\n")
            
            print(f"Exported to text: {output_file}")
            return True
        except Exception as e:
            print(f"Text export failed: {e}")
            return False
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions"""
        return [".json", ".csv", ".txt"]
    
    def get_export_options(self) -> Dict[str, Any]:
        """Get export options schema"""
        return {
            "format": {
                "type": "choice",
                "choices": ["json", "csv", "txt"],
                "default": "json",
                "description": "Output format"
            },
            "include_metadata": {
                "type": "boolean",
                "default": True,
                "description": "Include export metadata"
            },
            "include_timestamps": {
                "type": "boolean",
                "default": True,
                "description": "Include timestamps"
            }
        }
    
    def get_settings_ui(self) -> Dict[str, Any]:
        """Get settings UI definition"""
        return {
            "type": "form",
            "fields": [
                {
                    "name": "format",
                    "label": "Export Format",
                    "type": "select",
                    "options": [
                        {"value": "json", "label": "JSON"},
                        {"value": "csv", "label": "CSV"},
                        {"value": "txt", "label": "Text"}
                    ],
                    "default": "json"
                },
                {
                    "name": "include_metadata",
                    "label": "Include Metadata",
                    "type": "checkbox",
                    "default": True
                },
                {
                    "name": "include_timestamps",
                    "label": "Include Timestamps",
                    "type": "checkbox",
                    "default": True
                }
            ]
        }
    
    def validate_settings(self, settings: Dict[str, Any]) -> tuple[bool, str]:
        """Validate plugin settings"""
        
        # Check format
        format_val = settings.get("format", "json")
        valid_formats = ["json", "csv", "txt"]
        if format_val not in valid_formats:
            return False, f"Format must be one of: {', '.join(valid_formats)}"
        
        # Check boolean settings
        for key in ["include_metadata", "include_timestamps"]:
            value = settings.get(key, True)
            if not isinstance(value, bool):
                return False, f"{key} must be a boolean value"
        
        return True, ""