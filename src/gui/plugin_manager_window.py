"""
Plugin Manager GUI Window
Interface for managing plugins and their settings
"""

import tkinter as tk
import sys
import os
from pathlib import Path
import threading
from typing import List, Dict, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import customtkinter as ctk
    from CTkMessagebox import CTkMessagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

from plugins.plugin_system import PluginManager, PluginInfo, PluginType


class PluginSettingsFrame(ctk.CTkScrollableFrame):
    """Frame for editing plugin settings"""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.plugin = None
        self.setting_widgets = {}
    
    def load_plugin_settings(self, plugin):
        """Load settings for a specific plugin"""
        self.plugin = plugin
        
        # Clear existing widgets
        for widget in self.winfo_children():
            widget.destroy()
        self.setting_widgets.clear()
        
        if not plugin or not plugin.info:
            return
        
        # Create settings UI based on plugin schema
        settings_ui = plugin.get_settings_ui()
        if not settings_ui or "fields" not in settings_ui:
            # No custom UI, use default based on schema
            self._create_default_settings_ui()
        else:
            # Use plugin-defined UI
            self._create_custom_settings_ui(settings_ui)
    
    def _create_default_settings_ui(self):
        """Create default settings UI from plugin schema"""
        if not self.plugin or not self.plugin.info.settings_schema:
            return
        
        for key, schema in self.plugin.info.settings_schema.items():
            self._create_setting_widget(key, schema)
    
    def _create_custom_settings_ui(self, settings_ui):
        """Create custom settings UI from plugin definition"""
        fields = settings_ui.get("fields", [])
        
        for field in fields:
            field_name = field.get("name", "")
            field_type = field.get("type", "text")
            field_label = field.get("label", field_name)
            
            # Create label
            label = ctk.CTkLabel(self, text=field_label)
            label.pack(anchor="w", padx=5, pady=(10, 0))
            
            current_value = self.plugin.settings.get(field_name, field.get("default"))
            
            if field_type == "slider":
                # Slider widget
                frame = ctk.CTkFrame(self)
                frame.pack(fill="x", padx=5, pady=2)
                
                slider = ctk.CTkSlider(
                    frame,
                    from_=field.get("min", 0),
                    to=field.get("max", 1),
                    number_of_steps=int((field.get("max", 1) - field.get("min", 0)) / field.get("step", 0.1))
                )
                slider.set(current_value)
                slider.pack(side="left", fill="x", expand=True, padx=5, pady=5)
                
                value_label = ctk.CTkLabel(frame, text=f"{current_value:.2f}", width=60)
                value_label.pack(side="right", padx=5, pady=5)
                
                slider.configure(command=lambda v: value_label.configure(text=f"{v:.2f}"))
                
                self.setting_widgets[field_name] = {
                    "widget": slider,
                    "get_value": lambda: slider.get(),
                    "type": "slider"
                }
            
            elif field_type == "select":
                # Combobox widget
                options = field.get("options", [])
                values = [opt["value"] for opt in options]
                
                combobox = ctk.CTkComboBox(
                    self,
                    values=values,
                    state="readonly"
                )
                if current_value in values:
                    combobox.set(current_value)
                combobox.pack(fill="x", padx=5, pady=2)
                
                self.setting_widgets[field_name] = {
                    "widget": combobox,
                    "get_value": lambda cb=combobox: cb.get(),
                    "type": "select"
                }
            
            elif field_type == "checkbox":
                # Checkbox widget
                var = tk.BooleanVar(value=current_value)
                checkbox = ctk.CTkCheckBox(self, text="", variable=var)
                checkbox.pack(anchor="w", padx=5, pady=2)
                
                self.setting_widgets[field_name] = {
                    "widget": checkbox,
                    "get_value": lambda: var.get(),
                    "type": "checkbox"
                }
            
            else:
                # Default to text entry
                entry = ctk.CTkEntry(self)
                entry.insert(0, str(current_value))
                entry.pack(fill="x", padx=5, pady=2)
                
                self.setting_widgets[field_name] = {
                    "widget": entry,
                    "get_value": lambda e=entry: e.get(),
                    "type": "text"
                }
    
    def _create_setting_widget(self, key, schema):
        """Create a widget for a setting based on its schema"""
        # This is a simplified version - could be expanded
        label = ctk.CTkLabel(self, text=key.replace("_", " ").title())
        label.pack(anchor="w", padx=5, pady=(10, 0))
        
        current_value = self.plugin.settings.get(key, schema.get("default", ""))
        
        entry = ctk.CTkEntry(self)
        entry.insert(0, str(current_value))
        entry.pack(fill="x", padx=5, pady=2)
        
        self.setting_widgets[key] = {
            "widget": entry,
            "get_value": lambda e=entry: e.get(),
            "type": "text"
        }
    
    def get_settings_values(self) -> Dict:
        """Get current values from all setting widgets"""
        values = {}
        for key, widget_info in self.setting_widgets.items():
            try:
                value = widget_info["get_value"]()
                values[key] = value
            except:
                pass
        return values


class PluginListFrame(ctk.CTkScrollableFrame):
    """Scrollable frame for displaying plugin list"""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.plugin_widgets = {}
        self.selected_plugin = None
        self.selection_callback = None
    
    def add_plugin_widget(self, plugin_info: PluginInfo, is_loaded: bool, is_enabled: bool):
        """Add a plugin widget to the list"""
        
        # Create plugin frame
        plugin_frame = ctk.CTkFrame(self)
        plugin_frame.pack(fill="x", padx=5, pady=2)
        plugin_frame.grid_columnconfigure(1, weight=1)
        
        # Plugin icon/type indicator
        type_colors = {
            PluginType.PROCESSOR: "blue",
            PluginType.FILTER: "green", 
            PluginType.EXPORT: "orange",
            PluginType.AI_MODEL: "purple"
        }
        
        type_badge = ctk.CTkLabel(
            plugin_frame,
            text=plugin_info.plugin_type.value[:4].upper(),
            fg_color=type_colors.get(plugin_info.plugin_type, "gray"),
            corner_radius=5,
            width=50
        )
        type_badge.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        # Plugin info
        info_frame = ctk.CTkFrame(plugin_frame)
        info_frame.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        info_frame.grid_columnconfigure(0, weight=1)
        
        # Plugin name and version
        name_label = ctk.CTkLabel(
            info_frame,
            text=f"{plugin_info.name} v{plugin_info.version}",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        name_label.grid(row=0, column=0, padx=5, pady=(5, 0), sticky="ew")
        
        # Description
        desc_label = ctk.CTkLabel(
            info_frame,
            text=plugin_info.description[:80] + "..." if len(plugin_info.description) > 80 else plugin_info.description,
            anchor="w",
            text_color="gray"
        )
        desc_label.grid(row=1, column=0, padx=5, pady=(0, 5), sticky="ew")
        
        # Control buttons
        controls_frame = ctk.CTkFrame(plugin_frame)
        controls_frame.grid(row=0, column=2, padx=5, pady=5)
        
        # Enable/Disable toggle
        enabled_var = tk.BooleanVar(value=is_enabled)
        enable_checkbox = ctk.CTkCheckBox(
            controls_frame,
            text="Enabled",
            variable=enabled_var,
            command=lambda: self._toggle_plugin_enabled(plugin_info.id, enabled_var.get())
        )
        enable_checkbox.pack(side="top", padx=5, pady=2)
        
        # Settings button
        if plugin_info.settings_schema:
            settings_btn = ctk.CTkButton(
                controls_frame,
                text="Settings",
                width=70,
                height=25,
                command=lambda: self._select_plugin(plugin_info.id)
            )
            settings_btn.pack(side="bottom", padx=5, pady=2)
        
        # Store references
        self.plugin_widgets[plugin_info.id] = {
            "frame": plugin_frame,
            "enabled_var": enabled_var,
            "enable_checkbox": enable_checkbox
        }
    
    def update_plugin_widget(self, plugin_id: str, is_enabled: bool):
        """Update plugin widget state"""
        if plugin_id in self.plugin_widgets:
            widgets = self.plugin_widgets[plugin_id]
            widgets["enabled_var"].set(is_enabled)
    
    def clear_all_widgets(self):
        """Clear all plugin widgets"""
        for plugin_id in list(self.plugin_widgets.keys()):
            self.plugin_widgets[plugin_id]["frame"].destroy()
        self.plugin_widgets.clear()
    
    def _toggle_plugin_enabled(self, plugin_id: str, enabled: bool):
        """Handle plugin enable/disable"""
        if self.selection_callback:
            self.selection_callback("toggle_enabled", plugin_id, enabled)
    
    def _select_plugin(self, plugin_id: str):
        """Handle plugin selection for settings"""
        self.selected_plugin = plugin_id
        if self.selection_callback:
            self.selection_callback("select", plugin_id, None)


class PluginManagerWindow:
    """Plugin manager window"""
    
    def __init__(self, parent=None):
        if not GUI_AVAILABLE:
            raise ImportError("GUI dependencies not available")
        
        self.parent = parent
        self.plugin_manager = PluginManager()
        
        self._setup_window()
        self._setup_ui()
        self._refresh_plugin_list()
    
    def _setup_window(self):
        """Setup plugin manager window"""
        self.window = ctk.CTkToplevel(self.parent)
        self.window.title("Plugin Manager - UpScale App")
        self.window.geometry("1000x700")
        
        # Make it modal if parent exists
        if self.parent:
            self.window.transient(self.parent)
            self.window.grab_set()
        
        # Configure grid
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        
        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self._on_window_close)
    
    def _setup_ui(self):
        """Setup user interface"""
        
        # Main container
        main_frame = ctk.CTkFrame(self.window)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        
        # Top controls
        self._setup_controls(main_frame)
        
        # Content area - split between plugin list and settings
        content_frame = ctk.CTkFrame(main_frame)
        content_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)
        
        # Plugin list
        self._setup_plugin_list(content_frame)
        
        # Plugin settings
        self._setup_plugin_settings(content_frame)
        
        # Bottom status
        self._setup_status_bar(main_frame)
    
    def _setup_controls(self, parent):
        """Setup control buttons"""
        
        controls_frame = ctk.CTkFrame(parent)
        controls_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        # Refresh button
        refresh_btn = ctk.CTkButton(
            controls_frame,
            text="🔄 Refresh",
            command=self._refresh_plugin_list,
            height=35
        )
        refresh_btn.pack(side="left", padx=5, pady=5)
        
        # Create plugin button
        create_btn = ctk.CTkButton(
            controls_frame,
            text="➕ Create Plugin",
            command=self._create_plugin_template,
            height=35
        )
        create_btn.pack(side="left", padx=5, pady=5)
        
        # Plugin folder button
        folder_btn = ctk.CTkButton(
            controls_frame,
            text="📁 Plugin Folder",
            command=self._open_plugin_folder,
            height=35,
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "gray90")
        )
        folder_btn.pack(side="right", padx=5, pady=5)
    
    def _setup_plugin_list(self, parent):
        """Setup plugin list display"""
        
        # Plugin list frame
        list_container = ctk.CTkFrame(parent)
        list_container.grid(row=0, column=0, sticky="nsew", padx=(5, 2), pady=5)
        list_container.grid_rowconfigure(1, weight=1)
        list_container.grid_columnconfigure(0, weight=1)
        
        # List title
        list_title = ctk.CTkLabel(
            list_container,
            text="Available Plugins",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        list_title.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        
        # Plugin list
        self.plugin_list = PluginListFrame(list_container)
        self.plugin_list.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.plugin_list.selection_callback = self._on_plugin_action
    
    def _setup_plugin_settings(self, parent):
        """Setup plugin settings panel"""
        
        # Settings frame
        settings_container = ctk.CTkFrame(parent)
        settings_container.grid(row=0, column=1, sticky="nsew", padx=(2, 5), pady=5)
        settings_container.grid_rowconfigure(1, weight=1)
        settings_container.grid_columnconfigure(0, weight=1)
        
        # Settings title
        self.settings_title = ctk.CTkLabel(
            settings_container,
            text="Plugin Settings",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.settings_title.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        
        # Settings content
        self.settings_frame = PluginSettingsFrame(settings_container)
        self.settings_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        # Settings buttons
        settings_buttons = ctk.CTkFrame(settings_container)
        settings_buttons.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        
        self.apply_btn = ctk.CTkButton(
            settings_buttons,
            text="Apply Settings",
            command=self._apply_settings,
            state="disabled"
        )
        self.apply_btn.pack(side="right", padx=5, pady=5)
        
        self.reset_btn = ctk.CTkButton(
            settings_buttons,
            text="Reset",
            command=self._reset_settings,
            state="disabled",
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "gray90")
        )
        self.reset_btn.pack(side="right", padx=5, pady=5)
    
    def _setup_status_bar(self, parent):
        """Setup status bar"""
        
        status_frame = ctk.CTkFrame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Ready",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(side="left", padx=10, pady=5)
        
        self.plugin_count_label = ctk.CTkLabel(
            status_frame,
            text="Plugins: 0 loaded, 0 enabled",
            font=ctk.CTkFont(size=12)
        )
        self.plugin_count_label.pack(side="right", padx=10, pady=5)
    
    def _refresh_plugin_list(self):
        """Refresh the plugin list display"""
        
        self.status_label.configure(text="Scanning plugins...")
        self.window.update()
        
        # Clear current list
        self.plugin_list.clear_all_widgets()
        
        # Scan for plugins
        available_plugins = self.plugin_manager.scan_plugins()
        loaded_plugins = {p.info.id for p in self.plugin_manager.get_loaded_plugins()}
        
        # Add plugin widgets
        for plugin_info in available_plugins:
            is_loaded = plugin_info.id in loaded_plugins
            is_enabled = self.plugin_manager.is_plugin_enabled(plugin_info.id)
            
            # Auto-load plugins for display
            if not is_loaded:
                self.plugin_manager.load_plugin(plugin_info.id)
            
            self.plugin_list.add_plugin_widget(plugin_info, is_loaded, is_enabled)
        
        # Update status
        loaded_count = len(self.plugin_manager.get_loaded_plugins())
        enabled_count = sum(1 for p in available_plugins 
                           if self.plugin_manager.is_plugin_enabled(p.id))
        
        self.plugin_count_label.configure(
            text=f"Plugins: {loaded_count} loaded, {enabled_count} enabled"
        )
        self.status_label.configure(text="Ready")
    
    def _on_plugin_action(self, action: str, plugin_id: str, value):
        """Handle plugin actions from list"""
        
        if action == "toggle_enabled":
            if value:
                self.plugin_manager.enable_plugin(plugin_id)
            else:
                self.plugin_manager.disable_plugin(plugin_id)
            
            self.status_label.configure(
                text=f"Plugin {plugin_id} {'enabled' if value else 'disabled'}"
            )
            
        elif action == "select":
            plugin = self.plugin_manager.get_plugin(plugin_id)
            if plugin:
                self.settings_frame.load_plugin_settings(plugin)
                self.settings_title.configure(text=f"Settings - {plugin.info.name}")
                self.apply_btn.configure(state="normal")
                self.reset_btn.configure(state="normal")
    
    def _apply_settings(self):
        """Apply current settings to plugin"""
        
        if not self.settings_frame.plugin:
            return
        
        plugin = self.settings_frame.plugin
        new_settings = self.settings_frame.get_settings_values()
        
        # Validate settings
        is_valid, error_msg = plugin.validate_settings(new_settings)
        
        if not is_valid:
            CTkMessagebox(
                title="Invalid Settings",
                message=f"Settings validation failed: {error_msg}",
                icon="warning"
            )
            return
        
        # Apply settings
        for key, value in new_settings.items():
            self.plugin_manager.set_plugin_setting(plugin.info.id, key, value)
            plugin.settings[key] = value
        
        self.status_label.configure(text=f"Settings applied to {plugin.info.name}")
        
        CTkMessagebox(
            title="Settings Applied",
            message="Plugin settings have been saved successfully.",
            icon="check"
        )
    
    def _reset_settings(self):
        """Reset settings to defaults"""
        
        if not self.settings_frame.plugin:
            return
        
        # Reload plugin settings (this will reset to defaults)
        plugin = self.settings_frame.plugin
        self.settings_frame.load_plugin_settings(plugin)
        
        self.status_label.configure(text=f"Reset settings for {plugin.info.name}")
    
    def _create_plugin_template(self):
        """Create a new plugin template"""
        
        # Simple template creation dialog
        dialog = ctk.CTkInputDialog(
            text="Enter plugin name:",
            title="Create Plugin Template"
        )
        
        plugin_name = dialog.get_input()
        if plugin_name:
            try:
                template_path = self.plugin_manager.create_plugin_template(
                    plugin_name, PluginType.PROCESSOR
                )
                
                CTkMessagebox(
                    title="Template Created",
                    message=f"Plugin template created at:\n{template_path}",
                    icon="check"
                )
                
                self._refresh_plugin_list()
                
            except Exception as e:
                CTkMessagebox(
                    title="Error",
                    message=f"Failed to create template: {str(e)}",
                    icon="cancel"
                )
    
    def _open_plugin_folder(self):
        """Open the plugins folder in file explorer"""
        import subprocess
        import os
        
        plugins_dir = Path("plugins").absolute()
        
        try:
            if os.name == 'nt':  # Windows
                os.startfile(plugins_dir)
            elif os.name == 'posix':  # macOS and Linux
                subprocess.call(['open', plugins_dir])
        except:
            CTkMessagebox(
                title="Error",
                message=f"Could not open folder:\n{plugins_dir}",
                icon="warning"
            )
    
    def _on_window_close(self):
        """Handle window close"""
        
        if self.parent:
            self.parent.grab_set()  # Return focus to parent
        
        self.window.destroy()
    
    def show(self):
        """Show the plugin manager window"""
        self.window.deiconify()
        self.window.lift()
        self.window.focus()