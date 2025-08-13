"""
Real-time Preview Window
Interactive preview interface with real-time parameter adjustment
"""

import tkinter as tk
import sys
import os
from pathlib import Path
import threading
import time
from typing import List, Dict, Optional, Callable
from PIL import Image, ImageTk
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import customtkinter as ctk
    from CTkMessagebox import CTkMessagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

from modules.preview_engine import PreviewEngine, PreviewSettings, PreviewQuality, PreviewMode, PreviewRegion
from modules.ai_model_manager import model_manager
from modules.model_selector import model_selector, ProcessingConstraints, ProcessingPriority
from config.settings import PATHS


class InteractiveImageCanvas(tk.Canvas):
    """Interactive canvas for image display and ROI selection"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.image = None
        self.image_tk = None
        self.image_id = None
        
        # ROI selection
        self.roi_start = None
        self.roi_rect = None
        self.roi_callback: Optional[Callable] = None
        
        # Mouse interaction
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Double-Button-1>", self._on_double_click)
        
        # Zoom functionality
        self.scale_factor = 1.0
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<Button-4>", self._on_mousewheel)  # Linux
        self.bind("<Button-5>", self._on_mousewheel)  # Linux
    
    def display_image(self, image_array: np.ndarray):
        """Display numpy image array"""
        try:
            # Convert numpy array to PIL Image
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)
            
            # Handle different array shapes
            if len(image_array.shape) == 3:
                if image_array.shape[2] == 3:  # RGB
                    pil_image = Image.fromarray(image_array, 'RGB')
                elif image_array.shape[2] == 4:  # RGBA
                    pil_image = Image.fromarray(image_array, 'RGBA')
                else:
                    pil_image = Image.fromarray(image_array[:,:,0], 'L')  # Grayscale
            else:  # Grayscale
                pil_image = Image.fromarray(image_array, 'L')
            
            # Scale image to fit canvas
            canvas_width = self.winfo_width()
            canvas_height = self.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # Calculate scale to fit
                img_width, img_height = pil_image.size
                scale_x = canvas_width / img_width
                scale_y = canvas_height / img_height
                scale = min(scale_x, scale_y, 1.0) * self.scale_factor
                
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.image_tk = ImageTk.PhotoImage(pil_image)
            
            # Display on canvas
            self.delete("all")
            self.image_id = self.create_image(
                canvas_width // 2, canvas_height // 2,
                image=self.image_tk, anchor=tk.CENTER
            )
            
            # Store original for ROI calculations
            self.image = pil_image
            
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    def _on_click(self, event):
        """Handle mouse click for ROI selection"""
        self.roi_start = (event.x, event.y)
        if self.roi_rect:
            self.delete(self.roi_rect)
        self.roi_rect = self.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="red", width=2, dash=(5, 5)
        )
    
    def _on_drag(self, event):
        """Handle mouse drag for ROI selection"""
        if self.roi_start and self.roi_rect:
            x1, y1 = self.roi_start
            x2, y2 = event.x, event.y
            self.coords(self.roi_rect, x1, y1, x2, y2)
    
    def _on_release(self, event):
        """Handle mouse release for ROI selection"""
        if self.roi_start and self.roi_callback:
            x1, y1 = self.roi_start
            x2, y2 = event.x, event.y
            
            # Convert canvas coordinates to image coordinates
            roi = self._canvas_to_image_coords(min(x1, x2), min(y1, y2), 
                                             abs(x2 - x1), abs(y2 - y1))
            if roi and roi.width > 10 and roi.height > 10:  # Minimum size
                self.roi_callback(roi)
    
    def _on_double_click(self, event):
        """Handle double click to clear ROI"""
        if self.roi_rect:
            self.delete(self.roi_rect)
            self.roi_rect = None
        if self.roi_callback:
            self.roi_callback(None)  # Clear ROI
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel for zoom"""
        if event.delta:
            delta = event.delta
        else:  # Linux
            delta = -1 if event.num == 5 else 1
        
        # Zoom in/out
        zoom_factor = 1.1 if delta > 0 else 0.9
        self.scale_factor = max(0.1, min(3.0, self.scale_factor * zoom_factor))
        
        # Redisplay image with new scale
        if hasattr(self, '_last_image_array'):
            self.display_image(self._last_image_array)
    
    def _canvas_to_image_coords(self, x: int, y: int, width: int, height: int) -> Optional[PreviewRegion]:
        """Convert canvas coordinates to image coordinates"""
        if not self.image:
            return None
        
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        img_width, img_height = self.image.size
        
        # Calculate image position on canvas
        img_x = (canvas_width - img_width) // 2
        img_y = (canvas_height - img_height) // 2
        
        # Convert coordinates
        img_x_coord = max(0, x - img_x)
        img_y_coord = max(0, y - img_y)
        img_width_coord = min(width, img_width - img_x_coord)
        img_height_coord = min(height, img_height - img_y_coord)
        
        return PreviewRegion(
            x=img_x_coord, y=img_y_coord,
            width=img_width_coord, height=img_height_coord
        )
    
    def set_roi_callback(self, callback: Callable):
        """Set callback for ROI selection"""
        self.roi_callback = callback


class PreviewControlPanel(ctk.CTkFrame):
    """Control panel for preview settings"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.settings_callback: Optional[Callable] = None
        
        self._setup_controls()
        self._create_default_settings()
    
    def _setup_controls(self):
        """Setup control widgets"""
        
        # Title
        title_label = ctk.CTkLabel(
            self, text="Preview Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.pack(pady=(10, 5))
        
        # Model selection
        model_frame = ctk.CTkFrame(self)
        model_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(model_frame, text="AI Model:").pack(anchor="w", padx=5, pady=2)
        
        self.model_var = ctk.StringVar(value="Auto Select")
        self.model_menu = ctk.CTkComboBox(
            model_frame,
            variable=self.model_var,
            values=["Auto Select"],
            command=self._on_setting_changed,
            state="readonly"
        )
        self.model_menu.pack(fill="x", padx=5, pady=2)
        
        # Scale factor
        scale_frame = ctk.CTkFrame(self)
        scale_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(scale_frame, text="Scale Factor:").pack(anchor="w", padx=5, pady=2)
        
        self.scale_var = ctk.DoubleVar(value=1.5)
        self.scale_slider = ctk.CTkSlider(
            scale_frame,
            from_=1.0, to=4.0, number_of_steps=30,
            variable=self.scale_var,
            command=self._on_scale_changed
        )
        self.scale_slider.pack(fill="x", padx=5, pady=2)
        
        self.scale_label = ctk.CTkLabel(scale_frame, text="1.5x")
        self.scale_label.pack(padx=5, pady=2)
        
        # Preview quality
        quality_frame = ctk.CTkFrame(self)
        quality_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(quality_frame, text="Preview Quality:").pack(anchor="w", padx=5, pady=2)
        
        self.quality_var = ctk.StringVar(value="Medium")
        quality_menu = ctk.CTkComboBox(
            quality_frame,
            variable=self.quality_var,
            values=["Low", "Medium", "High"],
            command=self._on_setting_changed,
            state="readonly"
        )
        quality_menu.pack(fill="x", padx=5, pady=2)
        
        # Preview mode
        mode_frame = ctk.CTkFrame(self)
        mode_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(mode_frame, text="Display Mode:").pack(anchor="w", padx=5, pady=2)
        
        self.mode_var = ctk.StringVar(value="Split Vertical")
        mode_menu = ctk.CTkComboBox(
            mode_frame,
            variable=self.mode_var,
            values=["Split Vertical", "Split Horizontal", "Overlay", "Difference", "Original Only", "Processed Only"],
            command=self._on_setting_changed,
            state="readonly"
        )
        mode_menu.pack(fill="x", padx=5, pady=2)
        
        # Processing parameters
        params_frame = ctk.CTkFrame(self)
        params_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(params_frame, text="Processing Parameters:").pack(anchor="w", padx=5, pady=2)
        
        # Inference steps
        ctk.CTkLabel(params_frame, text="Inference Steps:", font=ctk.CTkFont(size=12)).pack(anchor="w", padx=10, pady=(10, 2))
        self.steps_var = ctk.IntVar(value=20)
        steps_slider = ctk.CTkSlider(
            params_frame,
            from_=10, to=50, number_of_steps=8,
            variable=self.steps_var,
            command=self._on_setting_changed
        )
        steps_slider.pack(fill="x", padx=10, pady=2)
        
        self.steps_label = ctk.CTkLabel(params_frame, text="20 steps")
        self.steps_label.pack(padx=10, pady=2)
        
        # Guidance scale
        ctk.CTkLabel(params_frame, text="Guidance Scale:", font=ctk.CTkFont(size=12)).pack(anchor="w", padx=10, pady=(10, 2))
        self.guidance_var = ctk.DoubleVar(value=7.5)
        guidance_slider = ctk.CTkSlider(
            params_frame,
            from_=1.0, to=20.0, number_of_steps=19,
            variable=self.guidance_var,
            command=self._on_setting_changed
        )
        guidance_slider.pack(fill="x", padx=10, pady=2)
        
        self.guidance_label = ctk.CTkLabel(params_frame, text="7.5")
        self.guidance_label.pack(padx=10, pady=2)
        
        # ROI controls
        roi_frame = ctk.CTkFrame(self)
        roi_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(roi_frame, text="Region of Interest:").pack(anchor="w", padx=5, pady=2)
        
        roi_controls = ctk.CTkFrame(roi_frame)
        roi_controls.pack(fill="x", padx=5, pady=2)
        
        self.roi_active_var = ctk.BooleanVar(value=False)
        roi_checkbox = ctk.CTkCheckBox(
            roi_controls,
            text="Use ROI",
            variable=self.roi_active_var,
            command=self._on_setting_changed
        )
        roi_checkbox.pack(side="left", padx=5)
        
        roi_clear_btn = ctk.CTkButton(
            roi_controls,
            text="Clear ROI",
            width=80,
            height=28,
            command=self._clear_roi
        )
        roi_clear_btn.pack(side="right", padx=5)
        
        self.roi_info_label = ctk.CTkLabel(roi_frame, text="No ROI selected")
        self.roi_info_label.pack(padx=5, pady=2)
        
        # Action buttons
        action_frame = ctk.CTkFrame(self)
        action_frame.pack(fill="x", padx=10, pady=10)
        
        refresh_btn = ctk.CTkButton(
            action_frame,
            text="🔄 Refresh Preview",
            command=self._refresh_preview,
            height=35
        )
        refresh_btn.pack(fill="x", padx=5, pady=5)
        
        save_btn = ctk.CTkButton(
            action_frame,
            text="💾 Save Preview",
            command=self._save_preview,
            height=35,
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "gray90")
        )
        save_btn.pack(fill="x", padx=5, pady=5)
    
    def _create_default_settings(self):
        """Create default preview settings"""
        self.current_settings = PreviewSettings()
        self.current_roi: Optional[PreviewRegion] = None
        
        # Update model list
        self._update_model_list()
    
    def _update_model_list(self):
        """Update available models list"""
        try:
            available_models = model_manager.get_available_models()
            model_names = ["Auto Select"] + [model.name for model in available_models]
            self.model_menu.configure(values=model_names)
        except:
            pass
    
    def _on_scale_changed(self, value):
        """Handle scale factor change"""
        self.scale_label.configure(text=f"{value:.1f}x")
        self._on_setting_changed()
    
    def _on_setting_changed(self, value=None):
        """Handle setting change"""
        # Update steps label
        steps_value = self.steps_var.get()
        self.steps_label.configure(text=f"{steps_value} steps")
        
        # Update guidance label
        guidance_value = self.guidance_var.get()
        self.guidance_label.configure(text=f"{guidance_value:.1f}")
        
        # Create updated settings
        settings = self._get_current_settings()
        
        if self.settings_callback:
            self.settings_callback(settings)
    
    def _get_current_settings(self) -> PreviewSettings:
        """Get current preview settings"""
        
        # Map quality
        quality_map = {
            "Low": PreviewQuality.LOW,
            "Medium": PreviewQuality.MEDIUM,
            "High": PreviewQuality.HIGH
        }
        
        # Map mode
        mode_map = {
            "Split Vertical": PreviewMode.SPLIT_VERTICAL,
            "Split Horizontal": PreviewMode.SPLIT_HORIZONTAL,
            "Overlay": PreviewMode.OVERLAY,
            "Difference": PreviewMode.DIFFERENCE,
            "Original Only": PreviewMode.ORIGINAL_ONLY,
            "Processed Only": PreviewMode.PROCESSED_ONLY
        }
        
        # Determine model ID
        model_name = self.model_var.get()
        model_id = None
        if model_name != "Auto Select":
            available_models = model_manager.get_available_models()
            for model in available_models:
                if model.name == model_name:
                    model_id = model.id
                    break
        
        settings = PreviewSettings(
            quality=quality_map[self.quality_var.get()],
            mode=mode_map[self.mode_var.get()],
            scale_factor=self.scale_var.get(),
            model_id=model_id,
            roi=self.current_roi if self.roi_active_var.get() else None,
            processing_params={
                "num_inference_steps": self.steps_var.get(),
                "guidance_scale": self.guidance_var.get()
            }
        )
        
        return settings
    
    def _refresh_preview(self):
        """Manual preview refresh"""
        settings = self._get_current_settings()
        if self.settings_callback:
            self.settings_callback(settings, force_refresh=True)
    
    def _save_preview(self):
        """Save current preview"""
        # This would be implemented to save the current preview image
        CTkMessagebox(
            title="Save Preview",
            message="Preview save functionality would be implemented here.",
            icon="info"
        )
    
    def _clear_roi(self):
        """Clear ROI selection"""
        self.current_roi = None
        self.roi_active_var.set(False)
        self.roi_info_label.configure(text="No ROI selected")
        self._on_setting_changed()
    
    def set_roi(self, roi: Optional[PreviewRegion]):
        """Set ROI from external source (e.g., canvas selection)"""
        self.current_roi = roi
        
        if roi:
            self.roi_active_var.set(True)
            self.roi_info_label.configure(
                text=f"ROI: {roi.width}×{roi.height} at ({roi.x},{roi.y})"
            )
        else:
            self.roi_active_var.set(False)
            self.roi_info_label.configure(text="No ROI selected")
        
        self._on_setting_changed()
    
    def set_settings_callback(self, callback: Callable):
        """Set callback for settings changes"""
        self.settings_callback = callback


class PreviewWindow:
    """Main preview window with interactive controls"""
    
    def __init__(self, parent=None, initial_image_path: str = None):
        if not GUI_AVAILABLE:
            raise ImportError("GUI dependencies not available")
        
        self.parent = parent
        self.preview_engine = PreviewEngine()
        self.current_image_path = initial_image_path
        
        self._setup_window()
        self._setup_ui()
        self._setup_preview_engine()
        
        # Load initial image if provided
        if initial_image_path and Path(initial_image_path).exists():
            self._load_initial_image()
    
    def _setup_window(self):
        """Setup preview window"""
        self.window = ctk.CTkToplevel(self.parent)
        self.window.title("Real-time Preview - UpScale App")
        self.window.geometry("1200x800")
        
        # Make it modal if parent exists
        if self.parent:
            self.window.transient(self.parent)
            # Don't grab_set for preview window to allow interaction
        
        # Configure grid
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        
        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self._on_window_close)
    
    def _setup_ui(self):
        """Setup user interface"""
        
        # Main container
        main_container = ctk.CTkFrame(self.window)
        main_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=3)
        main_container.grid_columnconfigure(1, weight=1)
        
        # Preview canvas
        self.canvas_frame = ctk.CTkFrame(main_container)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(5, 2), pady=5)
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Create canvas with scrollbars
        canvas_container = tk.Frame(self.canvas_frame)
        canvas_container.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)
        
        self.preview_canvas = InteractiveImageCanvas(
            canvas_container,
            bg="black",
            highlightthickness=0
        )
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self.preview_canvas.set_roi_callback(self._on_roi_selected)
        
        # Control panel
        self.control_panel = PreviewControlPanel(main_container)
        self.control_panel.grid(row=0, column=1, sticky="nsew", padx=(2, 5), pady=5)
        self.control_panel.set_settings_callback(self._on_settings_changed)
        
        # Status bar
        self.status_frame = ctk.CTkFrame(main_container)
        self.status_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=(0, 5))
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready - Select an image to preview",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(side="left", padx=10, pady=5)
        
        self.progress_bar = ctk.CTkProgressBar(self.status_frame)
        self.progress_bar.pack(side="right", padx=10, pady=5)
        self.progress_bar.set(0)
    
    def _setup_preview_engine(self):
        """Setup preview engine callbacks"""
        self.preview_engine.set_preview_callback(self._on_preview_complete)
        self.preview_engine.set_error_callback(self._on_preview_error)
    
    def _load_initial_image(self):
        """Load and display initial image"""
        try:
            import cv2
            
            # Load original image
            image = cv2.imread(self.current_image_path, cv2.IMREAD_COLOR)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.preview_canvas.display_image(image_rgb)
                self.status_label.configure(text=f"Loaded: {Path(self.current_image_path).name}")
                
                # Generate initial preview
                self._generate_preview()
                
        except Exception as e:
            self.status_label.configure(text=f"Error loading image: {str(e)}")
    
    def _on_settings_changed(self, settings: PreviewSettings, force_refresh: bool = False):
        """Handle settings change from control panel"""
        if self.current_image_path:
            self._generate_preview(settings, force_refresh)
    
    def _on_roi_selected(self, roi: Optional[PreviewRegion]):
        """Handle ROI selection from canvas"""
        self.control_panel.set_roi(roi)
    
    def _generate_preview(self, settings: Optional[PreviewSettings] = None, 
                         force_refresh: bool = False):
        """Generate preview with current settings"""
        
        if not self.current_image_path:
            return
        
        if settings is None:
            settings = self.control_panel._get_current_settings()
        
        # Clear cache if force refresh
        if force_refresh:
            self.preview_engine.clear_cache()
        
        self.status_label.configure(text="Generating preview...")
        self.progress_bar.set(0.3)
        
        # Generate preview asynchronously
        self.preview_engine.generate_preview(
            self.current_image_path,
            settings,
            async_mode=True
        )
    
    def _on_preview_complete(self, image_path: str, result):
        """Handle preview completion"""
        if image_path == self.current_image_path:
            # Update display
            self.preview_canvas.display_image(result.composite_image)
            
            # Update status
            processing_time = result.processing_time
            self.status_label.configure(
                text=f"Preview generated in {processing_time:.2f}s"
            )
            self.progress_bar.set(1.0)
            
            # Reset progress bar after delay
            self.window.after(2000, lambda: self.progress_bar.set(0))
    
    def _on_preview_error(self, image_path: str, error_msg: str):
        """Handle preview error"""
        if image_path == self.current_image_path:
            self.status_label.configure(text=f"Preview error: {error_msg}")
            self.progress_bar.set(0)
    
    def set_image(self, image_path: str):
        """Set new image for preview"""
        if Path(image_path).exists():
            self.current_image_path = image_path
            self._load_initial_image()
    
    def _on_window_close(self):
        """Handle window close"""
        try:
            self.preview_engine.shutdown()
        except:
            pass
        
        if self.parent:
            self.parent.focus()
        
        self.window.destroy()
    
    def show(self):
        """Show the preview window"""
        self.window.deiconify()
        self.window.lift()
        self.window.focus()


def main():
    """Test preview window"""
    if not GUI_AVAILABLE:
        print("GUI dependencies not available")
        return
    
    # Create test window
    root = ctk.CTk()
    root.title("Preview Window Test")
    root.geometry("400x300")
    
    def open_preview():
        preview = PreviewWindow(root)
        preview.show()
    
    btn = ctk.CTkButton(root, text="Open Preview", command=open_preview)
    btn.pack(expand=True)
    
    root.mainloop()


if __name__ == "__main__":
    main()