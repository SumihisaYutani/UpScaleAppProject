"""
Main GUI Window for UpScale App
Modern GUI using CustomTkinter
"""

import tkinter as tk
import sys
import os
from pathlib import Path
import threading
import time
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import customtkinter as ctk
    from CTkMessagebox import CTkMessagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("GUI dependencies not available. Install with: pip install -r requirements_gui.txt")

from enhanced_upscale_app import EnhancedUpScaleApp
from config.settings import VIDEO_SETTINGS, AI_SETTINGS, ENVIRONMENT_STATUS

# Configure CustomTkinter
if GUI_AVAILABLE:
    ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class ProgressDialog(ctk.CTkToplevel):
    """Progress dialog for processing"""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        self.title("Processing Video")
        self.geometry("500x300")
        self.resizable(False, False)
        
        # Make it modal
        self.transient(parent)
        self.grab_set()
        
        # Center on parent
        parent.update_idletasks()
        x = (parent.winfo_x() + parent.winfo_width() // 2) - 250
        y = (parent.winfo_y() + parent.winfo_height() // 2) - 150
        self.geometry(f"+{x}+{y}")
        
        self._setup_ui()
        
        self.cancelled = False
        
    def _setup_ui(self):
        """Setup progress dialog UI"""
        
        # Main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame, 
            text="🎬 Processing Video with AI Upscaling", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.title_label.pack(pady=(10, 20))
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self.main_frame, 
            text="Initializing...",
            font=ctk.CTkFont(size=14)
        )
        self.status_label.pack(pady=5)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.main_frame)
        self.progress_bar.pack(fill="x", padx=20, pady=10)
        self.progress_bar.set(0)
        
        # Progress percentage
        self.progress_label = ctk.CTkLabel(
            self.main_frame, 
            text="0%",
            font=ctk.CTkFont(size=12)
        )
        self.progress_label.pack(pady=5)
        
        # System stats frame
        self.stats_frame = ctk.CTkFrame(self.main_frame)
        self.stats_frame.pack(fill="x", padx=10, pady=10)
        
        self.stats_label = ctk.CTkLabel(
            self.stats_frame,
            text="System: Initializing...",
            font=ctk.CTkFont(size=11)
        )
        self.stats_label.pack(pady=10)
        
        # Cancel button
        self.cancel_button = ctk.CTkButton(
            self.main_frame,
            text="Cancel",
            command=self._on_cancel,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "gray90")
        )
        self.cancel_button.pack(pady=(10, 0))
        
    def update_progress(self, progress: float, message: str):
        """Update progress display"""
        self.progress_bar.set(progress / 100)
        self.progress_label.configure(text=f"{progress:.1f}%")
        self.status_label.configure(text=message)
        
        # Update window
        self.update()
        
    def update_stats(self, stats: str):
        """Update system stats"""
        self.stats_label.configure(text=f"System: {stats}")
        self.update()
        
    def _on_cancel(self):
        """Handle cancel button"""
        self.cancelled = True
        self.cancel_button.configure(text="Cancelling...", state="disabled")


class MainWindow:
    """Main GUI Window for UpScale App"""
    
    def __init__(self):
        if not GUI_AVAILABLE:
            raise ImportError("GUI dependencies not available")
        
        # Initialize the app
        self.app = None
        self.processing_thread = None
        self.is_processing = False
        
        # Initialize GUI
        self._setup_window()
        self._setup_ui()
        self._setup_bindings()
        
        # Initialize backend app
        self._initialize_backend()
        
    def _setup_window(self):
        """Setup main window"""
        self.root = ctk.CTk()
        self.root.title("UpScale App - AI Video Upscaling Tool v0.2.0")
        self.root.geometry("800x700")
        
        # Set icon (if available)
        try:
            # You would set an icon here if you have one
            pass
        except:
            pass
            
        # Configure grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
    def _setup_ui(self):
        """Setup user interface"""
        
        # Main container
        self.main_frame = ctk.CTkScrollableFrame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame, 
            text="🎬 UpScale App", 
            font=ctk.CTkFont(size=28, weight="bold")
        )
        self.title_label.pack(pady=(10, 5))
        
        self.subtitle_label = ctk.CTkLabel(
            self.main_frame,
            text="AI-Powered Video Upscaling Tool",
            font=ctk.CTkFont(size=16)
        )
        self.subtitle_label.pack(pady=(0, 20))
        
        # File selection section
        self._setup_file_section()
        
        # Settings section
        self._setup_settings_section()
        
        # Preview section
        self._setup_preview_section()
        
        # Processing section
        self._setup_processing_section()
        
        # Status section
        self._setup_status_section()
        
    def _setup_file_section(self):
        """Setup file selection section"""
        
        # File section frame
        file_frame = ctk.CTkFrame(self.main_frame)
        file_frame.pack(fill="x", padx=10, pady=10)
        
        # Input file
        input_label = ctk.CTkLabel(file_frame, text="📁 Input Video:", font=ctk.CTkFont(weight="bold"))
        input_label.pack(anchor="w", padx=15, pady=(15, 5))
        
        input_container = ctk.CTkFrame(file_frame)
        input_container.pack(fill="x", padx=15, pady=(0, 10))
        
        self.input_entry = ctk.CTkEntry(
            input_container, 
            placeholder_text="Select a video file...",
            height=35
        )
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(10, 5), pady=10)
        
        self.browse_input_button = ctk.CTkButton(
            input_container,
            text="Browse",
            command=self._browse_input_file,
            width=80
        )
        self.browse_input_button.pack(side="right", padx=(5, 10), pady=10)
        
        # Output folder
        output_label = ctk.CTkLabel(file_frame, text="📂 Output Folder:", font=ctk.CTkFont(weight="bold"))
        output_label.pack(anchor="w", padx=15, pady=(10, 5))
        
        output_container = ctk.CTkFrame(file_frame)
        output_container.pack(fill="x", padx=15, pady=(0, 15))
        
        self.output_entry = ctk.CTkEntry(
            output_container,
            placeholder_text="Select output folder...",
            height=35
        )
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(10, 5), pady=10)
        
        self.browse_output_button = ctk.CTkButton(
            output_container,
            text="Browse", 
            command=self._browse_output_folder,
            width=80
        )
        self.browse_output_button.pack(side="right", padx=(5, 10), pady=10)
        
    def _setup_settings_section(self):
        """Setup settings section"""
        
        settings_frame = ctk.CTkFrame(self.main_frame)
        settings_frame.pack(fill="x", padx=10, pady=10)
        
        settings_label = ctk.CTkLabel(settings_frame, text="⚙️ Settings", font=ctk.CTkFont(size=16, weight="bold"))
        settings_label.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Settings grid
        settings_grid = ctk.CTkFrame(settings_frame)
        settings_grid.pack(fill="x", padx=15, pady=(0, 15))
        
        # Scale factor
        ctk.CTkLabel(settings_grid, text="Scale Factor:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.scale_var = ctk.StringVar(value="1.5x")
        scale_menu = ctk.CTkComboBox(
            settings_grid,
            values=["1.2x", "1.5x", "2.0x", "2.5x"],
            variable=self.scale_var,
            state="readonly",
            width=100
        )
        scale_menu.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Quality preset
        ctk.CTkLabel(settings_grid, text="Quality:").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        self.quality_var = ctk.StringVar(value="Balanced")
        quality_menu = ctk.CTkComboBox(
            settings_grid,
            values=["Fast", "Balanced", "Quality"],
            variable=self.quality_var,
            state="readonly",
            width=120
        )
        quality_menu.grid(row=0, column=3, padx=10, pady=10, sticky="w")
        
        # AI Model
        ctk.CTkLabel(settings_grid, text="AI Model:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        ai_available = ENVIRONMENT_STATUS.get("ai_available", False)
        waifu2x_amd_available = ENVIRONMENT_STATUS.get("waifu2x_amd_available", False)
        
        ai_options = []
        if ai_available:
            ai_options.extend(["Enhanced AI", "Basic AI"])
        if waifu2x_amd_available:
            ai_options.append("Waifu2x AMD GPU")
        if ENVIRONMENT_STATUS.get("waifu2x_available", False):
            ai_options.append("Waifu2x")
        if not ai_options:
            ai_options = ["Simple Only"]
        if "Simple" not in ai_options:
            ai_options.append("Simple")
        self.ai_var = ctk.StringVar(value=ai_options[0])
        ai_menu = ctk.CTkComboBox(
            settings_grid,
            values=ai_options,
            variable=self.ai_var,
            state="readonly",
            width=120
        )
        ai_menu.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        # Advanced settings button
        self.advanced_button = ctk.CTkButton(
            settings_grid,
            text="Advanced Settings",
            command=self._show_advanced_settings,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "gray90")
        )
        self.advanced_button.grid(row=1, column=2, columnspan=2, padx=10, pady=10, sticky="e")
        
    def _setup_preview_section(self):
        """Setup preview section"""
        
        preview_frame = ctk.CTkFrame(self.main_frame)
        preview_frame.pack(fill="x", padx=10, pady=10)
        
        preview_label = ctk.CTkLabel(preview_frame, text="🎬 Preview", font=ctk.CTkFont(size=16, weight="bold"))
        preview_label.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Preview container
        preview_container = ctk.CTkFrame(preview_frame)
        preview_container.pack(fill="x", padx=15, pady=(0, 10))
        
        # Info display
        self.info_text = ctk.CTkTextbox(preview_container, height=100)
        self.info_text.pack(fill="x", padx=10, pady=10)
        self.info_text.insert("0.0", "Select a video file to see information...")
        self.info_text.configure(state="disabled")
        
        # Preview button
        self.preview_button = ctk.CTkButton(
            preview_container,
            text="Analyze Video",
            command=self._analyze_video,
            state="disabled"
        )
        self.preview_button.pack(pady=(0, 15))
        
    def _setup_processing_section(self):
        """Setup processing section"""
        
        processing_frame = ctk.CTkFrame(self.main_frame)
        processing_frame.pack(fill="x", padx=10, pady=10)
        
        processing_label = ctk.CTkLabel(processing_frame, text="🚀 Processing", font=ctk.CTkFont(size=16, weight="bold"))
        processing_label.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Button container
        button_container = ctk.CTkFrame(processing_frame)
        button_container.pack(fill="x", padx=15, pady=(0, 15))
        
        self.process_button = ctk.CTkButton(
            button_container,
            text="Start Processing",
            command=self._start_processing,
            height=40,
            font=ctk.CTkFont(size=16, weight="bold"),
            state="disabled"
        )
        self.process_button.pack(side="left", padx=10, pady=15, fill="x", expand=True)
        
        self.batch_button = ctk.CTkButton(
            button_container,
            text="Batch Mode",
            command=self._show_batch_mode,
            height=40,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "gray90")
        )
        self.batch_button.pack(side="right", padx=10, pady=15)
        
    def _setup_status_section(self):
        """Setup status section"""
        
        status_frame = ctk.CTkFrame(self.main_frame)
        status_frame.pack(fill="x", padx=10, pady=10)
        
        status_label = ctk.CTkLabel(status_frame, text="📊 System Status", font=ctk.CTkFont(size=16, weight="bold"))
        status_label.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Status display
        self.status_text = ctk.CTkTextbox(status_frame, height=80)
        self.status_text.pack(fill="x", padx=15, pady=(0, 15))
        
        # Initial status
        self._update_system_status()
        
    def _setup_bindings(self):
        """Setup event bindings"""
        
        # File drag and drop (if available)
        try:
            # This would require tkinterdnd2
            pass
        except:
            pass
            
        # Entry change detection
        self.input_entry.bind("<KeyRelease>", self._on_input_change)
        
    def _initialize_backend(self):
        """Initialize backend application"""
        try:
            ai_selection = self.ai_var.get()
            use_ai = ai_selection not in ["Simple Only", "Simple"]
            use_enhanced = "Enhanced" in ai_selection
            use_waifu2x_amd = "Waifu2x AMD GPU" in ai_selection
            
            # Configure waifu2x backend if AMD GPU selected
            if use_waifu2x_amd:
                from config.settings import WAIFU2X_SETTINGS
                WAIFU2X_SETTINGS["backend"] = "amd"
            
            self.app = EnhancedUpScaleApp(
                use_ai=use_ai,
                use_enhanced_ai=use_enhanced,
                enable_monitoring=True
            )
            
            self._update_system_status()
            
        except Exception as e:
            self._show_error(f"Failed to initialize backend: {e}")
            
    def _browse_input_file(self):
        """Browse for input file"""
        from tkinter import filedialog
        
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, file_path)
            self._on_input_change(None)
            
    def _browse_output_folder(self):
        """Browse for output folder"""
        from tkinter import filedialog
        
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        
        if folder_path:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, folder_path)
            
    def _on_input_change(self, event):
        """Handle input file change"""
        input_path = self.input_entry.get().strip()
        
        if input_path and os.path.exists(input_path):
            self.preview_button.configure(state="normal")
            self.process_button.configure(state="normal")
        else:
            self.preview_button.configure(state="disabled")
            self.process_button.configure(state="disabled")
            
    def _analyze_video(self):
        """Analyze selected video"""
        input_path = self.input_entry.get().strip()
        
        if not input_path or not self.app:
            return
            
        try:
            # Validate video file
            validation = self.app.video_processor.validate_video_file(input_path)
            
            self.info_text.configure(state="normal")
            self.info_text.delete("0.0", "end")
            
            if validation["valid"]:
                info = validation["info"]
                
                info_text = f"""✅ Valid Video File
                
📹 File Information:
• Duration: {info['duration']:.2f} seconds  
• Resolution: {info['width']}x{info['height']}
• Codec: {info['codec_name']}
• Frame Rate: {info['frame_rate']:.2f} fps
• Frame Count: {info['frame_count']}
• File Size: {info['size'] / (1024*1024):.1f} MB

🔍 Upscaling Preview:
• Scale Factor: {self.scale_var.get()}
• Target Resolution: {int(info['width'] * float(self.scale_var.get().rstrip('x')))}x{int(info['height'] * float(self.scale_var.get().rstrip('x')))}

⏱️ Estimated Processing Time:
• With GPU: {self.app.video_processor.estimate_processing_time(info)['estimated_gpu_minutes']:.1f} minutes
• With CPU: {self.app.video_processor.estimate_processing_time(info)['estimated_cpu_minutes']:.1f} minutes
"""
            else:
                info_text = f"❌ Invalid Video File\n\nError: {validation['error']}"
                
            self.info_text.insert("0.0", info_text)
            self.info_text.configure(state="disabled")
            
        except Exception as e:
            self._show_error(f"Failed to analyze video: {e}")
            
    def _start_processing(self):
        """Start video processing"""
        
        if self.is_processing:
            return
            
        input_path = self.input_entry.get().strip()
        output_folder = self.output_entry.get().strip()
        
        if not input_path:
            self._show_error("Please select an input video file")
            return
            
        if not output_folder:
            output_folder = str(Path(input_path).parent / "output")
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, output_folder)
            
        # Create output path
        output_filename = Path(input_path).stem + "_upscaled.mp4"
        output_path = str(Path(output_folder) / output_filename)
        
        # Get settings
        scale_factor = float(self.scale_var.get().rstrip('x'))
        quality_preset = self.quality_var.get().lower()
        
        # Show progress dialog
        progress_dialog = ProgressDialog(self.root)
        
        # Start processing in background thread
        self.is_processing = True
        self.process_button.configure(text="Processing...", state="disabled")
        
        def process_video():
            try:
                def progress_callback(progress, message):
                    if progress_dialog.cancelled:
                        return
                        
                    # Update progress dialog
                    self.root.after(0, lambda: progress_dialog.update_progress(progress, message))
                    
                    # Update system stats
                    if hasattr(self.app, 'monitor') and self.app.monitor:
                        try:
                            stats = self.app.monitor.get_current_stats()
                            system_info = stats.get('system', {})
                            stats_text = f"CPU: {system_info.get('cpu_percent', 0):.1f}% | RAM: {system_info.get('memory_percent', 0):.1f}% | GPU: {system_info.get('gpu_memory_used_gb', 0):.1f}GB"
                            self.root.after(0, lambda: progress_dialog.update_stats(stats_text))
                        except:
                            pass
                
                # Process video
                result = self.app.process_video_enhanced(
                    input_path,
                    output_path,
                    scale_factor,
                    progress_callback=progress_callback
                )
                
                # Handle result
                self.root.after(0, lambda: self._on_processing_complete(result, progress_dialog))
                
            except Exception as e:
                self.root.after(0, lambda: self._on_processing_error(str(e), progress_dialog))
        
        self.processing_thread = threading.Thread(target=process_video, daemon=True)
        self.processing_thread.start()
        
    def _on_processing_complete(self, result, progress_dialog):
        """Handle processing completion"""
        progress_dialog.destroy()
        
        self.is_processing = False
        self.process_button.configure(text="Start Processing", state="normal")
        
        if result["success"]:
            CTkMessagebox(
                title="Success!",
                message=f"Video processing completed successfully!\n\nOutput saved to:\n{result['output_path']}",
                icon="check"
            )
        else:
            self._show_error(f"Processing failed: {result.get('error', 'Unknown error')}")
            
    def _on_processing_error(self, error, progress_dialog):
        """Handle processing error"""
        progress_dialog.destroy()
        
        self.is_processing = False
        self.process_button.configure(text="Start Processing", state="normal")
        
        self._show_error(f"Processing failed: {error}")
        
    def _show_advanced_settings(self):
        """Show advanced settings dialog"""
        CTkMessagebox(
            title="Advanced Settings", 
            message="Advanced settings will be available in the next update!",
            icon="info"
        )
        
    def _show_batch_mode(self):
        """Show batch processing mode"""
        CTkMessagebox(
            title="Batch Mode",
            message="Batch processing will be available in the next update!",
            icon="info"
        )
        
    def _update_system_status(self):
        """Update system status display"""
        if not self.app:
            return
            
        try:
            system_info = self.app.get_system_info_enhanced()
            
            # Get AMD GPU information
            try:
                from modules.amd_gpu_detector import get_amd_gpu_info
                amd_info = get_amd_gpu_info()
            except:
                amd_info = {'amd_gpus_found': 0, 'amd_gpus': []}
            
            status_text = f"""🖥️ System Information:
• Platform: {system_info.get('platform', 'Unknown')}
• Python: {system_info.get('python_version', 'Unknown')}
• CUDA Available: {'Yes' if system_info.get('cuda_available') else 'No'}
"""
            
            # Add GPU information
            if system_info.get('cuda_available'):
                status_text += f"• NVIDIA GPU: {system_info.get('cuda_device_name', 'Unknown')}\n"
            
            if amd_info.get('amd_gpus_found', 0) > 0:
                amd_gpu = amd_info['amd_gpus'][0]
                status_text += f"• AMD GPU: {amd_gpu.get('name', 'Unknown')} ({amd_gpu.get('memory', 0) // 1024**3}GB)\n"
                
            # Enhanced AI status
            enhanced_ai_available = system_info.get('enhanced_ai_enabled')
            waifu2x_available = ENVIRONMENT_STATUS.get("waifu2x_available", False)
            amd_waifu2x_available = ENVIRONMENT_STATUS.get("waifu2x_amd_available", False)
            
            if enhanced_ai_available:
                status_text += "• Enhanced AI: Available\n"
            elif amd_waifu2x_available:
                status_text += "• AMD GPU AI: Available\n"
            elif waifu2x_available:
                status_text += "• Waifu2x AI: Available\n"
            else:
                status_text += "• Enhanced AI: Not Available\n"
            
            # Remove trailing newline
            status_text = status_text.rstrip()
            
            self.status_text.configure(state="normal")
            self.status_text.delete("0.0", "end")
            self.status_text.insert("0.0", status_text)
            self.status_text.configure(state="disabled")
            
        except Exception as e:
            print(f"Failed to update system status: {e}")
            
    def _show_error(self, message):
        """Show error message"""
        CTkMessagebox(
            title="Error",
            message=message,
            icon="cancel"
        )
        
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()
        
    def cleanup(self):
        """Cleanup resources"""
        if self.app:
            self.app.cleanup()


def main():
    """Main GUI entry point"""
    if not GUI_AVAILABLE:
        print("❌ GUI dependencies not available")
        print("📦 Install GUI dependencies with:")
        print("   pip install -r requirements_gui.txt")
        return 1
        
    try:
        app = MainWindow()
        app.run()
        app.cleanup()
        return 0
        
    except Exception as e:
        print(f"💥 GUI Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())