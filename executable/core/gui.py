"""
UpScale App - GUI Module
Simplified GUI using standard tkinter for better compatibility
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .utils import ProgressCallback, SimpleTimer, format_duration

logger = logging.getLogger(__name__)

class ProgressDialog:
    """Progress dialog for processing operations"""
    
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Processing Video")
        self.window.geometry("500x350")
        self.window.resizable(False, False)
        
        # Make modal
        self.window.transient(parent)
        self.window.grab_set()
        
        # Center on parent
        parent.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - 250
        y = parent.winfo_y() + (parent.winfo_height() // 2) - 175
        self.window.geometry(f"+{x}+{y}")
        
        self.cancelled = False
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup progress dialog UI"""
        # Main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Processing Video with AI Upscaling",
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=(0, 15))
        
        # Status
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_label.pack(pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame, 
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill="x", pady=10)
        
        # Progress percentage
        self.progress_text_var = tk.StringVar(value="0%")
        progress_text_label = ttk.Label(main_frame, textvariable=self.progress_text_var)
        progress_text_label.pack(pady=5)
        
        # Log display
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding=10)
        log_frame.pack(fill="both", expand=True, pady=(10, 0))
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=8,
            width=60,
            font=("Consolas", 9)
        )
        self.log_text.pack(fill="both", expand=True)
        
        # Cancel button
        self.cancel_button = ttk.Button(
            main_frame,
            text="Cancel",
            command=self._on_cancel
        )
        self.cancel_button.pack(pady=(15, 0))
        
    def update_progress(self, progress: float, message: str):
        """Update progress display"""
        self.progress_var.set(progress)
        self.progress_text_var.set(f"{progress:.1f}%")
        self.status_var.set(message)
        self.window.update_idletasks()
        
    def add_log_message(self, message: str):
        """Add log message to display"""
        import time
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\\n"
        
        self.log_text.insert(tk.END, formatted_message)
        self.log_text.see(tk.END)
        self.window.update_idletasks()
        
    def _on_cancel(self):
        """Handle cancel button"""
        self.cancelled = True
        self.cancel_button.config(text="Cancelling...", state="disabled")
        
    def destroy(self):
        """Destroy progress dialog"""
        try:
            self.window.destroy()
        except:
            pass

class MainGUI:
    """Main GUI application window"""
    
    def __init__(self, video_processor, ai_processor, gpu_info):
        self.video_processor = video_processor
        self.ai_processor = ai_processor  
        self.gpu_info = gpu_info
        
        self.root = None
        self.processing_thread = None
        self.is_processing = False
        
        # GUI variables
        self.input_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.scale_var = tk.StringVar(value="2.0")
        self.quality_var = tk.StringVar(value="Balanced")
        
    def _create_window(self):
        """Create main window"""
        self.root = tk.Tk()
        self.root.title("UpScale App - AI Video Upscaling Tool v2.0.0")
        self.root.geometry("700x600")
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')  # Modern looking theme
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup main user interface"""
        # Main container with scrolling
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Title section
        self._setup_title_section(scrollable_frame)
        
        # File selection section
        self._setup_file_section(scrollable_frame)
        
        # Settings section  
        self._setup_settings_section(scrollable_frame)
        
        # System info section
        self._setup_system_section(scrollable_frame)
        
        # Processing section
        self._setup_processing_section(scrollable_frame)
        
    def _setup_title_section(self, parent):
        """Setup title section"""
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill="x", padx=20, pady=20)
        
        title_label = ttk.Label(
            title_frame,
            text="UpScale App",
            font=("Arial", 24, "bold")
        )
        title_label.pack()
        
        subtitle_label = ttk.Label(
            title_frame,
            text="AI-Powered Video Upscaling Tool - Executable Edition",
            font=("Arial", 12)
        )
        subtitle_label.pack(pady=(5, 0))
        
    def _setup_file_section(self, parent):
        """Setup file selection section"""
        file_frame = ttk.LabelFrame(parent, text="File Selection", padding=15)
        file_frame.pack(fill="x", padx=20, pady=10)
        
        # Input file
        ttk.Label(file_frame, text="Input Video:").grid(row=0, column=0, sticky="w", pady=5)
        
        input_frame = ttk.Frame(file_frame)
        input_frame.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=5)
        input_frame.columnconfigure(0, weight=1)
        
        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_path_var)
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        ttk.Button(
            input_frame, 
            text="Browse", 
            command=self._browse_input
        ).grid(row=0, column=1)
        
        # Output folder
        ttk.Label(file_frame, text="Output Folder:").grid(row=1, column=0, sticky="w", pady=5)
        
        output_frame = ttk.Frame(file_frame)
        output_frame.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=5)
        output_frame.columnconfigure(0, weight=1)
        
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_path_var)
        self.output_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        ttk.Button(
            output_frame,
            text="Browse",
            command=self._browse_output
        ).grid(row=0, column=1)
        
        file_frame.columnconfigure(1, weight=1)
        
    def _setup_settings_section(self, parent):
        """Setup settings section"""
        settings_frame = ttk.LabelFrame(parent, text="Processing Settings", padding=15)
        settings_frame.pack(fill="x", padx=20, pady=10)
        
        # Scale factor
        ttk.Label(settings_frame, text="Scale Factor:").grid(row=0, column=0, sticky="w", padx=5)
        scale_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.scale_var,
            values=["1.5", "2.0", "2.5", "3.0", "4.0"],
            state="readonly",
            width=10
        )
        scale_combo.grid(row=0, column=1, padx=5, sticky="w")
        
        # Quality setting
        ttk.Label(settings_frame, text="Quality:").grid(row=0, column=2, sticky="w", padx=5)
        quality_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.quality_var,
            values=["Fast", "Balanced", "Quality"],
            state="readonly",
            width=12
        )
        quality_combo.grid(row=0, column=3, padx=5, sticky="w")
        
    def _setup_system_section(self, parent):
        """Setup system information section"""
        system_frame = ttk.LabelFrame(parent, text="System Information", padding=15)
        system_frame.pack(fill="x", padx=20, pady=10)
        
        # Get AI backend info
        backend_info = self.ai_processor.get_backend_info()
        gpu_summary = self.gpu_info.get('best_backend', 'cpu')
        
        info_text = f"""GPU Backend: {gpu_summary.upper()}
AI Processor: {backend_info.get('backend', 'unknown')}
GPU Mode: {'Yes' if backend_info.get('gpu_mode', False) else 'No'}
Total GPUs: {sum(1 for gpu_type in ['nvidia', 'amd'] if self.gpu_info.get(gpu_type, {}).get('available', False))}"""
        
        info_label = ttk.Label(system_frame, text=info_text, justify="left")
        info_label.pack(anchor="w")
        
    def _setup_processing_section(self, parent):
        """Setup processing section"""
        process_frame = ttk.LabelFrame(parent, text="Processing", padding=15)
        process_frame.pack(fill="x", padx=20, pady=20)
        
        self.process_button = ttk.Button(
            process_frame,
            text="Start Processing",
            command=self._start_processing,
            state="disabled"
        )
        self.process_button.pack(pady=10)
        
        # Status display
        self.status_text = scrolledtext.ScrolledText(
            process_frame,
            height=6,
            width=70,
            font=("Consolas", 9)
        )
        self.status_text.pack(fill="both", expand=True, pady=(10, 0))
        
        self._add_status_message("Ready - Select input video to begin")
        
        # Bind input change event
        self.input_path_var.trace("w", self._on_input_change)
        
    def _add_status_message(self, message: str):
        """Add message to status display"""
        import time
        timestamp = time.strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}\\n"
        
        self.status_text.insert(tk.END, formatted)
        self.status_text.see(tk.END)
        
    def _browse_input(self):
        """Browse for input video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("MOV files", "*.mov"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.input_path_var.set(file_path)
            
    def _browse_output(self):
        """Browse for output folder"""
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        
        if folder_path:
            self.output_path_var.set(folder_path)
            
    def _on_input_change(self, *args):
        """Handle input file change"""
        input_path = self.input_path_var.get().strip()
        
        if input_path and Path(input_path).exists():
            self.process_button.config(state="normal")
            self._add_status_message(f"Video selected: {Path(input_path).name}")
        else:
            self.process_button.config(state="disabled")
            
    def _start_processing(self):
        """Start video processing"""
        if self.is_processing:
            return
            
        input_path = self.input_path_var.get().strip()
        output_folder = self.output_path_var.get().strip()
        
        if not input_path:
            messagebox.showerror("Error", "Please select an input video file")
            return
            
        if not output_folder:
            output_folder = str(Path(input_path).parent / "output")
            self.output_path_var.set(output_folder)
            
        # Create output directory
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        output_filename = f"{Path(input_path).stem}_upscaled.mp4"
        output_path = str(Path(output_folder) / output_filename)
        
        # Get settings
        try:
            scale_factor = float(self.scale_var.get())
        except ValueError:
            scale_factor = 2.0
            
        # Show progress dialog
        progress_dialog = ProgressDialog(self.root)
        
        # Start processing thread
        self.is_processing = True
        self.process_button.config(text="Processing...", state="disabled")
        
        def process_video():
            """Background video processing"""
            timer = SimpleTimer()
            
            try:
                progress_dialog.add_log_message("Starting video processing...")
                
                def progress_callback(progress, message):
                    if not progress_dialog.cancelled:
                        self.root.after(0, lambda: progress_dialog.update_progress(progress, message))
                        self.root.after(0, lambda: progress_dialog.add_log_message(message))
                
                # Step 1: Validate video
                progress_callback(5, "Validating video file...")
                validation = self.video_processor.validate_video(input_path)
                
                if not validation['valid']:
                    raise RuntimeError(f"Invalid video: {validation['error']}")
                
                video_info = validation['info']
                progress_dialog.add_log_message(f"Video: {video_info['width']}x{video_info['height']}, {video_info['duration']:.1f}s")
                
                # Step 2: Extract frames
                progress_callback(10, "Extracting frames...")
                frame_paths = self.video_processor.extract_frames(input_path, progress_callback)
                
                if not frame_paths:
                    raise RuntimeError("Failed to extract frames")
                
                progress_dialog.add_log_message(f"Extracted {len(frame_paths)} frames")
                
                # Step 3: Upscale frames
                progress_callback(20, "Upscaling frames with AI...")
                upscaled_frames = self.ai_processor.upscale_frames(
                    frame_paths, 
                    str(self.video_processor.temp_dir / "upscaled"),
                    scale_factor,
                    progress_callback
                )
                
                if not upscaled_frames:
                    raise RuntimeError("Failed to upscale frames")
                
                progress_dialog.add_log_message(f"Upscaled {len(upscaled_frames)} frames")
                
                # Step 4: Combine to video
                progress_callback(90, "Combining frames to video...")
                success = self.video_processor.combine_frames_to_video(
                    upscaled_frames,
                    output_path,
                    input_path,
                    video_info.get('frame_rate', 30.0),
                    progress_callback
                )
                
                if not success:
                    raise RuntimeError("Failed to combine frames to video")
                
                # Completed
                progress_callback(100, "Processing complete!")
                processing_time = timer.elapsed_str()
                
                progress_dialog.add_log_message(f"Processing completed in {processing_time}")
                progress_dialog.add_log_message(f"Output saved: {output_path}")
                
                # Show completion dialog after short delay
                self.root.after(2000, lambda: self._on_processing_complete(output_path, progress_dialog))
                
            except Exception as e:
                error_msg = str(e)
                progress_dialog.add_log_message(f"Error: {error_msg}")
                self.root.after(1000, lambda: self._on_processing_error(error_msg, progress_dialog))
        
        self.processing_thread = threading.Thread(target=process_video, daemon=True)
        self.processing_thread.start()
        
    def _on_processing_complete(self, output_path, progress_dialog):
        """Handle processing completion"""
        progress_dialog.destroy()
        
        self.is_processing = False
        self.process_button.config(text="Start Processing", state="normal")
        
        messagebox.showinfo(
            "Success!",
            f"Video processing completed successfully!\\n\\nOutput saved to:\\n{output_path}"
        )
        
        self._add_status_message(f"Processing completed: {Path(output_path).name}")
        
    def _on_processing_error(self, error_msg, progress_dialog):
        """Handle processing error"""
        progress_dialog.destroy()
        
        self.is_processing = False
        self.process_button.config(text="Start Processing", state="normal")
        
        messagebox.showerror("Processing Error", f"Processing failed:\\n\\n{error_msg}")
        self._add_status_message(f"Processing failed: {error_msg}")
        
    def run(self) -> int:
        """Run the GUI application"""
        try:
            self._create_window()
            
            logger.info("Starting GUI application")
            self.root.mainloop()
            
            return 0
            
        except Exception as e:
            logger.error(f"GUI error: {e}")
            return 1
            
    def cleanup(self):
        """Cleanup GUI resources"""
        try:
            if self.root:
                self.root.quit()
        except:
            pass