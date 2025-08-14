#!/usr/bin/env python3
"""
Simple UpScale App GUI - Lightweight version without AI dependencies
For testing GUI functionality without heavy ML dependencies
"""

import sys
import os
import logging
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import subprocess
import signal

# Add src to path
src_path = str(Path(__file__).parent / "src")
sys.path.insert(0, src_path)
sys.path.insert(0, str(Path(__file__).parent))

try:
    import customtkinter as ctk
    from CTkMessagebox import CTkMessagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("GUI dependencies not available. Install with: pip install -r requirements_gui.txt")

# Configure CustomTkinter
if GUI_AVAILABLE:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "gui_debug.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=== GUI Debug Session Started ===")
    logger.info(f"Log file: {log_file}")
    return logger

# Initialize logger
logger = setup_logging()


class SimpleUpScaleGUI:
    """Simple GUI for UpScale App without AI dependencies"""
    
    def __init__(self):
        if not GUI_AVAILABLE:
            raise ImportError("GUI dependencies not available")
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing GUI...")
        
        self.ffmpeg_path = self._find_ffmpeg()
        self.logger.info(f"FFmpeg path: {self.ffmpeg_path}")
        
        # Process management
        self.current_process = None
        self.processing = False
        self.cancel_requested = False
        
        self._setup_window()
        self._setup_ui()
        
        self.logger.info("GUI initialization complete")
    
    def _find_ffmpeg(self):
        """Find FFmpeg executable"""
        import subprocess
        
        ffmpeg_paths = [
            "ffmpeg",
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\ffmpeg\ffmpeg.exe",
            "ffmpeg.exe"
        ]
        
        for ffmpeg_path in ffmpeg_paths:
            try:
                result = subprocess.run(
                    [ffmpeg_path, "-version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if result.returncode == 0:
                    return ffmpeg_path
            except:
                continue
        return None
        
    def _setup_window(self):
        """Setup main window"""
        self.root = ctk.CTk()
        self.root.title("UpScale App - Simple GUI v0.1.0")
        self.root.geometry("600x500")
        
        # Configure grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
    def _setup_ui(self):
        """Setup user interface"""
        
        # Main container
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame, 
            text="UpScale App - Simple GUI", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=(20, 10))
        
        self.subtitle_label = ctk.CTkLabel(
            self.main_frame,
            text="Basic Video File Processing Tool",
            font=ctk.CTkFont(size=14)
        )
        self.subtitle_label.pack(pady=(0, 20))
        
        # File selection section
        self._setup_file_section()
        
        # Settings section
        self._setup_settings_section()
        
        # Actions section
        self._setup_actions_section()
        
        # Status section
        self._setup_status_section()
        
    def _setup_file_section(self):
        """Setup file selection section"""
        
        file_frame = ctk.CTkFrame(self.main_frame)
        file_frame.pack(fill="x", padx=20, pady=10)
        
        # Input file
        input_label = ctk.CTkLabel(file_frame, text="Input Video:", font=ctk.CTkFont(weight="bold"))
        input_label.pack(anchor="w", padx=15, pady=(15, 5))
        
        input_container = ctk.CTkFrame(file_frame)
        input_container.pack(fill="x", padx=15, pady=(0, 15))
        
        self.input_entry = ctk.CTkEntry(
            input_container, 
            placeholder_text="Select a video file...",
            height=35
        )
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(10, 5), pady=10)
        
        self.browse_button = ctk.CTkButton(
            input_container,
            text="Browse",
            command=self._browse_input_file,
            width=80
        )
        self.browse_button.pack(side="right", padx=(5, 10), pady=10)
        
    def _setup_settings_section(self):
        """Setup settings section"""
        
        settings_frame = ctk.CTkFrame(self.main_frame)
        settings_frame.pack(fill="x", padx=20, pady=10)
        
        settings_label = ctk.CTkLabel(settings_frame, text="Settings", font=ctk.CTkFont(size=16, weight="bold"))
        settings_label.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Settings container
        settings_container = ctk.CTkFrame(settings_frame)
        settings_container.pack(fill="x", padx=15, pady=(0, 15))
        
        # Scale factor
        ctk.CTkLabel(settings_container, text="Scale Factor:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.scale_var = ctk.StringVar(value="1.5x")
        scale_menu = ctk.CTkComboBox(
            settings_container,
            values=["1.2x", "1.5x", "2.0x", "2.5x"],
            variable=self.scale_var,
            state="readonly",
            width=100
        )
        scale_menu.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Processing mode
        ctk.CTkLabel(settings_container, text="Mode:").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        self.mode_var = ctk.StringVar(value="Simple")
        
        # Check for waifu2x availability
        try:
            from config.settings import ENVIRONMENT_STATUS
            waifu2x_available = ENVIRONMENT_STATUS.get("waifu2x_available", False)
        except:
            waifu2x_available = False
        
        # Mode options based on availability
        if waifu2x_available:
            mode_options = ["Simple", "Waifu2x (High Quality)", "Basic Processing"]
        else:
            mode_options = ["Simple", "Basic Processing"]
            
        mode_menu = ctk.CTkComboBox(
            settings_container,
            values=mode_options,
            variable=self.mode_var,
            state="readonly",
            width=150
        )
        mode_menu.grid(row=0, column=3, padx=10, pady=10, sticky="w")
        
        # Waifu2x specific settings (row 1)
        if waifu2x_available:
            ctk.CTkLabel(settings_container, text="Noise Reduction:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
            self.noise_var = ctk.StringVar(value="1")
            noise_menu = ctk.CTkComboBox(
                settings_container,
                values=["0 (None)", "1 (Weak)", "2 (Medium)", "3 (Strong)"],
                variable=self.noise_var,
                state="readonly",
                width=100
            )
            noise_menu.grid(row=1, column=1, padx=10, pady=10, sticky="w")
            
            ctk.CTkLabel(settings_container, text="Model:").grid(row=1, column=2, padx=10, pady=10, sticky="w")
            self.model_var = ctk.StringVar(value="CUNet")
            model_menu = ctk.CTkComboBox(
                settings_container,
                values=["CUNet (Balanced)", "Anime Style", "Photo"],
                variable=self.model_var,
                state="readonly",
                width=150
            )
            model_menu.grid(row=1, column=3, padx=10, pady=10, sticky="w")
        
    def _setup_actions_section(self):
        """Setup actions section"""
        
        actions_frame = ctk.CTkFrame(self.main_frame)
        actions_frame.pack(fill="x", padx=20, pady=10)
        
        actions_label = ctk.CTkLabel(actions_frame, text="Actions", font=ctk.CTkFont(size=16, weight="bold"))
        actions_label.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Button container
        button_container = ctk.CTkFrame(actions_frame)
        button_container.pack(fill="x", padx=15, pady=(0, 15))
        
        self.analyze_button = ctk.CTkButton(
            button_container,
            text="Analyze Video",
            command=self._analyze_video,
            height=35,
            state="disabled"
        )
        self.analyze_button.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        
        # Process button
        self.process_button = ctk.CTkButton(
            button_container,
            text="Start Processing",
            command=self._start_processing,
            height=35,
            state="disabled",
            fg_color="green",
            hover_color="darkgreen"
        )
        self.process_button.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        
        # Cancel button
        self.cancel_button = ctk.CTkButton(
            button_container,
            text="Cancel",
            command=self._cancel_processing,
            height=35,
            state="disabled",
            fg_color="red",
            hover_color="darkred"
        )
        self.cancel_button.pack(side="left", padx=5, pady=10, fill="x", expand=True)
        
        self.test_button = ctk.CTkButton(
            button_container,
            text="Test FFmpeg",
            command=self._test_ffmpeg,
            height=35,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "gray90")
        )
        self.test_button.pack(side="right", padx=(5, 10), pady=10)
        
        # Add waifu2x test button if available
        try:
            from config.settings import ENVIRONMENT_STATUS
            if ENVIRONMENT_STATUS.get("waifu2x_available", False):
                self.waifu2x_test_button = ctk.CTkButton(
                    button_container,
                    text="Test Waifu2x",
                    command=self._test_waifu2x,
                    height=35,
                    fg_color="transparent",
                    border_width=2,
                    text_color=("gray10", "gray90")
                )
                self.waifu2x_test_button.pack(side="right", padx=(5, 5), pady=10)
        except:
            pass
        
    def _setup_status_section(self):
        """Setup status section"""
        
        status_frame = ctk.CTkFrame(self.main_frame)
        status_frame.pack(fill="x", padx=20, pady=10)
        
        status_label = ctk.CTkLabel(status_frame, text="Status", font=ctk.CTkFont(size=16, weight="bold"))
        status_label.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Progress bar
        self.progress_frame = ctk.CTkFrame(status_frame)
        self.progress_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        self.progress_label = ctk.CTkLabel(self.progress_frame, text="Progress: Ready")
        self.progress_label.pack(pady=(10, 5))
        
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.pack(fill="x", padx=15, pady=(0, 10))
        self.progress_bar.set(0)
        
        # Status display
        self.status_text = ctk.CTkTextbox(status_frame, height=100)
        self.status_text.pack(fill="x", padx=15, pady=(0, 15))
        self.status_text.insert("0.0", "Ready. Select a video file to begin.")
        
    def _browse_input_file(self):
        """Browse for input file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov"),
                ("MP4 files", "*.mp4"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, file_path)
            self.analyze_button.configure(state="normal")
            self.process_button.configure(state="disabled")  # Reset process button
            
            self.status_text.delete("0.0", "end")
            self.status_text.insert("0.0", f"Selected: {os.path.basename(file_path)}")
            
    def _analyze_video(self):
        """Analyze selected video"""
        input_path = self.input_entry.get().strip()
        
        if not input_path or not os.path.exists(input_path):
            CTkMessagebox(
                title="Error",
                message="Please select a valid video file",
                icon="cancel"
            )
            return
            
        try:
            # Basic file info
            file_size = os.path.getsize(input_path) / (1024*1024)  # MB
            
            # Try to get video info with opencv
            try:
                import cv2
                cap = cv2.VideoCapture(input_path)
                
                if not cap.isOpened():
                    raise Exception("Could not open video file")
                    
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                cap.release()
                
                # Calculate target resolution
                scale_factor = float(self.scale_var.get().rstrip('x'))
                target_width = int(width * scale_factor)
                target_height = int(height * scale_factor)
                
                info_text = f"""Video Information:
File: {os.path.basename(input_path)}
Size: {file_size:.1f} MB
Resolution: {width}x{height}
Frame Rate: {fps:.2f} fps
Duration: {duration:.2f} seconds
Frames: {frame_count}

Upscaling Preview:
Scale Factor: {self.scale_var.get()}
Target Resolution: {target_width}x{target_height}
Processing Mode: {self.mode_var.get()}

Status: Analysis complete - Ready for processing"""
                
            except ImportError:
                info_text = f"""Basic File Information:
File: {os.path.basename(input_path)}
Size: {file_size:.1f} MB

Note: Install opencv-python for detailed video analysis
Scale Factor: {self.scale_var.get()}
Processing Mode: {self.mode_var.get()}

Status: Basic analysis complete"""
                
            except Exception as e:
                info_text = f"""File Information:
File: {os.path.basename(input_path)}
Size: {file_size:.1f} MB

Error analyzing video: {str(e)}
This may not be a valid video file or the codec is not supported.

Status: Analysis failed"""
            
            self.status_text.delete("0.0", "end")
            self.status_text.insert("0.0", info_text)
            
            # Enable process button if analysis was successful
            if "Analysis complete" in info_text:
                self.process_button.configure(state="normal")
            
        except Exception as e:
            CTkMessagebox(
                title="Analysis Error",
                message=f"Failed to analyze video: {str(e)}",
                icon="cancel"
            )
            
    def _test_ffmpeg(self):
        """Test FFmpeg availability"""
        if self.ffmpeg_path:
            try:
                import subprocess
                result = subprocess.run(
                    [self.ffmpeg_path, "-version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                
                if result.returncode == 0:
                    version_line = result.stdout.split('\n')[0]
                    CTkMessagebox(
                        title="FFmpeg Test",
                        message=f"FFmpeg is available!\n\nPath: {self.ffmpeg_path}\n{version_line}",
                        icon="check"
                    )
                else:
                    CTkMessagebox(
                        title="FFmpeg Test",
                        message="FFmpeg found but returned error",
                        icon="warning"
                    )
            except Exception as e:
                CTkMessagebox(
                    title="FFmpeg Test",
                    message=f"Error testing FFmpeg: {str(e)}",
                    icon="cancel"
                )
        else:
            CTkMessagebox(
                title="FFmpeg Test",
                message="FFmpeg not found!\n\nNote: Video processing will use OpenCV instead.\nFFmpeg is optional but recommended for advanced features.",
                icon="warning"
            )
    
    def _test_waifu2x(self):
        """Test Waifu2x availability"""
        try:
            # Try to import and test waifu2x
            try:
                # Try multiple import paths
                try:
                    from src.modules.waifu2x_processor import test_waifu2x_availability, Waifu2xUpscaler
                except ImportError:
                    try:
                        from modules.waifu2x_processor import test_waifu2x_availability, Waifu2xUpscaler
                    except ImportError:
                        import sys
                        sys.path.append('src')
                        from modules.waifu2x_processor import test_waifu2x_availability, Waifu2xUpscaler
                
                availability = test_waifu2x_availability()
                
                if availability["any_available"]:
                    # Test actual initialization
                    try:
                        upscaler = Waifu2xUpscaler()
                        if upscaler.is_available():
                            backend_info = upscaler.get_info()
                            
                            message = f"""Waifu2x is available!
                            
Backend: {backend_info['backend']}
GPU ID: {backend_info['gpu_id']}
Scale: {backend_info['scale']}x
Noise Level: {backend_info['noise']}
Model: {backend_info['model']}

Supported Scales: {', '.join(map(str, backend_info['supported_scales']))}
Supported Noise Levels: {', '.join(map(str, backend_info['supported_noise_levels']))}"""
                            
                            CTkMessagebox(
                                title="Waifu2x Test",
                                message=message,
                                icon="check"
                            )
                        else:
                            CTkMessagebox(
                                title="Waifu2x Test",
                                message="Waifu2x imported but initialization failed",
                                icon="warning"
                            )
                    except Exception as e:
                        CTkMessagebox(
                            title="Waifu2x Test",
                            message=f"Waifu2x initialization failed: {str(e)}",
                            icon="warning"
                        )
                else:
                    backends_status = []
                    if not availability["ncnn"]:
                        backends_status.append("NCNN-Vulkan: Not available")
                    if not availability["chainer"]:
                        backends_status.append("Chainer: Not available")
                    
                    message = f"""Waifu2x not available.

{chr(10).join(backends_status)}

To install Waifu2x support:
pip install waifu2x-ncnn-vulkan-python

Note: Requires Vulkan-compatible GPU"""
                    
                    CTkMessagebox(
                        title="Waifu2x Test",
                        message=message,
                        icon="cancel"
                    )
                    
            except ImportError:
                CTkMessagebox(
                    title="Waifu2x Test",
                    message="Waifu2x module not found.\n\nInstall with:\npip install waifu2x-ncnn-vulkan-python",
                    icon="cancel"
                )
                
        except Exception as e:
            CTkMessagebox(
                title="Waifu2x Test",
                message=f"Error testing Waifu2x: {str(e)}",
                icon="cancel"
            )
    
    def _start_processing(self):
        """Start video processing with Waifu2x"""
        input_path = self.input_entry.get().strip()
        
        self.logger.info(f"=== Starting video processing ===")
        self.logger.info(f"Input path: {input_path}")
        
        if not input_path or not os.path.exists(input_path):
            self.logger.error(f"Invalid input path: {input_path}")
            CTkMessagebox(
                title="Error",
                message="Please select a valid video file first",
                icon="cancel"
            )
            return
        
        # Reset cancel flag and set processing state
        self.cancel_requested = False
        self.processing = True
        
        try:
            # Disable/enable appropriate buttons during processing
            self.process_button.configure(state="disabled")
            self.analyze_button.configure(state="disabled")
            self.cancel_button.configure(state="normal")
            
            # Update status
            self.status_text.delete("0.0", "end")
            self.status_text.insert("0.0", "Starting video processing...")
            self.progress_label.configure(text="Progress: Initializing...")
            self.progress_bar.set(0)
            self.root.update()
            
            # Get settings
            scale_factor = float(self.scale_var.get().rstrip('x'))
            mode = self.mode_var.get()
            
            self.logger.info(f"Settings - Scale: {scale_factor}, Mode: {mode}")
            
            # Import processing modules
            try:
                self.logger.info("Attempting to import Waifu2x modules...")
                
                # Try multiple import paths for Waifu2x only
                try:
                    self.logger.debug("Trying: from src.modules.waifu2x_processor import Waifu2xUpscaler")
                    from src.modules.waifu2x_processor import Waifu2xUpscaler
                    self.logger.info("Successfully imported from src.modules.waifu2x_processor")
                except ImportError as e1:
                    self.logger.debug(f"First import failed: {e1}")
                    try:
                        self.logger.debug("Trying: from modules.waifu2x_processor import Waifu2xUpscaler")
                        from modules.waifu2x_processor import Waifu2xUpscaler
                        self.logger.info("Successfully imported from modules.waifu2x_processor")
                    except ImportError as e2:
                        self.logger.debug(f"Second import failed: {e2}")
                        import sys
                        self.logger.debug(f"Current sys.path: {sys.path}")
                        sys.path.append('src')
                        self.logger.debug("Added 'src' to sys.path, trying again...")
                        from modules.waifu2x_processor import Waifu2xUpscaler
                        self.logger.info("Successfully imported after adding src to path")
                
                # Create output filename
                input_name = os.path.splitext(os.path.basename(input_path))[0]
                output_path = f"output/{input_name}_waifu2x_{scale_factor}x.mp4"
                os.makedirs("output", exist_ok=True)
                
                # Initialize processors
                if "Waifu2x" in mode:
                    # Get Waifu2x specific settings
                    noise_level = 1  # Default
                    model_type = "models-cunet"  # Default
                    
                    if hasattr(self, 'noise_var'):
                        noise_level = int(self.noise_var.get().split()[0])
                    if hasattr(self, 'model_var'):
                        model_map = {
                            "CUNet (Balanced)": "models-cunet",
                            "Anime Style": "models-upconv_7_anime_style_art_rgb",
                            "Photo": "models-upconv_7_photo"
                        }
                        model_type = model_map.get(self.model_var.get(), "models-cunet")
                    
                    upscaler = Waifu2xUpscaler(
                        scale=int(scale_factor),
                        noise=noise_level,
                        model=model_type
                    )
                else:
                    upscaler = None
                
                # Process video
                self._process_video_with_progress(
                    input_path, output_path, upscaler, scale_factor
                )
                
            except ImportError as e:
                self.logger.error(f"Failed to import Waifu2x modules: {e}")
                self.logger.error(f"Full traceback:", exc_info=True)
                
                # Fallback: use simple processing without advanced modules
                self.logger.info("Falling back to simple processing mode")
                self.status_text.delete("0.0", "end")
                self.status_text.insert("0.0", f"Import warning: {str(e)}\n\nUsing basic processing mode...")
                self.root.update()
                
                # Simple processing without advanced modules
                self._simple_video_processing(input_path, scale_factor)
                
        except Exception as e:
            self.logger.error(f"Unexpected error during processing: {e}")
            self.logger.error(f"Full traceback:", exc_info=True)
            
            CTkMessagebox(
                title="Processing Error",
                message=f"Failed to start processing: {str(e)}",
                icon="cancel"
            )
        finally:
            # Reset processing state
            self.processing = False
            self.current_process = None
            
            # Re-enable buttons
            self.process_button.configure(state="normal")
            self.analyze_button.configure(state="normal")
            self.cancel_button.configure(state="disabled")
            self.logger.info("=== Processing session ended ===")
    
    def _process_video_with_progress(self, input_path, output_path, upscaler, scale_factor):
        """Process video with progress updates"""
        try:
            import cv2
            import threading
            
            # Update status
            self.progress_label.configure(text="Progress: Extracting frames...")
            self.status_text.delete("0.0", "end")
            self.status_text.insert("0.0", "Extracting frames from video...")
            self.root.update()
            
            # Extract frames
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create temp directory
            temp_dir = "temp/frames"
            os.makedirs(temp_dir, exist_ok=True)
            
            frame_files = []
            for i in range(total_frames):
                # Check for cancellation
                self._check_cancel_requested()
                
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_path = f"{temp_dir}/frame_{i:06d}.png"
                cv2.imwrite(frame_path, frame)
                frame_files.append(frame_path)
                
                # Update progress
                progress = (i + 1) / total_frames * 0.3  # 30% for extraction
                self.progress_bar.set(progress)
                self.progress_label.configure(text=f"Progress: Extracting frames... {i+1}/{total_frames}")
                if i % 10 == 0:  # Update UI every 10 frames
                    self.root.update()
            
            cap.release()
            
            # Process frames with Waifu2x
            self.progress_label.configure(text="Progress: Upscaling frames...")
            self.status_text.delete("0.0", "end")
            self.status_text.insert("0.0", f"Upscaling {len(frame_files)} frames with Waifu2x...")
            self.root.update()
            
            upscaled_dir = "temp/upscaled"
            os.makedirs(upscaled_dir, exist_ok=True)
            
            def progress_callback(progress, message):
                # Update progress bar (30% + 60% for processing)
                total_progress = 0.3 + (progress / 100) * 0.6
                self.progress_bar.set(total_progress)
                self.progress_label.configure(text=f"Progress: {message}")
                self.root.update()
            
            if upscaler:
                upscaled_files = upscaler.upscale_frames(
                    frame_files, upscaled_dir, progress_callback
                )
            else:
                # Simple upscaling fallback
                upscaled_files = []
                for i, frame_file in enumerate(frame_files):
                    from PIL import Image
                    with Image.open(frame_file) as img:
                        width, height = img.size
                        new_size = (int(width * scale_factor), int(height * scale_factor))
                        upscaled = img.resize(new_size, Image.LANCZOS)
                        
                        output_file = f"{upscaled_dir}/frame_{i:06d}_upscaled.png"
                        upscaled.save(output_file)
                        upscaled_files.append(output_file)
                    
                    progress = (i + 1) / len(frame_files) * 100
                    progress_callback(progress, f"Simple upscaling {i+1}/{len(frame_files)}")
            
            # Rebuild video
            self.progress_label.configure(text="Progress: Rebuilding video...")
            self.status_text.delete("0.0", "end")
            self.status_text.insert("0.0", "Rebuilding video from upscaled frames...")
            self.root.update()
            
            if upscaled_files:
                self._rebuild_video(upscaled_files, output_path, fps)
                
                # Complete
                self.progress_bar.set(1.0)
                self.progress_label.configure(text="Progress: Complete!")
                
                result_text = f"""Processing Complete!

Input: {os.path.basename(input_path)}
Output: {output_path}
Scale Factor: {scale_factor}x
Frames Processed: {len(upscaled_files)}/{len(frame_files)}
Method: {"Waifu2x (Mock)" if upscaler else "Simple Upscaling"}

Output file saved successfully."""
                
                self.status_text.delete("0.0", "end")
                self.status_text.insert("0.0", result_text)
                
                CTkMessagebox(
                    title="Processing Complete",
                    message=f"Video processing complete!\n\nOutput saved to: {output_path}",
                    icon="check"
                )
            else:
                raise Exception("No frames were successfully processed")
                
        except Exception as e:
            self.progress_bar.set(0)
            self.progress_label.configure(text="Progress: Error occurred")
            
            error_text = f"""Processing Failed!

Error: {str(e)}

Please check:
- Input file is a valid video
- Sufficient disk space available
- All dependencies are installed"""
            
            self.status_text.delete("0.0", "end")
            self.status_text.insert("0.0", error_text)
            
            CTkMessagebox(
                title="Processing Error",
                message=f"Processing failed: {str(e)}",
                icon="cancel"
            )
    
    def _rebuild_video(self, frame_files, output_path, fps):
        """Rebuild video from frames"""
        try:
            import cv2
            
            if not frame_files:
                raise Exception("No frames to rebuild video")
            
            # Get frame dimensions
            sample_frame = cv2.imread(frame_files[0])
            height, width, channels = sample_frame.shape
            
            # Create temporary video without audio
            temp_video = output_path.replace('.mp4', '_temp.mp4')
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
            
            for i, frame_file in enumerate(frame_files):
                frame = cv2.imread(frame_file)
                if frame is not None:
                    out.write(frame)
                
                # Update progress
                progress = 0.9 + (i + 1) / len(frame_files) * 0.05  # 5% for video
                self.progress_bar.set(progress)
                if i % 10 == 0:
                    self.root.update()
            
            out.release()
            
            # Try to add audio from original video if FFmpeg is available
            if self.ffmpeg_path:
                self._add_audio_to_video(temp_video, output_path)
                import os
                os.remove(temp_video)  # Remove temp file
            else:
                # Just rename temp to final if no FFmpeg
                import os
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_video, output_path)
            
        except Exception as e:
            raise Exception(f"Failed to rebuild video: {str(e)}")
    
    def _add_audio_to_video(self, video_path, output_path):
        """Add audio from original video to processed video"""
        try:
            original_video = self.input_entry.get().strip()
            
            if self.ffmpeg_path and original_video:
                # Use FFmpeg to combine processed video with original audio
                cmd = [
                    self.ffmpeg_path,
                    "-i", video_path,
                    "-i", original_video,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-shortest",
                    "-y",  # Overwrite output
                    output_path
                ]
                
                # Start process with proper management
                self.current_process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                try:
                    stdout, stderr = self.current_process.communicate(timeout=300)
                    
                    if self.current_process.returncode != 0:
                        # If audio addition fails, just copy the video-only file
                        import shutil
                        shutil.copy2(video_path, output_path)
                        
                except subprocess.TimeoutExpired:
                    self._kill_current_process()
                    raise Exception("FFmpeg audio processing timeout")
                    
                # Update progress
                self.progress_bar.set(0.95)
                self.root.update()
                
        except Exception as e:
            # If audio addition fails, just copy the video-only file
            import shutil
            shutil.copy2(video_path, output_path)
        finally:
            self.current_process = None
    
    def _simple_video_processing(self, input_path, scale_factor):
        """Simple video processing fallback"""
        try:
            import cv2
            import numpy as np
            from PIL import Image
            
            # Create output filename
            input_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = f"output/{input_name}_simple_{scale_factor}x.mp4"
            os.makedirs("output", exist_ok=True)
            
            # Update status
            self.progress_label.configure(text="Progress: Simple processing mode...")
            self.status_text.delete("0.0", "end")
            self.status_text.insert("0.0", "Using simple processing mode...\nExtracting frames...")
            self.root.update()
            
            # Extract and process frames
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create temp directory
            temp_dir = "temp/simple_frames"
            os.makedirs(temp_dir, exist_ok=True)
            
            processed_frames = []
            
            for i in range(total_frames):
                # Check for cancellation
                self._check_cancel_requested()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to PIL for upscaling
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Simple upscaling
                width, height = pil_image.size
                new_size = (int(width * scale_factor), int(height * scale_factor))
                upscaled = pil_image.resize(new_size, Image.LANCZOS)
                
                # Convert back to OpenCV format
                upscaled_np = cv2.cvtColor(np.array(upscaled), cv2.COLOR_RGB2BGR)
                processed_frames.append(upscaled_np)
                
                # Update progress
                progress = (i + 1) / total_frames * 0.8  # 80% for processing
                self.progress_bar.set(progress)
                self.progress_label.configure(text=f"Progress: Processing frame {i+1}/{total_frames}")
                
                if i % 10 == 0:
                    self.root.update()
            
            cap.release()
            
            # Create output video
            if processed_frames:
                self.progress_label.configure(text="Progress: Creating output video...")
                self.status_text.delete("0.0", "end")
                self.status_text.insert("0.0", "Creating output video...")
                self.root.update()
                
                height, width, channels = processed_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                for i, frame in enumerate(processed_frames):
                    out.write(frame)
                    
                    # Update progress
                    progress = 0.8 + (i + 1) / len(processed_frames) * 0.2  # Final 20%
                    self.progress_bar.set(progress)
                    
                    if i % 10 == 0:
                        self.root.update()
                
                out.release()
                
                # Complete
                self.progress_bar.set(1.0)
                self.progress_label.configure(text="Progress: Complete!")
                
                result_text = f"""Simple Processing Complete!

Input: {os.path.basename(input_path)}
Output: {output_path}
Scale Factor: {scale_factor}x
Frames Processed: {len(processed_frames)}
Method: Simple LANCZOS Upscaling

Output file saved successfully."""
                
                self.status_text.delete("0.0", "end")
                self.status_text.insert("0.0", result_text)
                
                CTkMessagebox(
                    title="Processing Complete",
                    message=f"Simple video processing complete!\n\nOutput saved to: {output_path}",
                    icon="check"
                )
            else:
                raise Exception("No frames were processed")
                
        except Exception as e:
            self.progress_bar.set(0)
            self.progress_label.configure(text="Progress: Error occurred")
            
            error_text = f"""Simple Processing Failed!

Error: {str(e)}

Please check:
- Input file is a valid video
- OpenCV is installed properly
- Sufficient disk space available"""
            
            self.status_text.delete("0.0", "end")
            self.status_text.insert("0.0", error_text)
            
            CTkMessagebox(
                title="Processing Error",
                message=f"Simple processing failed: {str(e)}",
                icon="cancel"
            )
        finally:
            # Reset processing state
            self.processing = False
            
            # Re-enable buttons
            self.process_button.configure(state="normal")
            self.analyze_button.configure(state="normal")
            self.cancel_button.configure(state="disabled")
    
    def _cancel_processing(self):
        """Cancel current processing operation"""
        if not self.processing:
            return
            
        self.cancel_requested = True
        self.logger.info("Cancel requested by user")
        
        # Update UI
        self.progress_label.configure(text="Progress: Cancelling...")
        self.status_text.delete("0.0", "end")
        self.status_text.insert("0.0", "Cancelling processing...")
        self.root.update()
        
        # Kill current process if exists
        self._kill_current_process()
        
        # Show completion message
        CTkMessagebox(
            title="Processing Cancelled",
            message="Video processing has been cancelled.",
            icon="info"
        )
        
        # Reset UI
        self.progress_bar.set(0)
        self.progress_label.configure(text="Progress: Cancelled")
        self.status_text.delete("0.0", "end")
        self.status_text.insert("0.0", "Processing cancelled by user.")
    
    def _kill_current_process(self):
        """Kill the current subprocess if it exists"""
        if self.current_process is None:
            return
            
        try:
            if self.current_process.poll() is None:  # Process is still running
                self.logger.info(f"Terminating process PID: {self.current_process.pid}")
                
                if os.name == 'nt':  # Windows
                    # Use taskkill to terminate the entire process tree
                    subprocess.run([
                        "taskkill", "/F", "/T", "/PID", str(self.current_process.pid)
                    ], capture_output=True)
                else:  # Unix-like
                    # Send SIGTERM first, then SIGKILL if necessary
                    self.current_process.terminate()
                    try:
                        self.current_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.current_process.kill()
                        self.current_process.wait()
                
                self.logger.info("Process terminated successfully")
                
        except Exception as e:
            self.logger.error(f"Error terminating process: {e}")
        finally:
            self.current_process = None
    
    def _check_cancel_requested(self):
        """Check if cancellation was requested and handle it"""
        if self.cancel_requested:
            self._kill_current_process()
            raise Exception("Processing cancelled by user")
            
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()


def main():
    """Main entry point"""
    if not GUI_AVAILABLE:
        print("GUI dependencies not available")
        print("Install GUI dependencies with:")
        print("   pip install -r requirements_gui.txt")
        return 1
        
    try:
        app = SimpleUpScaleGUI()
        app.run()
        return 0
        
    except Exception as e:
        print(f"GUI Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())