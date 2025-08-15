"""
UpScale App - GUI Module
Modern GUI using CustomTkinter for better compatibility
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog
import threading
import logging
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import customtkinter as ctk
    from CTkMessagebox import CTkMessagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    # Fallback to standard tkinter
    from tkinter import ttk, messagebox, scrolledtext
    ctk = None

from .utils import ProgressCallback, SimpleTimer, format_duration

logger = logging.getLogger(__name__)

# Configure CustomTkinter
if GUI_AVAILABLE and ctk:
    ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class ProcessingStepTracker:
    """実行ステップごとのステータス追跡クラス"""
    
    STEPS = [
        {"id": "validate", "name": "動画検証", "emoji": "🔍", "weight": 5},
        {"id": "extract", "name": "フレーム抽出", "emoji": "📸", "weight": 15}, 
        {"id": "upscale", "name": "AI処理", "emoji": "🤖", "weight": 70},
        {"id": "combine", "name": "動画結合", "emoji": "🎬", "weight": 10}
    ]
    
    def __init__(self):
        self.current_step = 0
        self.step_progress = {}
        self.step_timers = {}
        self.step_status = {}
        
        # Initialize all steps
        for i, step in enumerate(self.STEPS):
            self.step_progress[step["id"]] = 0.0
            self.step_timers[step["id"]] = None
            self.step_status[step["id"]] = "pending"  # pending, active, completed, error
    
    def start_step(self, step_id: str):
        """指定ステップの開始"""
        import time
        if step_id in self.step_timers:
            self.step_timers[step_id] = time.time()
            self.step_status[step_id] = "active"
            
    def update_step(self, step_id: str, progress: float):
        """ステップ内の進捗更新"""
        if step_id in self.step_progress:
            self.step_progress[step_id] = min(100.0, max(0.0, progress))
            
    def complete_step(self, step_id: str):
        """ステップ完了"""
        import time
        if step_id in self.step_status:
            self.step_status[step_id] = "completed"
            self.step_progress[step_id] = 100.0
            if self.step_timers[step_id]:
                elapsed = time.time() - self.step_timers[step_id]
                self.step_timers[step_id] = elapsed
    
    def error_step(self, step_id: str):
        """ステップエラー"""
        if step_id in self.step_status:
            self.step_status[step_id] = "error"
    
    def get_overall_progress(self) -> float:
        """全体の進捗計算"""
        total_weight = sum(step["weight"] for step in self.STEPS)
        completed_weight = 0.0
        
        for step in self.STEPS:
            step_id = step["id"]
            if self.step_status[step_id] == "completed":
                completed_weight += step["weight"]
            elif self.step_status[step_id] == "active":
                step_progress = self.step_progress[step_id] / 100.0
                completed_weight += step["weight"] * step_progress
                
        return (completed_weight / total_weight) * 100.0
    
    def get_step_info(self, step_id: str) -> dict:
        """ステップ情報取得"""
        for step in self.STEPS:
            if step["id"] == step_id:
                return {
                    "name": step["name"],
                    "emoji": step["emoji"],
                    "progress": self.step_progress[step_id],
                    "status": self.step_status[step_id],
                    "timer": self.step_timers[step_id]
                }
        return {}

class ProgressDialog:
    """Progress dialog for processing operations"""
    
    def __init__(self, parent):
        self.parent = parent
        self.step_tracker = ProcessingStepTracker()
        
        if GUI_AVAILABLE and ctk:
            # Use CustomTkinter
            self.window = ctk.CTkToplevel(parent)
            self.window.title("Processing Video")
            self.window.geometry("875x650")
            self.window.resizable(False, False)
            
            # Make modal
            self.window.transient(parent)
            self.window.grab_set()
            
            # Center on parent
            parent.update_idletasks()
            x = (parent.winfo_x() + parent.winfo_width() // 2) - 300
            y = (parent.winfo_y() + parent.winfo_height() // 2) - 200
            self.window.geometry(f"+{x}+{y}")
        else:
            # Fallback to standard tkinter
            self.window = tk.Toplevel(parent)
            self.window.title("Processing Video")
            self.window.geometry("625x572")
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
        self.current_process = None  # Track current subprocess
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup progress dialog UI"""
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            # Main frame
            self.main_frame = ctk.CTkFrame(self.window)
            self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Title
            self.title_label = ctk.CTkLabel(
                self.main_frame, 
                text="🎬 Processing Video with AI Upscaling", 
                font=ctk.CTkFont(size=18, weight="bold")
            )
            self.title_label.pack(pady=(10, 20))
            
            # Overall progress section
            overall_frame = ctk.CTkFrame(self.main_frame)
            overall_frame.pack(fill="x", padx=10, pady=(0, 10))
            
            # Overall status label
            self.status_label = ctk.CTkLabel(
                overall_frame, 
                text="Initializing...",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            self.status_label.pack(pady=(10, 5))
            
            # Overall progress bar
            self.progress_bar = ctk.CTkProgressBar(overall_frame)
            self.progress_bar.pack(fill="x", padx=20, pady=5)
            self.progress_bar.set(0)
            
            # Overall progress percentage
            self.progress_label = ctk.CTkLabel(
                overall_frame, 
                text="0%",
                font=ctk.CTkFont(size=12)
            )
            self.progress_label.pack(pady=(0, 10))
            
            # Step progress section
            steps_frame = ctk.CTkFrame(self.main_frame)
            steps_frame.pack(fill="x", padx=10, pady=(0, 10))
            
            steps_title = ctk.CTkLabel(
                steps_frame,
                text="📋 Processing Steps",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            steps_title.pack(pady=(10, 5))
            
            # Create step displays
            self.step_displays = {}
            for step in self.step_tracker.STEPS:
                step_container = ctk.CTkFrame(steps_frame)
                step_container.pack(fill="x", padx=10, pady=2)
                
                # Step info frame (left side)
                info_frame = ctk.CTkFrame(step_container)
                info_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
                
                # Step name with emoji and status
                step_label = ctk.CTkLabel(
                    info_frame,
                    text=f"{step['emoji']} {step['name']}",
                    font=ctk.CTkFont(size=12),
                    anchor="w"
                )
                step_label.pack(side="left", padx=5)
                
                # Step status indicator
                status_label = ctk.CTkLabel(
                    info_frame,
                    text="⏳",
                    font=ctk.CTkFont(size=12)
                )
                status_label.pack(side="right", padx=5)
                
                # Step progress bar (right side)
                progress_frame = ctk.CTkFrame(step_container)
                progress_frame.pack(side="right", padx=5, pady=5)
                
                step_progress = ctk.CTkProgressBar(progress_frame, width=100)
                step_progress.pack(padx=5, pady=2)
                step_progress.set(0)
                
                step_percent = ctk.CTkLabel(
                    progress_frame,
                    text="0%",
                    font=ctk.CTkFont(size=10)
                )
                step_percent.pack(padx=5)
                
                self.step_displays[step["id"]] = {
                    "container": step_container,
                    "status_label": status_label,
                    "progress_bar": step_progress,
                    "percent_label": step_percent
                }
            
            steps_frame.pack(pady=(0, 10))
            
            # Log button section
            log_button_frame = ctk.CTkFrame(self.main_frame)
            log_button_frame.pack(fill="x", padx=10, pady=(0, 10))
            
            self.log_button = ctk.CTkButton(
                log_button_frame,
                text="📋 詳細ログを表示",
                command=self._show_log_window,
                width=150,
                height=30
            )
            self.log_button.pack(pady=10)
            
            # Initialize log window as None
            self.log_window = None
            self.log_messages = []
            
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
        else:
            # Standard tkinter fallback
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
            
            # Log button
            log_button_frame = ttk.Frame(main_frame)
            log_button_frame.pack(fill="x", pady=(10, 0))
            
            self.log_button = ttk.Button(
                log_button_frame,
                text="詳細ログを表示",
                command=self._show_log_window
            )
            self.log_button.pack(pady=10)
            
            # Initialize log window as None
            self.log_window = None
            self.log_messages = []
            
            # Cancel button
            self.cancel_button = ttk.Button(
                main_frame,
                text="Cancel",
                command=self._on_cancel
            )
            self.cancel_button.pack(pady=(15, 0))
        
    def update_progress(self, progress: float, message: str):
        """Update progress display"""
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            self.progress_bar.set(progress / 100)
            self.progress_label.configure(text=f"{progress:.1f}%")
            self.status_label.configure(text=message)
            # Update window
            self.window.update()
        else:
            # Standard tkinter fallback
            self.progress_var.set(progress)
            self.progress_text_var.set(f"{progress:.1f}%")
            self.status_var.set(message)
            self.window.update_idletasks()
    
    def update_step_progress(self, step_id: str, progress: float, status: str = None):
        """Update individual step progress"""
        if step_id not in self.step_displays:
            return
            
        # Update step tracker
        if status == "start":
            self.step_tracker.start_step(step_id)
        elif status == "complete":
            self.step_tracker.complete_step(step_id)
        elif status == "error":
            self.step_tracker.error_step(step_id)
        else:
            self.step_tracker.update_step(step_id, progress)
        
        # Update UI
        if GUI_AVAILABLE and ctk:
            display = self.step_displays[step_id]
            
            # Update progress bar and percentage
            display["progress_bar"].set(progress / 100)
            display["percent_label"].configure(text=f"{progress:.0f}%")
            
            # Update status indicator
            step_status = self.step_tracker.step_status[step_id]
            if step_status == "pending":
                display["status_label"].configure(text="⏳")
            elif step_status == "active":
                display["status_label"].configure(text="🔄")
            elif step_status == "completed":
                display["status_label"].configure(text="✅")
            elif step_status == "error":
                display["status_label"].configure(text="❌")
            
            # Update overall progress
            overall_progress = self.step_tracker.get_overall_progress()
            self.progress_bar.set(overall_progress / 100)
            self.progress_label.configure(text=f"{overall_progress:.1f}%")
            
            # Update window
            self.window.update()
        
    def add_log_message(self, message: str):
        """Add log message to display"""
        import time
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # Store message in list
        self.log_messages.append(formatted_message)
        
        # Update log window if it's open
        if self.log_window and hasattr(self.log_window, 'log_text'):
            try:
                self.log_window.log_text.insert("end", formatted_message + "\n")
                self.log_window.log_text.see("end")
                self.log_window.update()
            except:
                pass  # Window might be closed
    
    def _show_log_window(self):
        """Show separate log window"""
        if self.log_window is None or not self.log_window.winfo_exists():
            if GUI_AVAILABLE and ctk:
                # CustomTkinter version
                self.log_window = ctk.CTkToplevel(self.window)
                self.log_window.title("処理ログ")
                self.log_window.geometry("600x400")
                
                # Position relative to main dialog
                x = self.window.winfo_x() + self.window.winfo_width() + 10
                y = self.window.winfo_y()
                self.log_window.geometry(f"+{x}+{y}")
                
                # Log display
                log_frame = ctk.CTkFrame(self.log_window)
                log_frame.pack(fill="both", expand=True, padx=10, pady=10)
                
                log_label = ctk.CTkLabel(
                    log_frame,
                    text="📋 処理ログメッセージ",
                    font=ctk.CTkFont(size=14, weight="bold")
                )
                log_label.pack(pady=(5, 10))
                
                # Scrollable text area
                self.log_window.log_text = ctk.CTkTextbox(
                    log_frame,
                    font=ctk.CTkFont(size=10, family="Consolas")
                )
                self.log_window.log_text.pack(fill="both", expand=True, padx=5, pady=5)
            else:
                # Standard tkinter version
                self.log_window = tk.Toplevel(self.window)
                self.log_window.title("処理ログ")
                self.log_window.geometry("600x400")
                
                # Position relative to main dialog
                x = self.window.winfo_x() + self.window.winfo_width() + 10
                y = self.window.winfo_y()
                self.log_window.geometry(f"+{x}+{y}")
                
                # Log display
                log_frame = ttk.LabelFrame(self.log_window, text="処理ログメッセージ", padding=10)
                log_frame.pack(fill="both", expand=True, padx=10, pady=10)
                
                from tkinter import scrolledtext
                self.log_window.log_text = scrolledtext.ScrolledText(
                    log_frame,
                    font=("Consolas", 9)
                )
                self.log_window.log_text.pack(fill="both", expand=True)
            
            # Add existing messages
            for msg in self.log_messages:
                self.log_window.log_text.insert("end", msg + "\n")
            self.log_window.log_text.see("end")
        else:
            # Bring window to front
            self.log_window.lift()
            self.log_window.focus()
        
    def _on_cancel(self):
        """Handle cancel button"""
        self.cancelled = True
        
        # Update button state
        if GUI_AVAILABLE and ctk:
            self.cancel_button.configure(text="Cancelling...", state="disabled")
        else:
            self.cancel_button.config(text="Cancelling...", state="disabled")
        
        # Kill any running subprocess forcefully
        if self.current_process and self.current_process.poll() is None:
            try:
                # Try graceful termination first
                self.current_process.terminate()
                try:
                    self.current_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # Force kill if terminate didn't work
                    self.current_process.kill()
                    self.current_process.wait(timeout=3)
                    
                # Also kill any child processes on Windows
                import os
                import signal
                if os.name == 'nt':  # Windows
                    try:
                        import psutil
                        parent = psutil.Process(self.current_process.pid)
                        for child in parent.children(recursive=True):
                            child.kill()
                        parent.kill()
                    except:
                        # Fallback: use taskkill on Windows
                        try:
                            import subprocess
                            subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.current_process.pid)], 
                                         capture_output=True, timeout=5)
                        except:
                            pass
            except Exception as e:
                print(f"Error killing process: {e}")
        
        # Kill all FFmpeg processes as final measure
        try:
            import subprocess
            subprocess.run(['taskkill', '/F', '/IM', 'ffmpeg.exe'], capture_output=True, timeout=5)
        except:
            pass
        
        # Close progress dialog after short delay
        def close_dialog():
            try:
                self.destroy()
            except:
                pass
        
        if GUI_AVAILABLE and ctk:
            self.window.after(2000, close_dialog)
        else:
            self.window.after(2000, close_dialog)
        
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
        if GUI_AVAILABLE and ctk:
            # Use CustomTkinter
            self.root = ctk.CTk()
            self.root.title("UpScale App - AI Video Upscaling Tool v2.0.0")
            self.root.geometry("1200x1100")
            
            # Configure grid
            self.root.grid_rowconfigure(0, weight=1)
            self.root.grid_columnconfigure(0, weight=1)
        else:
            # Fallback to standard tkinter
            self.root = tk.Tk()
            self.root.title("UpScale App - AI Video Upscaling Tool v2.0.0")
            self.root.geometry("1000x875")
            
            # Configure style
            style = ttk.Style()
            style.theme_use('clam')  # Modern looking theme
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup main user interface"""
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            # Main container
            self.main_frame = ctk.CTkScrollableFrame(self.root)
            self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
            
            # Title section
            self._setup_title_section(self.main_frame)
            
            # File selection section
            self._setup_file_section(self.main_frame)
            
            # Settings section  
            self._setup_settings_section(self.main_frame)
            
            # System info section
            self._setup_system_section(self.main_frame)
            
            # Processing section
            self._setup_processing_section(self.main_frame)
            
            # Status section
            self._setup_status_section(self.main_frame)
        else:
            # Standard tkinter fallback
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
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            # Title
            self.title_label = ctk.CTkLabel(
                parent, 
                text="🎬 UpScale App", 
                font=ctk.CTkFont(size=28, weight="bold")
            )
            self.title_label.pack(pady=(10, 5))
            
            self.subtitle_label = ctk.CTkLabel(
                parent,
                text="AI-Powered Video Upscaling Tool",
                font=ctk.CTkFont(size=16)
            )
            self.subtitle_label.pack(pady=(0, 20))
        else:
            # Standard tkinter fallback
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
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            # File section frame
            file_frame = ctk.CTkFrame(parent)
            file_frame.pack(fill="x", padx=10, pady=10)
            
            # Input file
            input_label = ctk.CTkLabel(file_frame, text="📁 Input Video:", font=ctk.CTkFont(weight="bold"))
            input_label.pack(anchor="w", padx=15, pady=(15, 5))
            
            input_container = ctk.CTkFrame(file_frame)
            input_container.pack(fill="x", padx=15, pady=(0, 10))
            
            self.input_entry = ctk.CTkEntry(
                input_container, 
                placeholder_text="Select a video file...",
                height=35,
                textvariable=self.input_path_var
            )
            self.input_entry.pack(side="left", fill="x", expand=True, padx=(10, 5), pady=10)
            
            self.browse_input_button = ctk.CTkButton(
                input_container,
                text="Browse",
                command=self._browse_input,
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
                height=35,
                textvariable=self.output_path_var
            )
            self.output_entry.pack(side="left", fill="x", expand=True, padx=(10, 5), pady=10)
            
            self.browse_output_button = ctk.CTkButton(
                output_container,
                text="Browse", 
                command=self._browse_output,
                width=80
            )
            self.browse_output_button.pack(side="right", padx=(5, 10), pady=10)
        else:
            # Standard tkinter fallback
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
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            settings_frame = ctk.CTkFrame(parent)
            settings_frame.pack(fill="x", padx=10, pady=10)
            
            settings_label = ctk.CTkLabel(settings_frame, text="⚙️ Settings", font=ctk.CTkFont(size=16, weight="bold"))
            settings_label.pack(anchor="w", padx=15, pady=(15, 10))
            
            # Settings grid
            settings_grid = ctk.CTkFrame(settings_frame)
            settings_grid.pack(fill="x", padx=15, pady=(0, 15))
            
            # Scale factor
            ctk.CTkLabel(settings_grid, text="Scale Factor:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
            scale_menu = ctk.CTkComboBox(
                settings_grid,
                values=["1.2x", "1.5x", "2.0x", "2.5x", "3.0x", "4.0x"],
                variable=self.scale_var,
                state="readonly",
                width=100
            )
            scale_menu.grid(row=0, column=1, padx=10, pady=10, sticky="w")
            
            # Quality preset
            ctk.CTkLabel(settings_grid, text="Quality:").grid(row=0, column=2, padx=10, pady=10, sticky="w")
            quality_menu = ctk.CTkComboBox(
                settings_grid,
                values=["Fast", "Balanced", "Quality"],
                variable=self.quality_var,
                state="readonly",
                width=120
            )
            quality_menu.grid(row=0, column=3, padx=10, pady=10, sticky="w")
        else:
            # Standard tkinter fallback
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
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            system_frame = ctk.CTkFrame(parent)
            system_frame.pack(fill="x", padx=10, pady=10)
            
            system_label = ctk.CTkLabel(system_frame, text="💻 System Information", font=ctk.CTkFont(size=16, weight="bold"))
            system_label.pack(anchor="w", padx=15, pady=(15, 10))
        else:
            # Standard tkinter fallback  
            system_frame = ttk.LabelFrame(parent, text="System Information", padding=15)
            system_frame.pack(fill="x", padx=20, pady=10)
        
        # Get AI backend info
        backend_info = self.ai_processor.get_backend_info()
        gpu_summary = self.gpu_info.get('best_backend', 'cpu')
        
        # Count all available GPUs - ensure proper counting
        total_gpus = 0
        gpu_names = []
        all_gpu_details = []
        
        # Log GPU info for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"GPU Info in GUI: {self.gpu_info}")
        
        # Count NVIDIA GPUs
        nvidia_info = self.gpu_info.get('nvidia', {})
        if nvidia_info.get('available', False):
            nvidia_gpus = nvidia_info.get('gpus', [])
            for gpu in nvidia_gpus:
                gpu_name = gpu.get('name', 'Unknown NVIDIA')
                gpu_names.append(gpu_name)
                all_gpu_details.append(f"NVIDIA: {gpu_name}")
            logger.info(f"Found {len(nvidia_gpus)} NVIDIA GPUs: {[g.get('name') for g in nvidia_gpus]}")
        
        # Count AMD GPUs
        amd_info = self.gpu_info.get('amd', {})
        if amd_info.get('available', False):
            amd_gpus = amd_info.get('gpus', [])
            for gpu in amd_gpus:
                gpu_name = gpu.get('name', 'Unknown AMD')
                gpu_names.append(gpu_name)
                all_gpu_details.append(f"AMD: {gpu_name}")
            logger.info(f"Found {len(amd_gpus)} AMD GPUs: {[g.get('name') for g in amd_gpus]}")
        
        # Count Intel GPUs
        intel_info = self.gpu_info.get('intel', {})
        if intel_info.get('available', False):
            intel_gpus = intel_info.get('gpus', [])
            for gpu in intel_gpus:
                gpu_name = gpu.get('name', 'Unknown Intel')
                gpu_names.append(gpu_name)
                all_gpu_details.append(f"Intel: {gpu_name}")
            logger.info(f"Found {len(intel_gpus)} Intel GPUs: {[g.get('name') for g in intel_gpus]}")
        
        # Count Vulkan devices (only if no other GPUs found)
        vulkan_info = self.gpu_info.get('vulkan', {})
        if vulkan_info.get('available', False):
            vulkan_devices = vulkan_info.get('devices', [])
            # If no discrete GPUs found, use Vulkan devices
            if len(gpu_names) == 0:
                for device in vulkan_devices:
                    device_name = device.get('name', 'Vulkan Device')
                    gpu_names.append(device_name)
                    all_gpu_details.append(f"Vulkan: {device_name}")
            logger.info(f"Found {len(vulkan_devices)} Vulkan devices: {[d.get('name') for d in vulkan_devices]}")
        
        # Set total GPU count
        total_gpus = len(gpu_names)
        logger.info(f"Total GPU count: {total_gpus}")
        logger.info(f"All GPU details: {all_gpu_details}")
        
        # Determine if GPU mode is active - more comprehensive check
        gpu_mode_active = (
            backend_info.get('gpu_mode', False) or 
            gpu_summary != 'cpu' or 
            total_gpus > 0 or
            nvidia_info.get('available', False) or
            amd_info.get('available', False) or
            intel_info.get('available', False) or
            vulkan_info.get('available', False)
        )
        
        logger.info(f"GPU Mode Active: {gpu_mode_active}, Total GPUs: {total_gpus}, Best Backend: {gpu_summary}")
        
        info_text = f"""GPU Backend: {gpu_summary.upper()}
AI Processor: {backend_info.get('backend', 'unknown')}
GPU Mode: {'Yes' if gpu_mode_active else 'No'}
Total GPUs: {total_gpus}"""
        
        # Add GPU details if available
        if gpu_names:
            # Show primary GPU
            primary_gpu = gpu_names[0][:35] + "..." if len(gpu_names[0]) > 35 else gpu_names[0]
            info_text += f"""
Primary GPU: {primary_gpu}"""
            
            # Show secondary GPU if available
            if len(gpu_names) > 1:
                secondary_gpu = gpu_names[1][:35] + "..." if len(gpu_names[1]) > 35 else gpu_names[1]
                info_text += f"""
Secondary GPU: {secondary_gpu}"""
            
            # Show additional GPUs count if more than 2
            if len(gpu_names) > 2:
                additional_count = len(gpu_names) - 2
                info_text += f"""
Additional GPUs: +{additional_count} more"""
        
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            info_label = ctk.CTkLabel(system_frame, text=info_text, justify="left", anchor="w")
            info_label.pack(anchor="w", padx=15, pady=(0, 15))
        else:
            # Standard tkinter fallback
            info_label = ttk.Label(system_frame, text=info_text, justify="left")
            info_label.pack(anchor="w")
        
    def _setup_processing_section(self, parent):
        """Setup processing section"""
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            processing_frame = ctk.CTkFrame(parent)
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
        else:
            # Standard tkinter fallback
            process_frame = ttk.LabelFrame(parent, text="Processing", padding=15)
            process_frame.pack(fill="x", padx=20, pady=20)
            
            self.process_button = ttk.Button(
                process_frame,
                text="Start Processing",
                command=self._start_processing,
                state="disabled"
            )
            self.process_button.pack(pady=10)
        
        # Bind input change event
        self.input_path_var.trace("w", self._on_input_change)
        
    def _setup_status_section(self, parent):
        """Setup status section"""
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            status_frame = ctk.CTkFrame(parent)
            status_frame.pack(fill="x", padx=10, pady=10)
            
            status_label = ctk.CTkLabel(status_frame, text="📊 System Status", font=ctk.CTkFont(size=16, weight="bold"))
            status_label.pack(anchor="w", padx=15, pady=(15, 10))
            
            # Status display
            self.status_text = ctk.CTkTextbox(status_frame, height=80)
            self.status_text.pack(fill="x", padx=15, pady=(0, 15))
        else:
            # Standard tkinter fallback
            status_frame = ttk.LabelFrame(parent, text="Status", padding=15)
            status_frame.pack(fill="x", padx=20, pady=10)
            
            # Status display
            self.status_text = scrolledtext.ScrolledText(
                status_frame,
                height=6,
                width=70,
                font=("Consolas", 9)
            )
            self.status_text.pack(fill="both", expand=True, pady=(10, 0))
        
        self._add_status_message("Ready - Select input video to begin")
        
    def _add_status_message(self, message: str):
        """Add message to status display"""
        import time
        timestamp = time.strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}\\n"
        
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            self.status_text.insert("end", formatted)
            self.status_text.see("end")
        else:
            # Standard tkinter fallback
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
            # Log for debugging
            logger.info(f"Setting input file path: {file_path}")
            
            # Set the variable
            self.input_path_var.set(file_path)
            
            # Force multiple update methods
            self.input_entry.update()
            self.input_entry.update_idletasks()
            self.root.update()
            self.root.update_idletasks()
            
            # Direct manipulation of entry widget
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, file_path)
            # Note: self.input_entry.see(0) not available in tkinter Entry
            
            # Force focus and refresh
            self.input_entry.focus_set()
            self.input_entry.focus_force()
            self.root.after(100, lambda: self.input_entry.focus_set())
            
            # Add status message
            self._add_status_message(f"Selected: {Path(file_path).name}")
            logger.info(f"Input entry content: '{self.input_entry.get()}'")
            
    def _browse_output(self):
        """Browse for output folder"""
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        
        if folder_path:
            # Log for debugging
            logger.info(f"Setting output folder path: {folder_path}")
            
            # Set the variable
            self.output_path_var.set(folder_path)
            
            # Force multiple update methods
            self.output_entry.update()
            self.output_entry.update_idletasks()
            self.root.update()
            self.root.update_idletasks()
            
            # Direct manipulation of entry widget
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, folder_path)
            # Note: self.output_entry.see(0) not available in tkinter Entry
            
            # Force focus and refresh
            self.output_entry.focus_set()
            self.output_entry.focus_force()
            self.root.after(100, lambda: self.output_entry.focus_set())
            
            # Add status message
            self._add_status_message(f"Output folder: {Path(folder_path).name}")
            logger.info(f"Output entry content: '{self.output_entry.get()}'")
            
    def _on_input_change(self, *args):
        """Handle input file change"""
        input_path = self.input_path_var.get().strip()
        
        if input_path and Path(input_path).exists():
            if GUI_AVAILABLE and ctk:
                self.process_button.configure(state="normal")
            else:
                self.process_button.config(state="normal")
            self._add_status_message(f"Video selected: {Path(input_path).name}")
        else:
            if GUI_AVAILABLE and ctk:
                self.process_button.configure(state="disabled")
            else:
                self.process_button.config(state="disabled")
            
    def _start_processing(self):
        """Start video processing"""
        if self.is_processing:
            return
            
        input_path = self.input_path_var.get().strip()
        output_folder = self.output_path_var.get().strip()
        
        if not input_path:
            if GUI_AVAILABLE and ctk:
                CTkMessagebox(
                    title="Error",
                    message="Please select an input video file",
                    icon="cancel"
                )
            else:
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
        
        # Start processing thread first
        self.is_processing = True
        if GUI_AVAILABLE and ctk:
            self.process_button.configure(text="Processing...", state="disabled")
        else:
            self.process_button.config(text="Processing...", state="disabled")
        
        def process_video():
            """Background video processing"""
            timer = SimpleTimer()
            progress_dialog = None
            
            try:
                # Show progress dialog immediately on GUI thread
                def create_progress_dialog():
                    nonlocal progress_dialog
                    progress_dialog = ProgressDialog(self.root)
                    progress_dialog.add_log_message("Initializing video processing...")
                    progress_dialog.update_progress(0, "Starting processing...")
                
                self.root.after(0, create_progress_dialog)
                
                # Wait for progress dialog to be created
                import time
                while progress_dialog is None:
                    time.sleep(0.01)
                
                def progress_callback(progress, message):
                    if progress_dialog and not progress_dialog.cancelled:
                        self.root.after(0, lambda p=progress, m=message: progress_dialog.update_progress(p, m))
                        self.root.after(0, lambda m=message: progress_dialog.add_log_message(m))
                    
                    # Check for cancellation
                    if progress_dialog and progress_dialog.cancelled:
                        raise KeyboardInterrupt("Processing cancelled by user")
                
                # Step 1: Validate video
                self._safe_gui_update(lambda: progress_dialog.update_step_progress("validate", 0, "start"))
                progress_callback(5, "動画ファイル検証中...")
                validation = self.video_processor.validate_video(input_path)
                
                if not validation['valid']:
                    self._safe_gui_update(lambda: progress_dialog.update_step_progress("validate", 0, "error"))
                    raise RuntimeError(f"Invalid video: {validation['error']}")
                
                video_info = validation['info']
                self._safe_gui_update(lambda: progress_dialog.update_step_progress("validate", 100, "complete"))
                self._safe_gui_update(lambda: progress_dialog.add_log_message(f"動画情報: {video_info['width']}x{video_info['height']}, {video_info['duration']:.1f}秒"))
                
                # Step 2: Extract frames
                self._safe_gui_update(lambda: progress_dialog.update_step_progress("extract", 0, "start"))
                progress_callback(10, "フレーム抽出中...")
                
                def extract_progress_callback(progress, message):
                    # Map extraction progress to step progress (0-100)
                    step_progress = progress
                    self._safe_gui_update(lambda: progress_dialog.update_step_progress("extract", step_progress))
                    progress_callback(5 + (progress * 0.1), message)  # 5-15% overall
                
                frame_paths = self.video_processor.extract_frames(input_path, extract_progress_callback, progress_dialog)
                
                if not frame_paths:
                    self._safe_gui_update(lambda: progress_dialog.update_step_progress("extract", 0, "error"))
                    raise RuntimeError("Failed to extract frames")
                
                self._safe_gui_update(lambda: progress_dialog.update_step_progress("extract", 100, "complete"))
                self._safe_gui_update(lambda: progress_dialog.add_log_message(f"フレーム抽出完了: {len(frame_paths)}枚"))
                
                # Step 3: Upscale frames
                self._safe_gui_update(lambda: progress_dialog.update_step_progress("upscale", 0, "start"))
                progress_callback(20, "AIアップスケーリング実行中...")
                
                def upscale_progress_callback(progress, message):
                    # Map upscaling progress to step progress (0-100)
                    step_progress = progress
                    self._safe_gui_update(lambda: progress_dialog.update_step_progress("upscale", step_progress))
                    progress_callback(15 + (progress * 0.7), message)  # 15-85% overall
                
                upscaled_frames = self.ai_processor.upscale_frames(
                    frame_paths, 
                    str(self.video_processor.temp_dir / "upscaled"),
                    scale_factor,
                    upscale_progress_callback
                )
                
                if not upscaled_frames:
                    self._safe_gui_update(lambda: progress_dialog.update_step_progress("upscale", 0, "error"))
                    raise RuntimeError("Failed to upscale frames")
                
                self._safe_gui_update(lambda: progress_dialog.update_step_progress("upscale", 100, "complete"))
                self._safe_gui_update(lambda: progress_dialog.add_log_message(f"AIアップスケーリング完了: {len(upscaled_frames)}枚"))
                
                # Step 4: Combine to video
                self._safe_gui_update(lambda: progress_dialog.update_step_progress("combine", 0, "start"))
                progress_callback(90, "動画結合中...")
                
                def combine_progress_callback(progress, message):
                    # Map combine progress to step progress (0-100)
                    step_progress = progress
                    self._safe_gui_update(lambda: progress_dialog.update_step_progress("combine", step_progress))
                    progress_callback(85 + (progress * 0.15), message)  # 85-100% overall
                
                success = self.video_processor.combine_frames_to_video(
                    upscaled_frames,
                    output_path,
                    input_path,
                    video_info.get('frame_rate', 30.0),
                    combine_progress_callback
                )
                
                if not success:
                    self._safe_gui_update(lambda: progress_dialog.update_step_progress("combine", 0, "error"))
                    raise RuntimeError("Failed to combine frames to video")
                
                # Completed
                self._safe_gui_update(lambda: progress_dialog.update_step_progress("combine", 100, "complete"))
                progress_callback(100, "処理完了!")
                processing_time = timer.elapsed_str()
                
                self._safe_gui_update(lambda: progress_dialog.add_log_message(f"処理完了: {processing_time}"))
                self._safe_gui_update(lambda: progress_dialog.add_log_message(f"出力先: {output_path}"))
                
                # Show completion dialog after short delay
                self._safe_gui_update(lambda: self.root.after(2000, lambda: self._on_processing_complete(output_path, progress_dialog)))
                
            except KeyboardInterrupt:
                # Handle user cancellation
                error_msg = "Processing cancelled by user"
                if progress_dialog:
                    self._safe_gui_update(lambda: progress_dialog.add_log_message(error_msg))
                self._safe_gui_update(lambda: self.root.after(1000, lambda: self._on_processing_error(error_msg, progress_dialog)))
                
            except Exception as e:
                error_msg = str(e)
                if progress_dialog:
                    self._safe_gui_update(lambda: progress_dialog.add_log_message(f"Error: {error_msg}"))
                self._safe_gui_update(lambda: self.root.after(1000, lambda: self._on_processing_error(error_msg, progress_dialog)))
        
        self.processing_thread = threading.Thread(target=process_video, daemon=True)
        self.processing_thread.start()
    
    def _safe_gui_update(self, callback):
        """Safely update GUI from background thread"""
        try:
            if hasattr(self, 'root') and self.root and hasattr(self.root, 'winfo_exists'):
                if self.root.winfo_exists():
                    self.root.after_idle(callback)
        except (AttributeError, RuntimeError, tk.TclError):
            pass  # Ignore errors if window is destroyed or in invalid state
        
    def _on_processing_complete(self, output_path, progress_dialog):
        """Handle processing completion"""
        progress_dialog.destroy()
        
        self.is_processing = False
        if GUI_AVAILABLE and ctk:
            self.process_button.configure(text="Start Processing", state="normal")
            CTkMessagebox(
                title="Success!",
                message=f"Video processing completed successfully!\n\nOutput saved to:\n{output_path}",
                icon="check"
            )
        else:
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
        if GUI_AVAILABLE and ctk:
            self.process_button.configure(text="Start Processing", state="normal")
            CTkMessagebox(
                title="Processing Error",
                message=f"Processing failed:\n\n{error_msg}",
                icon="cancel"
            )
        else:
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