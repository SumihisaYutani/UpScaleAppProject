"""
Batch Processing GUI Window
Interface for managing batch video processing jobs
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

from modules.batch_processor import BatchProcessor, BatchJob, JobStatus
from config.settings import PATHS


class JobListFrame(ctk.CTkScrollableFrame):
    """Scrollable frame for displaying job list"""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.job_widgets: Dict[str, ctk.CTkFrame] = {}
        self.selected_jobs: List[str] = []
        self.job_update_callback = None
    
    def add_job_widget(self, job: BatchJob):
        """Add a job widget to the list"""
        
        # Create job frame
        job_frame = ctk.CTkFrame(self)
        job_frame.pack(fill="x", padx=5, pady=2)
        
        # Configure grid
        job_frame.grid_columnconfigure(1, weight=1)
        
        # Checkbox for selection
        checkbox_var = tk.BooleanVar()
        checkbox = ctk.CTkCheckBox(
            job_frame,
            text="",
            variable=checkbox_var,
            width=20,
            command=lambda: self._on_job_selection_changed(job.id, checkbox_var.get())
        )
        checkbox.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        # Job info frame
        info_frame = ctk.CTkFrame(job_frame)
        info_frame.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        info_frame.grid_columnconfigure(0, weight=1)
        
        # File name
        filename = Path(job.input_path).name
        filename_label = ctk.CTkLabel(
            info_frame,
            text=filename,
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        filename_label.grid(row=0, column=0, padx=5, pady=(5, 0), sticky="ew")
        
        # Status and progress
        status_frame = ctk.CTkFrame(info_frame)
        status_frame.grid(row=1, column=0, padx=5, pady=2, sticky="ew")
        status_frame.grid_columnconfigure(1, weight=1)
        
        # Status badge
        status_color = self._get_status_color(job.status)
        status_label = ctk.CTkLabel(
            status_frame,
            text=job.status.value.title(),
            fg_color=status_color,
            corner_radius=10,
            width=80
        )
        status_label.grid(row=0, column=0, padx=5, pady=2)
        
        # Progress bar (only show for processing/queued jobs)
        if job.status in [JobStatus.PROCESSING, JobStatus.QUEUED]:
            progress_bar = ctk.CTkProgressBar(status_frame)
            progress_bar.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
            progress_bar.set(job.progress / 100)
            
            progress_label = ctk.CTkLabel(
                status_frame,
                text=f"{job.progress:.1f}%"
            )
            progress_label.grid(row=0, column=2, padx=5, pady=2)
        elif job.status == JobStatus.FAILED:
            error_label = ctk.CTkLabel(
                status_frame,
                text=f"Error: {job.error_message[:50]}...",
                text_color="red"
            )
            error_label.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        
        # Control buttons
        controls_frame = ctk.CTkFrame(job_frame)
        controls_frame.grid(row=0, column=2, padx=5, pady=5)
        
        if job.status == JobStatus.FAILED:
            retry_btn = ctk.CTkButton(
                controls_frame,
                text="Retry",
                width=60,
                height=25,
                command=lambda: self._retry_job(job.id)
            )
            retry_btn.pack(side="left", padx=2)
        
        if job.status in [JobStatus.PENDING, JobStatus.QUEUED]:
            pause_btn = ctk.CTkButton(
                controls_frame,
                text="Pause",
                width=60,
                height=25,
                command=lambda: self._pause_job(job.id)
            )
            pause_btn.pack(side="left", padx=2)
        elif job.status == JobStatus.PAUSED:
            resume_btn = ctk.CTkButton(
                controls_frame,
                text="Resume",
                width=60,
                height=25,
                command=lambda: self._resume_job(job.id)
            )
            resume_btn.pack(side="left", padx=2)
        
        if job.status not in [JobStatus.PROCESSING]:
            remove_btn = ctk.CTkButton(
                controls_frame,
                text="Remove",
                width=60,
                height=25,
                fg_color="transparent",
                border_width=1,
                text_color=("gray10", "gray90"),
                command=lambda: self._remove_job(job.id)
            )
            remove_btn.pack(side="left", padx=2)
        
        # Store references
        self.job_widgets[job.id] = {
            "frame": job_frame,
            "checkbox": checkbox,
            "checkbox_var": checkbox_var,
            "status_label": status_label,
            "filename_label": filename_label
        }
    
    def update_job_widget(self, job: BatchJob):
        """Update existing job widget"""
        if job.id not in self.job_widgets:
            return
        
        widgets = self.job_widgets[job.id]
        
        # Update status color
        status_color = self._get_status_color(job.status)
        widgets["status_label"].configure(
            text=job.status.value.title(),
            fg_color=status_color
        )
        
        # Recreate the widget if needed for complex updates
        self.remove_job_widget(job.id)
        self.add_job_widget(job)
    
    def remove_job_widget(self, job_id: str):
        """Remove job widget from list"""
        if job_id in self.job_widgets:
            self.job_widgets[job_id]["frame"].destroy()
            del self.job_widgets[job_id]
            
            if job_id in self.selected_jobs:
                self.selected_jobs.remove(job_id)
    
    def clear_all_widgets(self):
        """Clear all job widgets"""
        for job_id in list(self.job_widgets.keys()):
            self.remove_job_widget(job_id)
    
    def _get_status_color(self, status: JobStatus) -> str:
        """Get color for job status"""
        color_map = {
            JobStatus.PENDING: "gray",
            JobStatus.QUEUED: "blue",
            JobStatus.PROCESSING: "orange",
            JobStatus.COMPLETED: "green",
            JobStatus.FAILED: "red",
            JobStatus.CANCELLED: "gray",
            JobStatus.PAUSED: "yellow"
        }
        return color_map.get(status, "gray")
    
    def _on_job_selection_changed(self, job_id: str, selected: bool):
        """Handle job selection change"""
        if selected and job_id not in self.selected_jobs:
            self.selected_jobs.append(job_id)
        elif not selected and job_id in self.selected_jobs:
            self.selected_jobs.remove(job_id)
    
    def _retry_job(self, job_id: str):
        """Retry a failed job"""
        if self.job_update_callback:
            self.job_update_callback("retry", job_id)
    
    def _pause_job(self, job_id: str):
        """Pause a job"""
        if self.job_update_callback:
            self.job_update_callback("pause", job_id)
    
    def _resume_job(self, job_id: str):
        """Resume a paused job"""
        if self.job_update_callback:
            self.job_update_callback("resume", job_id)
    
    def _remove_job(self, job_id: str):
        """Remove a job"""
        if self.job_update_callback:
            self.job_update_callback("remove", job_id)


class BatchWindow:
    """Batch processing window"""
    
    def __init__(self, parent=None):
        if not GUI_AVAILABLE:
            raise ImportError("GUI dependencies not available")
        
        self.parent = parent
        self.batch_processor = BatchProcessor(max_parallel_jobs=2)
        
        # Setup callbacks
        self.batch_processor.job_update_callback = self._on_job_update
        self.batch_processor.queue_update_callback = self._on_queue_update
        
        self._setup_window()
        self._setup_ui()
        self._refresh_job_list()
        
        # Start periodic updates
        self._start_periodic_updates()
    
    def _setup_window(self):
        """Setup batch window"""
        self.window = ctk.CTkToplevel(self.parent)
        self.window.title("Batch Processing - UpScale App")
        self.window.geometry("900x700")
        
        # Make it modal if parent exists
        if self.parent:
            self.window.transient(self.parent)
            self.window.grab_set()
        
        # Configure grid
        self.window.grid_rowconfigure(1, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        
        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self._on_window_close)
    
    def _setup_ui(self):
        """Setup user interface"""
        
        # Top controls
        self._setup_controls()
        
        # Job list
        self._setup_job_list()
        
        # Bottom status
        self._setup_status_bar()
    
    def _setup_controls(self):
        """Setup control buttons and settings"""
        
        controls_frame = ctk.CTkFrame(self.window)
        controls_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        controls_frame.grid_columnconfigure(2, weight=1)
        
        # Add files button
        add_files_btn = ctk.CTkButton(
            controls_frame,
            text="📁 Add Files",
            command=self._add_files,
            height=35
        )
        add_files_btn.grid(row=0, column=0, padx=5, pady=5)
        
        # Add folder button
        add_folder_btn = ctk.CTkButton(
            controls_frame,
            text="📂 Add Folder",
            command=self._add_folder,
            height=35
        )
        add_folder_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Settings frame
        settings_frame = ctk.CTkFrame(controls_frame)
        settings_frame.grid(row=0, column=2, padx=10, pady=5, sticky="ew")
        
        # Parallel jobs setting
        ctk.CTkLabel(settings_frame, text="Parallel Jobs:").grid(row=0, column=0, padx=5, pady=5)
        self.parallel_var = ctk.StringVar(value=str(self.batch_processor.max_parallel_jobs))
        parallel_menu = ctk.CTkComboBox(
            settings_frame,
            values=["1", "2", "3", "4"],
            variable=self.parallel_var,
            state="readonly",
            width=60,
            command=self._on_parallel_jobs_changed
        )
        parallel_menu.grid(row=0, column=1, padx=5, pady=5)
        
        # Quality setting
        ctk.CTkLabel(settings_frame, text="Quality:").grid(row=0, column=2, padx=5, pady=5)
        self.quality_var = ctk.StringVar(value="Balanced")
        quality_menu = ctk.CTkComboBox(
            settings_frame,
            values=["Fast", "Balanced", "Quality"],
            variable=self.quality_var,
            state="readonly",
            width=100
        )
        quality_menu.grid(row=0, column=3, padx=5, pady=5)
        
        # Queue controls
        queue_controls = ctk.CTkFrame(controls_frame)
        queue_controls.grid(row=0, column=3, padx=5, pady=5)
        
        self.pause_btn = ctk.CTkButton(
            queue_controls,
            text="⏸️ Pause Queue",
            command=self._toggle_queue_pause,
            width=100,
            height=35
        )
        self.pause_btn.pack(side="left", padx=2)
        
        clear_btn = ctk.CTkButton(
            queue_controls,
            text="🗑️ Clear Done",
            command=self._clear_completed,
            width=100,
            height=35,
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "gray90")
        )
        clear_btn.pack(side="right", padx=2)
    
    def _setup_job_list(self):
        """Setup job list display"""
        
        list_frame = ctk.CTkFrame(self.window)
        list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        
        # Job list
        self.job_list = JobListFrame(list_frame)
        self.job_list.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.job_list.job_update_callback = self._on_job_action
        
    def _setup_status_bar(self):
        """Setup status bar"""
        
        status_frame = ctk.CTkFrame(self.window)
        status_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Ready",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(side="left", padx=10, pady=5)
        
        self.stats_label = ctk.CTkLabel(
            status_frame,
            text="Jobs: 0 | Completed: 0 | Failed: 0",
            font=ctk.CTkFont(size=12)
        )
        self.stats_label.pack(side="right", padx=10, pady=5)
    
    def _add_files(self):
        """Add individual files to batch"""
        from tkinter import filedialog
        
        file_paths = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("All files", "*.*")
            ]
        )
        
        if file_paths:
            settings = {
                "quality_preset": self.quality_var.get().lower(),
                "scale_factor": 1.5,
                "use_ai": True,
                "use_enhanced_ai": True
            }
            
            job_ids = self.batch_processor.add_multiple_jobs(
                list(file_paths), 
                settings=settings
            )
            
            CTkMessagebox(
                title="Files Added",
                message=f"Added {len(job_ids)} files to batch queue.",
                icon="check"
            )
            
            self._refresh_job_list()
    
    def _add_folder(self):
        """Add all video files from a folder"""
        from tkinter import filedialog
        
        folder_path = filedialog.askdirectory(title="Select Folder with Videos")
        
        if folder_path:
            # Find video files
            video_files = []
            for ext in [".mp4", ".MP4"]:
                video_files.extend(Path(folder_path).glob(f"*{ext}"))
            
            if not video_files:
                CTkMessagebox(
                    title="No Videos Found",
                    message="No video files found in the selected folder.",
                    icon="warning"
                )
                return
            
            settings = {
                "quality_preset": self.quality_var.get().lower(),
                "scale_factor": 1.5,
                "use_ai": True,
                "use_enhanced_ai": True
            }
            
            job_ids = self.batch_processor.add_multiple_jobs(
                [str(f) for f in video_files],
                settings=settings
            )
            
            CTkMessagebox(
                title="Folder Added",
                message=f"Added {len(job_ids)} video files from folder to batch queue.",
                icon="check"
            )
            
            self._refresh_job_list()
    
    def _on_parallel_jobs_changed(self, value):
        """Handle parallel jobs setting change"""
        try:
            max_jobs = int(value)
            self.batch_processor.set_max_parallel_jobs(max_jobs)
        except ValueError:
            pass
    
    def _toggle_queue_pause(self):
        """Toggle queue pause/resume"""
        status = self.batch_processor.get_queue_status()
        
        if status["is_paused"]:
            self.batch_processor.resume_queue()
            self.pause_btn.configure(text="⏸️ Pause Queue")
        else:
            self.batch_processor.pause_queue()
            self.pause_btn.configure(text="▶️ Resume Queue")
    
    def _clear_completed(self):
        """Clear completed jobs"""
        self.batch_processor.clear_completed_jobs()
        self._refresh_job_list()
    
    def _on_job_action(self, action: str, job_id: str):
        """Handle job actions from list"""
        if action == "retry":
            self.batch_processor.retry_job(job_id)
        elif action == "pause":
            self.batch_processor.pause_job(job_id)
        elif action == "resume":
            self.batch_processor.resume_job(job_id)
        elif action == "remove":
            self.batch_processor.remove_job(job_id)
        
        self._refresh_job_list()
    
    def _on_job_update(self, job: BatchJob):
        """Handle job update from batch processor"""
        # Update specific job widget
        self.job_list.update_job_widget(job)
    
    def _on_queue_update(self, status: Dict):
        """Handle queue update from batch processor"""
        # Update status bar
        self.window.after(0, lambda: self._update_status_bar(status))
    
    def _update_status_bar(self, status: Dict):
        """Update status bar display"""
        stats = status["stats"]
        status_counts = status["status_counts"]
        
        # Status message
        active = status["active_jobs"]
        if active > 0:
            self.status_label.configure(text=f"Processing {active} jobs...")
        elif status["is_paused"]:
            self.status_label.configure(text="Queue paused")
        else:
            self.status_label.configure(text="Ready")
        
        # Stats
        stats_text = (f"Jobs: {status['total_jobs']} | "
                     f"Completed: {stats['completed_jobs']} | "
                     f"Failed: {stats['failed_jobs']}")
        self.stats_label.configure(text=stats_text)
    
    def _refresh_job_list(self):
        """Refresh the job list display"""
        # Clear current widgets
        self.job_list.clear_all_widgets()
        
        # Add all jobs
        jobs = self.batch_processor.get_all_jobs()
        for job in jobs:
            self.job_list.add_job_widget(job)
    
    def _start_periodic_updates(self):
        """Start periodic UI updates"""
        def update_loop():
            if self.window.winfo_exists():
                # Update status
                status = self.batch_processor.get_queue_status()
                self._update_status_bar(status)
                
                # Schedule next update
                self.window.after(2000, update_loop)  # Update every 2 seconds
        
        self.window.after(1000, update_loop)  # Start after 1 second
    
    def _on_window_close(self):
        """Handle window close"""
        try:
            self.batch_processor.shutdown()
        except:
            pass
        
        if self.parent:
            self.parent.grab_set()  # Return focus to parent
        
        self.window.destroy()
    
    def show(self):
        """Show the batch window"""
        self.window.deiconify()
        self.window.lift()
        self.window.focus()