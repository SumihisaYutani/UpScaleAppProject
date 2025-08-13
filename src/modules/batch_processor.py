"""
Batch Processing Module
Handles multiple video file processing with queue management
"""

import os
import json
import time
import threading
import queue
from pathlib import Path
from typing import List, Dict, Callable, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from enhanced_upscale_app import EnhancedUpScaleApp
from config.settings import PATHS

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job status enumeration"""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class BatchJob:
    """Represents a single batch processing job"""
    id: str
    input_path: str
    output_path: str
    settings: Dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    error_message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Dict] = None
    priority: int = 0  # Higher number = higher priority
    
    @property
    def processing_time(self) -> float:
        """Get processing time in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0.0
    
    @property
    def output_size(self) -> Optional[int]:
        """Get output file size if available"""
        if self.status == JobStatus.COMPLETED and os.path.exists(self.output_path):
            return os.path.getsize(self.output_path)
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "settings": self.settings,
            "status": self.status.value,
            "progress": self.progress,
            "error_message": self.error_message,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "result": self.result,
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BatchJob':
        """Create from dictionary"""
        job = cls(
            id=data["id"],
            input_path=data["input_path"],
            output_path=data["output_path"],
            settings=data.get("settings", {}),
            priority=data.get("priority", 0)
        )
        job.status = JobStatus(data.get("status", "pending"))
        job.progress = data.get("progress", 0.0)
        job.error_message = data.get("error_message", "")
        job.start_time = data.get("start_time")
        job.end_time = data.get("end_time")
        job.result = data.get("result")
        return job


class BatchProcessor:
    """Batch video processing manager"""
    
    def __init__(self, max_parallel_jobs: int = 2):
        self.max_parallel_jobs = max_parallel_jobs
        self.jobs: Dict[str, BatchJob] = {}
        self.job_queue = queue.PriorityQueue()
        self.processing_jobs: Dict[str, threading.Thread] = {}
        
        # Threading control
        self.executor = ThreadPoolExecutor(max_workers=max_parallel_jobs)
        self.shutdown_event = threading.Event()
        self.pause_event = threading.Event()
        
        # Callbacks
        self.job_update_callback: Optional[Callable] = None
        self.queue_update_callback: Optional[Callable] = None
        
        # Statistics
        self.stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "total_processing_time": 0.0,
            "total_input_size": 0,
            "total_output_size": 0
        }
        
        # Auto-save queue
        self.queue_file = PATHS["temp_dir"] / "batch_queue.json"
        self.load_queue()
        
        # Start queue processor
        self._start_queue_processor()
    
    def add_job(self, input_path: str, output_path: str = None, 
                settings: Dict = None, priority: int = 0) -> str:
        """Add a job to the batch queue"""
        
        # Generate job ID
        job_id = f"job_{int(time.time() * 1000)}_{len(self.jobs)}"
        
        # Generate output path if not provided
        if output_path is None:
            input_file = Path(input_path)
            output_dir = PATHS["output_dir"] / "batch"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"{input_file.stem}_upscaled.mp4")
        
        # Default settings
        if settings is None:
            settings = {
                "scale_factor": 1.5,
                "quality_preset": "balanced",
                "use_ai": True,
                "use_enhanced_ai": True
            }
        
        # Create job
        job = BatchJob(
            id=job_id,
            input_path=input_path,
            output_path=output_path,
            settings=settings,
            priority=priority
        )
        
        # Add to jobs dict and queue
        self.jobs[job_id] = job
        self.job_queue.put((-priority, time.time(), job_id))  # Negative for max-heap behavior
        
        self.stats["total_jobs"] += 1
        
        # Save queue and notify
        self.save_queue()
        self._notify_queue_update()
        
        logger.info(f"Added batch job {job_id}: {input_path}")
        return job_id
    
    def add_multiple_jobs(self, file_paths: List[str], output_dir: str = None,
                         settings: Dict = None, priority: int = 0) -> List[str]:
        """Add multiple jobs at once"""
        
        if output_dir is None:
            output_dir = str(PATHS["output_dir"] / "batch")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        job_ids = []
        for file_path in file_paths:
            input_file = Path(file_path)
            output_path = str(Path(output_dir) / f"{input_file.stem}_upscaled.mp4")
            job_id = self.add_job(file_path, output_path, settings, priority)
            job_ids.append(job_id)
        
        return job_ids
    
    def remove_job(self, job_id: str) -> bool:
        """Remove a job from the queue"""
        
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        # Can't remove jobs that are currently processing
        if job.status == JobStatus.PROCESSING:
            logger.warning(f"Cannot remove job {job_id}: currently processing")
            return False
        
        # Remove from jobs dict
        del self.jobs[job_id]
        
        # Remove from processing if applicable
        if job_id in self.processing_jobs:
            del self.processing_jobs[job_id]
        
        self.save_queue()
        self._notify_queue_update()
        
        logger.info(f"Removed batch job {job_id}")
        return True
    
    def pause_job(self, job_id: str) -> bool:
        """Pause a specific job"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        if job.status in [JobStatus.QUEUED, JobStatus.PROCESSING]:
            job.status = JobStatus.PAUSED
            self._notify_job_update(job)
            return True
        
        return False
    
    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        if job.status == JobStatus.PAUSED:
            job.status = JobStatus.QUEUED
            # Re-add to queue
            self.job_queue.put((-job.priority, time.time(), job_id))
            self._notify_job_update(job)
            return True
        
        return False
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        job.status = JobStatus.CANCELLED
        job.error_message = "Cancelled by user"
        
        self._notify_job_update(job)
        self.save_queue()
        
        return True
    
    def retry_job(self, job_id: str) -> bool:
        """Retry a failed job"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        if job.status == JobStatus.FAILED:
            job.status = JobStatus.QUEUED
            job.progress = 0.0
            job.error_message = ""
            job.start_time = None
            job.end_time = None
            job.result = None
            
            # Re-add to queue
            self.job_queue.put((-job.priority, time.time(), job_id))
            self._notify_job_update(job)
            
            return True
        
        return False
    
    def clear_completed_jobs(self):
        """Clear all completed and failed jobs"""
        to_remove = []
        for job_id, job in self.jobs.items():
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.jobs[job_id]
        
        self.save_queue()
        self._notify_queue_update()
        
        logger.info(f"Cleared {len(to_remove)} completed jobs")
    
    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get a specific job"""
        return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> List[BatchJob]:
        """Get all jobs sorted by priority and time"""
        return sorted(self.jobs.values(), 
                     key=lambda x: (-x.priority, x.start_time or time.time()))
    
    def get_jobs_by_status(self, status: JobStatus) -> List[BatchJob]:
        """Get jobs by status"""
        return [job for job in self.jobs.values() if job.status == status]
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status"""
        status_counts = {}
        for status in JobStatus:
            status_counts[status.value] = len(self.get_jobs_by_status(status))
        
        return {
            "total_jobs": len(self.jobs),
            "status_counts": status_counts,
            "active_jobs": len(self.processing_jobs),
            "max_parallel": self.max_parallel_jobs,
            "queue_size": self.job_queue.qsize(),
            "is_paused": self.pause_event.is_set(),
            "stats": self.stats.copy()
        }
    
    def pause_queue(self):
        """Pause the entire queue"""
        self.pause_event.set()
        logger.info("Batch queue paused")
    
    def resume_queue(self):
        """Resume the entire queue"""
        self.pause_event.clear()
        logger.info("Batch queue resumed")
    
    def set_max_parallel_jobs(self, max_jobs: int):
        """Set maximum parallel jobs"""
        if max_jobs < 1:
            max_jobs = 1
        elif max_jobs > 8:  # Reasonable limit
            max_jobs = 8
        
        self.max_parallel_jobs = max_jobs
        
        # Recreate executor if needed
        if max_jobs != self.executor._max_workers:
            self.executor.shutdown(wait=False)
            self.executor = ThreadPoolExecutor(max_workers=max_jobs)
        
        logger.info(f"Set max parallel jobs to {max_jobs}")
    
    def _start_queue_processor(self):
        """Start the background queue processor"""
        def queue_processor():
            while not self.shutdown_event.is_set():
                try:
                    # Wait if paused
                    if self.pause_event.is_set():
                        time.sleep(1)
                        continue
                    
                    # Check if we can start more jobs
                    if len(self.processing_jobs) >= self.max_parallel_jobs:
                        time.sleep(1)
                        continue
                    
                    # Get next job from queue
                    try:
                        priority, timestamp, job_id = self.job_queue.get(timeout=1)
                    except queue.Empty:
                        continue
                    
                    # Check if job still exists and is valid
                    if job_id not in self.jobs:
                        continue
                    
                    job = self.jobs[job_id]
                    if job.status not in [JobStatus.QUEUED, JobStatus.PENDING]:
                        continue
                    
                    # Start processing job
                    self._start_job_processing(job)
                    
                except Exception as e:
                    logger.error(f"Queue processor error: {e}")
                    time.sleep(5)
        
        self.queue_thread = threading.Thread(target=queue_processor, daemon=True)
        self.queue_thread.start()
    
    def _start_job_processing(self, job: BatchJob):
        """Start processing a specific job"""
        
        def process_job():
            job.status = JobStatus.PROCESSING
            job.start_time = time.time()
            job.progress = 0.0
            self._notify_job_update(job)
            
            try:
                # Create app instance for this job
                app = EnhancedUpScaleApp(
                    use_ai=job.settings.get("use_ai", True),
                    use_enhanced_ai=job.settings.get("use_enhanced_ai", True),
                    enable_monitoring=True
                )
                
                def progress_callback(progress, message):
                    job.progress = progress
                    self._notify_job_update(job)
                
                # Process the video
                result = app.process_video_enhanced(
                    input_path=job.input_path,
                    output_path=job.output_path,
                    scale_factor=job.settings.get("scale_factor", 1.5),
                    progress_callback=progress_callback,
                    quality_settings=self._get_quality_settings(job.settings.get("quality_preset", "balanced"))
                )
                
                # Update job with result
                job.end_time = time.time()
                job.result = result
                
                if result["success"]:
                    job.status = JobStatus.COMPLETED
                    job.progress = 100.0
                    self.stats["completed_jobs"] += 1
                    
                    # Update size stats
                    if job.output_size:
                        self.stats["total_output_size"] += job.output_size
                else:
                    job.status = JobStatus.FAILED
                    job.error_message = result.get("error", "Unknown error")
                    self.stats["failed_jobs"] += 1
                
                # Update processing time stats
                self.stats["total_processing_time"] += job.processing_time
                
                # Cleanup app
                app.cleanup()
                
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                job.end_time = time.time()
                self.stats["failed_jobs"] += 1
                logger.error(f"Job {job.id} failed: {e}")
            
            finally:
                # Remove from processing jobs
                if job.id in self.processing_jobs:
                    del self.processing_jobs[job.id]
                
                self._notify_job_update(job)
                self.save_queue()
                
        # Start job in executor
        future = self.executor.submit(process_job)
        self.processing_jobs[job.id] = future
        
        logger.info(f"Started processing job {job.id}: {job.input_path}")
    
    def _get_quality_settings(self, preset: str) -> Dict:
        """Get quality settings for preset"""
        quality_presets = {
            "fast": {"strength": 0.2, "num_inference_steps": 10},
            "balanced": {"strength": 0.3, "num_inference_steps": 20},
            "quality": {"strength": 0.4, "num_inference_steps": 30}
        }
        return quality_presets.get(preset.lower(), quality_presets["balanced"])
    
    def _notify_job_update(self, job: BatchJob):
        """Notify about job update"""
        if self.job_update_callback:
            try:
                self.job_update_callback(job)
            except Exception as e:
                logger.warning(f"Job update callback error: {e}")
    
    def _notify_queue_update(self):
        """Notify about queue update"""
        if self.queue_update_callback:
            try:
                self.queue_update_callback(self.get_queue_status())
            except Exception as e:
                logger.warning(f"Queue update callback error: {e}")
    
    def save_queue(self):
        """Save queue to file"""
        try:
            queue_data = {
                "jobs": {job_id: job.to_dict() for job_id, job in self.jobs.items()},
                "stats": self.stats,
                "settings": {
                    "max_parallel_jobs": self.max_parallel_jobs
                }
            }
            
            self.queue_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.queue_file, "w") as f:
                json.dump(queue_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save queue: {e}")
    
    def load_queue(self):
        """Load queue from file"""
        try:
            if not self.queue_file.exists():
                return
            
            with open(self.queue_file, "r") as f:
                queue_data = json.load(f)
            
            # Load jobs
            for job_id, job_data in queue_data.get("jobs", {}).items():
                job = BatchJob.from_dict(job_data)
                self.jobs[job_id] = job
                
                # Re-queue pending/queued jobs
                if job.status in [JobStatus.PENDING, JobStatus.QUEUED]:
                    self.job_queue.put((-job.priority, time.time(), job_id))
            
            # Load stats
            self.stats.update(queue_data.get("stats", {}))
            
            # Load settings
            settings = queue_data.get("settings", {})
            if "max_parallel_jobs" in settings:
                self.max_parallel_jobs = settings["max_parallel_jobs"]
            
            logger.info(f"Loaded {len(self.jobs)} jobs from queue file")
            
        except Exception as e:
            logger.warning(f"Failed to load queue: {e}")
    
    def shutdown(self):
        """Shutdown the batch processor"""
        logger.info("Shutting down batch processor...")
        
        self.shutdown_event.set()
        self.pause_queue()
        
        # Wait for current jobs to complete
        self.executor.shutdown(wait=True)
        
        # Save final state
        self.save_queue()
        
        logger.info("Batch processor shutdown complete")