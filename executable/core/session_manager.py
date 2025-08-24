"""
UpScale App - Session Management Module
Handles progress tracking and resume functionality for video processing
"""

import os
import json
import hashlib
import tempfile
import logging
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages processing sessions for resume functionality"""
    
    def __init__(self, base_temp_dir: Optional[Path] = None):
        # Use system temp directory with app-specific subdirectory
        if base_temp_dir is None:
            base_temp_dir = Path(tempfile.gettempdir()) / "upscale_app_sessions"
        
        self.sessions_dir = Path(base_temp_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SessionManager initialized - Sessions dir: {self.sessions_dir}")
    
    def generate_session_id(self, video_path: str, settings: Dict[str, Any]) -> str:
        """Generate unique session ID based on video file and settings"""
        # Create hash from video path and settings for unique identification
        video_stat = Path(video_path).stat()
        
        session_data = {
            'video_path': str(Path(video_path).resolve()).replace('\\', '/'),
            'video_size': video_stat.st_size,
            'video_mtime': video_stat.st_mtime,
            'scale_factor': settings.get('scale_factor', 2.0),
            'quality': settings.get('quality', 'Quality'),
            'noise_reduction': settings.get('noise_reduction', 3)
        }
        
        # === DEBUG: セッションID生成の詳細ログ ===
        logger.info(f"=== SESSION ID DEBUG: Generating session ID ===")
        logger.info(f"Session data: {session_data}")
        
        session_string = json.dumps(session_data, sort_keys=True)
        logger.info(f"Session string: {session_string}")
        
        session_hash = hashlib.md5(session_string.encode()).hexdigest()[:12]
        logger.info(f"Generated session ID: {session_hash} for {Path(video_path).name}")
        
        return session_hash
    
    def get_session_dir(self, session_id: str) -> Path:
        """Get session directory path"""
        return self.sessions_dir / session_id
    
    def create_session(self, video_path: str, settings: Dict[str, Any], video_info: Dict[str, Any]) -> str:
        """Create new processing session"""
        session_id = self.generate_session_id(video_path, settings)
        session_dir = self.get_session_dir(session_id)
        
        # Create session directory structure
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "frames").mkdir(exist_ok=True)
        (session_dir / "upscaled").mkdir(exist_ok=True)
        (session_dir / "logs").mkdir(exist_ok=True)
        
        # Initialize progress data
        progress_data = {
            'session_id': session_id,
            'video_file': str(Path(video_path).resolve()).replace('\\', '/'),
            'video_info': video_info,
            'settings': settings,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'status': 'created',
            'steps': {
                'validate': {
                    'status': 'pending',
                    'progress': 0,
                    'start_time': None,
                    'end_time': None,
                    'error': None
                },
                'extract': {
                    'status': 'pending',
                    'progress': 0,
                    'total_frames': video_info.get('frame_count', 0),
                    'extracted_frames': 0,
                    'completed_batches': [],
                    'start_time': None,
                    'end_time': None,
                    'error': None
                },
                'upscale': {
                    'status': 'pending',
                    'progress': 0,
                    'total_frames': video_info.get('frame_count', 0),
                    'completed_frames': [],
                    'failed_frames': [],
                    'current_batch': 0,
                    'start_time': None,
                    'end_time': None,
                    'error': None
                },
                'combine': {
                    'status': 'pending',
                    'progress': 0,
                    'output_path': None,
                    'start_time': None,
                    'end_time': None,
                    'error': None
                }
            }
        }
        
        # Save initial progress
        self.save_progress(session_id, progress_data)
        
        logger.info(f"Created session {session_id} for {Path(video_path).name}")
        return session_id
    
    def save_progress(self, session_id: str, progress_data: Dict[str, Any]) -> None:
        """Save progress data to JSON file"""
        try:
            session_dir = self.get_session_dir(session_id)
            progress_file = session_dir / "progress.json"
            
            # Update timestamp
            progress_data['last_updated'] = datetime.now().isoformat()
            
            # === DEBUG: セッション保存の詳細ログ ===
            completed_frames = progress_data.get('steps', {}).get('upscale', {}).get('completed_frames', [])
            logger.info(f"=== SESSION DEBUG: Saving progress for session {session_id} ===")
            logger.info(f"Session directory: {session_dir}")
            logger.info(f"Progress file: {progress_file}")
            logger.info(f"Saving {len(completed_frames)} completed frames")
            
            # Ensure session directory exists
            if not session_dir.exists():
                logger.info(f"Creating session directory: {session_dir}")
                session_dir.mkdir(parents=True, exist_ok=True)
            
            # Verify directory is writable
            test_file = session_dir / "test_write.tmp"
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                test_file.unlink()
                logger.info("Directory is writable")
            except Exception as test_e:
                logger.error(f"Directory is not writable: {test_e}")
                raise
            
            # Save with pretty formatting for debugging
            logger.info(f"Writing to file: {progress_file}")
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
            
            # Verify file was created
            if progress_file.exists():
                file_size = progress_file.stat().st_size
                logger.info(f"Progress saved successfully: {file_size} bytes")
            else:
                logger.error("Progress file was not created despite no errors")
            
        except Exception as e:
            logger.error(f"Failed to save progress for session {session_id}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def load_progress(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load progress data from JSON file"""
        try:
            session_dir = self.get_session_dir(session_id)
            progress_file = session_dir / "progress.json"
            
            if not progress_file.exists():
                return None
            
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            # === DEBUG: セッション読み込みの詳細ログ ===
            completed_frames = progress_data.get('steps', {}).get('upscale', {}).get('completed_frames', [])
            logger.info(f"=== SESSION DEBUG: Loaded progress for session {session_id} ===")
            logger.info(f"Progress file: {progress_file}")
            logger.info(f"Loaded {len(completed_frames)} completed frames")
            
            return progress_data
            
        except Exception as e:
            logger.warning(f"Failed to load progress for session {session_id}: {e}")
            return None
    
    def find_resumable_session(self, video_path: str, settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find existing resumable session for given video and settings"""
        # === DEBUG: レジューム検索の詳細ログ ===
        logger.info(f"=== RESUME DEBUG: Searching for resumable session ===")
        logger.info(f"Video path: {video_path}")
        logger.info(f"Settings: {settings}")
        
        session_id = self.generate_session_id(video_path, settings)
        logger.info(f"Generated session ID: {session_id}")
        
        progress_data = self.load_progress(session_id)
        
        if not progress_data:
            logger.info("No progress data found for this session")
            return None
        
        logger.info("Progress data found, checking if resumable...")
        resumable = self._is_session_resumable(progress_data)
        
        if resumable:
            logger.info(f"Found resumable session {session_id} for {Path(video_path).name}")
            return progress_data
        else:
            logger.info(f"Session {session_id} exists but is not resumable")
            return None
    
    def _is_session_resumable(self, progress_data: Dict[str, Any]) -> bool:
        """Check if session can be resumed"""
        try:
            # === DEBUG: レジューム可能性チェックの詳細ログ ===
            logger.info(f"=== RESUME DEBUG: Checking if session is resumable ===")
            
            # Check if video file still exists
            video_path = progress_data.get('video_file')
            logger.info(f"Video file in session: {video_path}")
            
            if not video_path:
                logger.info("No video file path in progress data - not resumable")
                return False
            
            if not Path(video_path).exists():
                logger.info(f"Video file does not exist: {video_path} - not resumable")
                return False
            
            logger.info("Video file exists")
            
            # Check if session is not too old (e.g., older than 7 days)
            last_updated_str = progress_data.get('last_updated', '')
            logger.info(f"Last updated: {last_updated_str}")
            
            if last_updated_str:
                last_updated = datetime.fromisoformat(last_updated_str)
                age = datetime.now() - last_updated
                logger.info(f"Session age: {age}")
                
                if age > timedelta(days=7):
                    logger.info("Session is too old (>7 days) - not resumable")
                    return False
            
            logger.info("Session age is acceptable")
            
            # Check if any meaningful progress was made
            steps = progress_data.get('steps', {})
            logger.info(f"Steps in session: {list(steps.keys())}")
            
            has_progress = False
            for step_name, step_data in steps.items():
                status = step_data.get('status')
                logger.info(f"Step {step_name}: status = {status}")
                
                if status in ['completed', 'in_progress', 'cancelled', 'failed']:
                    has_progress = True
                    logger.info(f"Found meaningful progress in step {step_name} (status: {status})")
                    
                    # Check for completed frames specifically
                    if step_name == 'upscale':
                        completed_frames = step_data.get('completed_frames', [])
                        logger.info(f"Upscale step has {len(completed_frames)} completed frames")
            
            if has_progress:
                logger.info("Session has meaningful progress - resumable")
            else:
                logger.info("No meaningful progress found - not resumable")
            
            return has_progress
            
        except Exception as e:
            logger.warning(f"Error checking session resumability: {e}")
            return False
    
    def get_all_resumable_sessions(self) -> List[Dict[str, Any]]:
        """Get all resumable sessions"""
        resumable_sessions = []
        
        try:
            for session_dir in self.sessions_dir.iterdir():
                if session_dir.is_dir():
                    progress_data = self.load_progress(session_dir.name)
                    if progress_data and self._is_session_resumable(progress_data):
                        resumable_sessions.append(progress_data)
        
        except Exception as e:
            logger.warning(f"Error scanning for resumable sessions: {e}")
        
        return resumable_sessions
    
    def update_step_status(self, session_id: str, step_name: str, status: str, 
                          progress: float = None, additional_data: Dict[str, Any] = None) -> None:
        """Update status of a specific processing step"""
        progress_data = self.load_progress(session_id)
        if not progress_data:
            logger.warning(f"Cannot update step {step_name}: session {session_id} not found")
            return
        
        step_data = progress_data['steps'].get(step_name, {})
        
        # Update basic status
        step_data['status'] = status
        if progress is not None:
            step_data['progress'] = progress
        
        # Set timestamps
        if status == 'in_progress' and not step_data.get('start_time'):
            step_data['start_time'] = datetime.now().isoformat()
        elif status in ['completed', 'failed']:
            step_data['end_time'] = datetime.now().isoformat()
        
        # Add additional data
        if additional_data:
            step_data.update(additional_data)
        
        progress_data['steps'][step_name] = step_data
        self.save_progress(session_id, progress_data)
        
        logger.debug(f"Updated step {step_name} status to {status} for session {session_id}")
    
    def add_completed_frame(self, session_id: str, frame_path: str) -> None:
        """Add completed frame to upscale step tracking"""
        # === DEBUG: フレーム追加処理の詳細ログ ===
        logger.info(f"=== SESSION DEBUG: Adding frame to session {session_id} ===")
        logger.info(f"Frame path: {frame_path}")
        
        progress_data = self.load_progress(session_id)
        if not progress_data:
            logger.warning(f"No progress data found for session {session_id}")
            return
        
        upscale_step = progress_data['steps']['upscale']
        completed_frames = upscale_step.get('completed_frames', [])
        
        logger.info(f"Current completed frames count: {len(completed_frames)}")
        
        normalized_frame_path = frame_path.replace('\\', '/')
        logger.info(f"Normalized path: {normalized_frame_path}")
        
        if normalized_frame_path not in completed_frames:
            completed_frames.append(normalized_frame_path)
            upscale_step['completed_frames'] = completed_frames
            
            # Update progress percentage
            total_frames = upscale_step.get('total_frames', 0)
            if total_frames > 0:
                progress = len(completed_frames) / total_frames * 100
                upscale_step['progress'] = progress
                logger.info(f"Progress updated: {len(completed_frames)}/{total_frames} ({progress:.1f}%)")
            
            logger.info(f"Frame added successfully. New total: {len(completed_frames)}")
            self.save_progress(session_id, progress_data)
        else:
            logger.info(f"Frame already exists in session, skipping duplicate")
    
    def get_completed_frames(self, session_id: str) -> List[str]:
        """Get list of completed upscaled frames"""
        # === DEBUG: フレーム取得処理の詳細ログ ===
        logger.info(f"=== SESSION DEBUG: Getting completed frames for session {session_id} ===")
        
        progress_data = self.load_progress(session_id)
        if not progress_data:
            logger.warning(f"No progress data found for session {session_id}")
            return []
        
        completed_frames = progress_data['steps']['upscale'].get('completed_frames', [])
        logger.info(f"Retrieved {len(completed_frames)} completed frames from session")
        
        if completed_frames:
            logger.info(f"First frame: {completed_frames[0]}")
            logger.info(f"Last frame: {completed_frames[-1]}")
        else:
            logger.warning("No completed frames found in session data")
        
        return completed_frames
    
    def get_remaining_frames(self, session_id: str, all_frames: List[str]) -> List[str]:
        """Get list of frames that still need processing"""
        completed_frames = set(self.get_completed_frames(session_id))
        return [frame for frame in all_frames if frame not in completed_frames]
    
    def cleanup_session(self, session_id: str) -> None:
        """Clean up session directory and files"""
        try:
            session_dir = self.get_session_dir(session_id)
            if session_dir.exists():
                shutil.rmtree(session_dir)
                logger.info(f"Cleaned up session {session_id}")
        except Exception as e:
            logger.warning(f"Failed to cleanup session {session_id}: {e}")
    
    def cleanup_old_sessions(self, max_age_days: int = 7) -> None:
        """Clean up sessions older than specified days"""
        try:
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            cleaned_count = 0
            
            for session_dir in self.sessions_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                
                progress_data = self.load_progress(session_dir.name)
                if not progress_data:
                    continue
                
                last_updated = datetime.fromisoformat(progress_data.get('last_updated', ''))
                if last_updated < cutoff_time:
                    self.cleanup_session(session_dir.name)
                    cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old sessions")
                
        except Exception as e:
            logger.warning(f"Error cleaning up old sessions: {e}")
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary information about a session for UI display"""
        progress_data = self.load_progress(session_id)
        if not progress_data:
            return {}
        
        video_info = progress_data.get('video_info', {})
        steps = progress_data.get('steps', {})
        
        # Calculate overall progress
        step_weights = {'validate': 5, 'extract': 15, 'upscale': 70, 'combine': 10}
        total_progress = 0
        
        for step_name, weight in step_weights.items():
            step_progress = steps.get(step_name, {}).get('progress', 0)
            total_progress += (step_progress * weight / 100)
        
        return {
            'session_id': session_id,
            'video_name': Path(progress_data.get('video_file', '')).name,
            'video_info': video_info,
            'settings': progress_data.get('settings', {}),
            'created_at': progress_data.get('created_at'),
            'last_updated': progress_data.get('last_updated'),
            'overall_progress': total_progress,
            'current_step': self._get_current_step(steps),
            'steps': steps
        }
    
    def _get_current_step(self, steps: Dict[str, Any]) -> str:
        """Determine current processing step"""
        step_order = ['validate', 'extract', 'upscale', 'combine']
        
        for step_name in reversed(step_order):
            step_status = steps.get(step_name, {}).get('status', 'pending')
            if step_status == 'in_progress':
                return step_name
            elif step_status == 'completed':
                # Return next step if available
                next_index = step_order.index(step_name) + 1
                if next_index < len(step_order):
                    return step_order[next_index]
                else:
                    return 'completed'
        
        return 'validate'  # Default starting step