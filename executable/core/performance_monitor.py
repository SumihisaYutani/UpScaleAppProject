"""
UpScale App - Performance Monitoring Module
GPU効率最適化のための処理時間計測と並列処理管理
"""

import time
import threading
import queue
import statistics
import logging
import psutil
import multiprocessing
import concurrent.futures
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ProcessingStats:
    """処理統計データクラス"""
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    total_processed: int = 0
    success_count: int = 0
    failure_count: int = 0
    throughput: float = 0.0  # frames per second
    
    def update(self, processing_time: float, success: bool = True):
        """統計を更新"""
        self.total_processed += 1
        if success:
            self.success_count += 1
            self.min_time = min(self.min_time, processing_time)
            self.max_time = max(self.max_time, processing_time)
            
            # 移動平均でavg_timeを更新
            if self.avg_time == 0.0:
                self.avg_time = processing_time
            else:
                # 指数移動平均（α=0.1）
                self.avg_time = 0.9 * self.avg_time + 0.1 * processing_time
                
            self.throughput = 1.0 / self.avg_time if self.avg_time > 0 else 0.0
        else:
            self.failure_count += 1

class PerformanceMonitor:
    """パフォーマンス監視クラス"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.stats = ProcessingStats()
        self.recent_times = []
        self.gpu_utilization_history = []
        self.lock = threading.Lock()
        
        # CPU/メモリ監視
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.gpu_memory_used = 0.0
        
    def start_measurement(self) -> float:
        """測定開始"""
        return time.time()
        
    def end_measurement(self, start_time: float, success: bool = True) -> float:
        """測定終了"""
        processing_time = time.time() - start_time
        
        with self.lock:
            self.stats.update(processing_time, success)
            
            # 最近の処理時間を記録（ウィンドウサイズ分のみ保持）
            if success:
                self.recent_times.append(processing_time)
                if len(self.recent_times) > self.window_size:
                    self.recent_times.pop(0)
        
        return processing_time
    
    def get_current_stats(self) -> ProcessingStats:
        """現在の統計を取得"""
        with self.lock:
            return self.stats
    
    def get_recommended_parallelism(self, max_workers: int = None) -> int:
        """推奨並列度を計算"""
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), 8)
        
        with self.lock:
            if len(self.recent_times) < 5:
                return 2  # 初期値
            
            # 最近の処理時間の変動係数を計算
            mean_time = statistics.mean(self.recent_times)
            std_time = statistics.stdev(self.recent_times) if len(self.recent_times) > 1 else 0
            cv = std_time / mean_time if mean_time > 0 else 0
            
            # CPU使用率を考慮
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # GPU最適化のため並列度を制限し、GPU内並列化を優先
            if cpu_usage < 50 and self.stats.avg_time > 0.3:  # GPU処理時間が短い場合
                return min(max_workers, 3)  # 最大3並列
            elif cpu_usage < 70:
                return min(max_workers, 2)  # 2並列
            else:
                return 1  # 逐次処理でGPU集中

class OptimizedParallelProcessor:
    """最適化された並列処理クラス"""
    
    def __init__(self, ai_processor, max_workers: int = None):
        self.ai_processor = ai_processor
        # GPU並列化を優先し、CPU並列を抑制
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 3)
        self.monitor = PerformanceMonitor()
        
        # 動的並列度調整 - GPU効率優先
        self.current_workers = min(3, self.max_workers)  # 3ワーカーで開始（GPU効率とのバランス）
        self.adjustment_interval = 15  # 15フレームごとに調整
        self.processed_count = 0
        self.gpu_intensive_mode = True  # GPU内並列化優先モード
        
        logger.info(f"OptimizedParallelProcessor initialized - Max workers: {self.max_workers}, Current workers: {self.current_workers}")
    
    def process_frames_parallel(self, frame_paths: List[str], output_dir: str, 
                              scale_factor: float = 2.0,
                              progress_callback: Optional[Callable] = None,
                              progress_dialog=None) -> List[str]:
        """並列処理でフレームを処理"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_frames = len(frame_paths)
        processed_frames = []
        
        logger.info(f"Starting parallel processing of {total_frames} frames with {self.current_workers} workers")
        
        # フレームを分割してワーカーに配布
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.current_workers) as executor:
            future_to_frame = {}
            
            for i, frame_path in enumerate(frame_paths):
                # 動的並列度調整
                if i > 0 and i % self.adjustment_interval == 0:
                    self._adjust_parallelism(executor)
                
                frame_name = Path(frame_path).stem
                output_path = output_dir / f"{frame_name}_upscaled.png"
                
                future = executor.submit(
                    self._process_single_frame,
                    frame_path, str(output_path), scale_factor, i, total_frames, progress_dialog
                )
                future_to_frame[future] = (i, frame_path, str(output_path))
            
            # 結果を収集
            for future in concurrent.futures.as_completed(future_to_frame):
                frame_index, frame_path, output_path = future_to_frame[future]
                try:
                    success, processing_time = future.result()
                    
                    if success:
                        processed_frames.append(output_path)
                    
                    # 進捗更新
                    self.processed_count += 1
                    if progress_callback:
                        progress = self.processed_count / total_frames * 100
                        progress_callback(progress, f"Processed frame {self.processed_count}/{total_frames}")
                    
                    # 統計更新
                    self.monitor.end_measurement(processing_time, success)
                    
                    # パフォーマンス情報をログ出力
                    if self.processed_count % 10 == 0:
                        stats = self.monitor.get_current_stats()
                        logger.info(f"Performance: {stats.throughput:.2f} fps, "
                                  f"avg: {stats.avg_time:.2f}s, workers: {self.current_workers}")
                    
                except Exception as e:
                    logger.error(f"Error processing frame {frame_path}: {e}")
        
        # 最終統計
        final_stats = self.monitor.get_current_stats()
        logger.info(f"Parallel processing completed: {len(processed_frames)}/{total_frames} frames")
        logger.info(f"Final performance: {final_stats.throughput:.2f} fps, "
                   f"avg time: {final_stats.avg_time:.2f}s")
        
        return processed_frames
    
    def _process_single_frame(self, frame_path: str, output_path: str, 
                            scale_factor: float, frame_index: int, total_frames: int,
                            progress_dialog=None) -> tuple:
        """単一フレームの処理"""
        start_time = self.monitor.start_measurement()
        
        try:
            # GPU処理開始をGUIに通知
            if progress_dialog:
                # 推定GPU利用率（並列度に基づく）
                estimated_utilization = min(90, 30 + (self.current_workers * 15))
                progress_dialog.window.after(0, 
                    lambda: progress_dialog.update_gpu_status(
                        True, 
                        estimated_utilization, 
                        f"Processing frame {frame_index + 1} (Worker {threading.current_thread().name[-1:]})"
                    ))
            
            # デバッグ情報（最初の数フレームのみ）
            if progress_dialog and frame_index < 3:
                worker_id = threading.current_thread().name
                progress_dialog.window.after(0, 
                    lambda idx=frame_index, wid=worker_id: progress_dialog.add_log_message(
                        f"DEBUG: Processing frame {idx + 1} in parallel worker {wid}"))
            
            success = self.ai_processor.backend.upscale_image(
                frame_path, output_path, scale_factor, 
                progress_dialog=progress_dialog if frame_index < 3 else None
            )
            
            processing_time = time.time() - start_time
            
            # GPU処理完了をGUIに通知（短時間後にアイドルに戻す）
            if progress_dialog:
                # 処理完了の短い表示
                progress_dialog.window.after(0, 
                    lambda: progress_dialog.update_gpu_status(
                        True, 
                        10, 
                        f"Frame {frame_index + 1} completed ({processing_time:.2f}s)"
                    ))
                # 次のフレーム処理準備中として継続（アイドルに戻さない）
                progress_dialog.window.after(200, 
                    lambda: progress_dialog.update_gpu_status(
                        True,  # アクティブ状態を継続
                        25,    # 低めの利用率で準備中を示す
                        f"Processing batch... ({min(frame_index + 2, self.total_frames)}/{self.total_frames})"
                    ))
            
            return success, processing_time
            
        except Exception as e:
            logger.error(f"Error in worker processing frame {frame_index}: {e}")
            processing_time = time.time() - start_time
            
            # エラー時もGPUアイドルに戻す
            if progress_dialog:
                progress_dialog.window.after(0, 
                    lambda: progress_dialog.update_gpu_status(
                        False, 
                        0, 
                        f"Error processing frame {frame_index + 1}"
                    ))
            
            return False, processing_time
    
    def _adjust_parallelism(self, executor):
        """動的並列度調整"""
        recommended = self.monitor.get_recommended_parallelism(self.max_workers)
        
        if recommended != self.current_workers:
            old_workers = self.current_workers
            self.current_workers = recommended
            logger.info(f"Adjusting parallelism: {old_workers} -> {self.current_workers} workers")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """パフォーマンスレポートを取得"""
        stats = self.monitor.get_current_stats()
        
        return {
            'average_time_per_frame': stats.avg_time,
            'throughput_fps': stats.throughput,
            'total_processed': stats.total_processed,
            'success_rate': stats.success_count / stats.total_processed if stats.total_processed > 0 else 0,
            'current_workers': self.current_workers,
            'recommended_workers': self.monitor.get_recommended_parallelism(self.max_workers)
        }