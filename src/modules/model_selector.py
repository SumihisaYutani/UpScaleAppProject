"""
Intelligent Model Selection System
Automatically selects the optimal AI model based on content analysis and constraints
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

from modules.ai_model_manager import ModelInfo, ModelType, model_manager

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of image content"""
    PHOTO_REALISTIC = "photo_realistic"
    ANIME_ILLUSTRATION = "anime_illustration"
    GRAPHIC_DESIGN = "graphic_design"
    LINE_ART = "line_art"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ProcessingPriority(Enum):
    """Processing priorities"""
    SPEED = "speed"           # Prioritize fast processing
    BALANCED = "balanced"     # Balance speed and quality
    QUALITY = "quality"       # Prioritize high quality
    MEMORY_EFFICIENT = "memory_efficient"  # Minimize memory usage


@dataclass
class ContentAnalysis:
    """Results of content analysis"""
    content_type: ContentType
    complexity_score: float      # 0.0-1.0, higher = more complex
    edge_density: float          # 0.0-1.0, edge pixel ratio
    texture_complexity: float    # 0.0-1.0, texture variation
    color_diversity: float       # 0.0-1.0, color palette richness
    noise_level: float          # 0.0-1.0, estimated noise
    resolution: Tuple[int, int] # Original resolution (width, height)
    file_size_mb: float         # File size in MB
    estimated_processing_time: Dict[str, float] = None  # Model ID -> estimated time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content_type": self.content_type.value,
            "complexity_score": self.complexity_score,
            "edge_density": self.edge_density,
            "texture_complexity": self.texture_complexity,
            "color_diversity": self.color_diversity,
            "noise_level": self.noise_level,
            "resolution": self.resolution,
            "file_size_mb": self.file_size_mb,
            "estimated_processing_time": self.estimated_processing_time or {}
        }


@dataclass
class ProcessingConstraints:
    """Processing constraints and preferences"""
    priority: ProcessingPriority = ProcessingPriority.BALANCED
    max_processing_time_seconds: Optional[float] = None
    max_memory_usage_mb: Optional[float] = None
    min_quality_score: float = 7.0
    target_scale_factor: float = 1.5
    gpu_available: bool = True
    prefer_model_types: List[ModelType] = None
    exclude_model_types: List[ModelType] = None
    
    def __post_init__(self):
        if self.prefer_model_types is None:
            self.prefer_model_types = []
        if self.exclude_model_types is None:
            self.exclude_model_types = []


@dataclass
class ModelScore:
    """Model evaluation score"""
    model_id: str
    total_score: float
    speed_score: float
    quality_score: float
    memory_score: float
    compatibility_score: float
    estimated_time: float
    estimated_memory: float
    reasoning: List[str]  # Human-readable reasoning


class ContentAnalyzer:
    """Analyzes image content for optimal model selection"""
    
    def __init__(self):
        self.cache = {}  # Cache analysis results
    
    def analyze_image(self, image_path: str) -> ContentAnalysis:
        """Analyze image content characteristics"""
        
        # Check cache
        if image_path in self.cache:
            return self.cache[image_path]
        
        try:
            # Load image
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Basic properties
            height, width = image.shape[:2]
            resolution = (width, height)
            file_size_mb = Path(image_path).stat().st_size / 1024 / 1024
            
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Analyze content characteristics
            content_type = self._detect_content_type(image, gray, hsv)
            complexity_score = self._calculate_complexity(gray)
            edge_density = self._calculate_edge_density(gray)
            texture_complexity = self._calculate_texture_complexity(gray)
            color_diversity = self._calculate_color_diversity(hsv)
            noise_level = self._estimate_noise_level(gray)
            
            analysis = ContentAnalysis(
                content_type=content_type,
                complexity_score=complexity_score,
                edge_density=edge_density,
                texture_complexity=texture_complexity,
                color_diversity=color_diversity,
                noise_level=noise_level,
                resolution=resolution,
                file_size_mb=file_size_mb
            )
            
            # Cache result
            self.cache[image_path] = analysis
            
            logger.debug(f"Content analysis for {image_path}: {analysis.to_dict()}")
            return analysis
            
        except Exception as e:
            logger.error(f"Content analysis failed for {image_path}: {e}")
            # Return default analysis
            return ContentAnalysis(
                content_type=ContentType.UNKNOWN,
                complexity_score=0.5,
                edge_density=0.5,
                texture_complexity=0.5,
                color_diversity=0.5,
                noise_level=0.3,
                resolution=(1024, 1024),
                file_size_mb=1.0
            )
    
    def _detect_content_type(self, image: np.ndarray, gray: np.ndarray, 
                           hsv: np.ndarray) -> ContentType:
        """Detect the type of image content"""
        
        # Analyze color characteristics
        saturation = hsv[:, :, 1].mean() / 255.0
        value_std = hsv[:, :, 2].std() / 255.0
        
        # Analyze edge characteristics
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # Analyze gradient characteristics
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_std = gradient_magnitude.std()
        
        # Classification rules (simplified heuristics)
        if saturation > 0.6 and edge_ratio > 0.15 and gradient_std > 30:
            return ContentType.ANIME_ILLUSTRATION
        elif saturation < 0.3 and edge_ratio > 0.2:
            return ContentType.LINE_ART
        elif value_std > 0.3 and gradient_std > 40:
            return ContentType.PHOTO_REALISTIC
        elif edge_ratio < 0.1 and saturation > 0.4:
            return ContentType.GRAPHIC_DESIGN
        else:
            return ContentType.MIXED
    
    def _calculate_complexity(self, gray: np.ndarray) -> float:
        """Calculate image complexity score"""
        
        # Use gradient magnitude as complexity measure
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-1 range
        complexity = np.clip(gradient_magnitude.mean() / 100.0, 0.0, 1.0)
        return float(complexity)
    
    def _calculate_edge_density(self, gray: np.ndarray) -> float:
        """Calculate edge pixel density"""
        
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        return float(edge_ratio)
    
    def _calculate_texture_complexity(self, gray: np.ndarray) -> float:
        """Calculate texture complexity using Local Binary Patterns"""
        
        try:
            # Simple texture measure using standard deviation of local patches
            kernel = np.ones((5, 5), np.float32) / 25
            mean_filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            variance = np.mean((gray.astype(np.float32) - mean_filtered) ** 2)
            texture_complexity = np.clip(variance / 1000.0, 0.0, 1.0)
            return float(texture_complexity)
        except:
            return 0.5
    
    def _calculate_color_diversity(self, hsv: np.ndarray) -> float:
        """Calculate color palette diversity"""
        
        # Count unique hue values
        hue = hsv[:, :, 0]
        unique_hues = len(np.unique(hue[hsv[:, :, 1] > 50]))  # Ignore low-saturation pixels
        
        # Normalize to 0-1 range (max 180 hue values)
        diversity = min(unique_hues / 180.0, 1.0)
        return float(diversity)
    
    def _estimate_noise_level(self, gray: np.ndarray) -> float:
        """Estimate noise level in image"""
        
        try:
            # Use Laplacian variance as noise estimate
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            noise_level = np.clip(laplacian_var / 2000.0, 0.0, 1.0)
            return float(noise_level)
        except:
            return 0.3


class ModelSelector:
    """Intelligently selects optimal AI model based on analysis and constraints"""
    
    def __init__(self):
        self.content_analyzer = ContentAnalyzer()
        self.performance_cache = {}  # Cache model performance data
    
    def select_optimal_model(self, image_path: str, 
                           constraints: ProcessingConstraints) -> Optional[str]:
        """Select the optimal model for given image and constraints"""
        
        # Analyze content
        content_analysis = self.content_analyzer.analyze_image(image_path)
        
        # Get available models
        available_models = model_manager.get_available_models()
        
        # Score all models
        model_scores = []
        for model_info in available_models:
            score = self._score_model(model_info, content_analysis, constraints)
            if score.total_score > 0:  # Only include compatible models
                model_scores.append(score)
        
        if not model_scores:
            logger.warning("No compatible models found")
            return None
        
        # Sort by total score (descending)
        model_scores.sort(key=lambda x: x.total_score, reverse=True)
        
        best_model = model_scores[0]
        logger.info(f"Selected model {best_model.model_id} with score {best_model.total_score:.2f}")
        logger.info(f"Selection reasoning: {'; '.join(best_model.reasoning)}")
        
        return best_model.model_id
    
    def rank_models(self, image_path: str, 
                   constraints: ProcessingConstraints) -> List[ModelScore]:
        """Rank all models for given image and constraints"""
        
        content_analysis = self.content_analyzer.analyze_image(image_path)
        available_models = model_manager.get_available_models()
        
        model_scores = []
        for model_info in available_models:
            score = self._score_model(model_info, content_analysis, constraints)
            model_scores.append(score)
        
        # Sort by total score (descending)
        model_scores.sort(key=lambda x: x.total_score, reverse=True)
        
        return model_scores
    
    def _score_model(self, model_info: ModelInfo, content_analysis: ContentAnalysis,
                    constraints: ProcessingConstraints) -> ModelScore:
        """Score a model for given content and constraints"""
        
        reasoning = []
        
        # Check basic compatibility
        if model_info.type in constraints.exclude_model_types:
            return ModelScore(
                model_id=model_info.id,
                total_score=0.0,
                speed_score=0.0,
                quality_score=0.0,
                memory_score=0.0,
                compatibility_score=0.0,
                estimated_time=float('inf'),
                estimated_memory=model_info.memory_usage_mb,
                reasoning=["Model type excluded by constraints"]
            )
        
        # Check GPU requirements
        if model_info.gpu_required and not constraints.gpu_available:
            return ModelScore(
                model_id=model_info.id,
                total_score=0.0,
                speed_score=0.0,
                quality_score=0.0,
                memory_score=0.0,
                compatibility_score=0.0,
                estimated_time=float('inf'),
                estimated_memory=model_info.memory_usage_mb,
                reasoning=["GPU required but not available"]
            )
        
        # Check memory constraints
        if (constraints.max_memory_usage_mb and 
            model_info.memory_usage_mb > constraints.max_memory_usage_mb):
            return ModelScore(
                model_id=model_info.id,
                total_score=0.0,
                speed_score=0.0,
                quality_score=0.0,
                memory_score=0.0,
                compatibility_score=0.0,
                estimated_time=float('inf'),
                estimated_memory=model_info.memory_usage_mb,
                reasoning=[f"Memory usage {model_info.memory_usage_mb}MB exceeds limit"]
            )
        
        # Check scale factor support
        if constraints.target_scale_factor not in model_info.supported_scales:
            # Find closest supported scale
            closest_scale = min(model_info.supported_scales, 
                              key=lambda x: abs(x - constraints.target_scale_factor))
            if abs(closest_scale - constraints.target_scale_factor) > 0.5:
                reasoning.append(f"Scale factor {constraints.target_scale_factor} not well supported")
        
        # Calculate individual scores
        speed_score = self._calculate_speed_score(model_info, content_analysis, constraints)
        quality_score = self._calculate_quality_score(model_info, content_analysis, constraints)
        memory_score = self._calculate_memory_score(model_info, constraints)
        compatibility_score = self._calculate_compatibility_score(model_info, content_analysis, constraints)
        
        # Estimate processing time and memory
        estimated_time = self._estimate_processing_time(model_info, content_analysis)
        estimated_memory = model_info.memory_usage_mb
        
        # Check time constraints
        if (constraints.max_processing_time_seconds and 
            estimated_time > constraints.max_processing_time_seconds):
            reasoning.append(f"Estimated time {estimated_time:.1f}s exceeds limit")
            speed_score *= 0.5  # Penalize but don't eliminate
        
        # Calculate weighted total score based on priority
        weights = self._get_priority_weights(constraints.priority)
        total_score = (
            weights['speed'] * speed_score +
            weights['quality'] * quality_score +
            weights['memory'] * memory_score +
            weights['compatibility'] * compatibility_score
        )
        
        # Apply preference bonuses
        if model_info.type in constraints.prefer_model_types:
            total_score *= 1.2
            reasoning.append("Preferred model type")
        
        # Add quality reasoning
        if quality_score > 0.8:
            reasoning.append("High quality model")
        if speed_score > 0.8:
            reasoning.append("Fast processing")
        if memory_score > 0.8:
            reasoning.append("Memory efficient")
        if compatibility_score > 0.8:
            reasoning.append("Well suited for content type")
        
        return ModelScore(
            model_id=model_info.id,
            total_score=total_score,
            speed_score=speed_score,
            quality_score=quality_score,
            memory_score=memory_score,
            compatibility_score=compatibility_score,
            estimated_time=estimated_time,
            estimated_memory=estimated_memory,
            reasoning=reasoning
        )
    
    def _calculate_speed_score(self, model_info: ModelInfo, content_analysis: ContentAnalysis,
                              constraints: ProcessingConstraints) -> float:
        """Calculate speed score (0-1, higher = faster)"""
        
        base_speed = model_info.processing_speed_fps if model_info.processing_speed_fps > 0 else 1.0
        
        # Adjust for content complexity
        complexity_factor = 1.0 - (content_analysis.complexity_score * 0.3)
        
        # Adjust for resolution
        pixel_count = content_analysis.resolution[0] * content_analysis.resolution[1]
        resolution_factor = max(0.1, 1.0 - (pixel_count / 4000000))  # 4MP baseline
        
        adjusted_speed = base_speed * complexity_factor * resolution_factor
        
        # Normalize to 0-1 scale (assuming max reasonable speed is 10 FPS)
        speed_score = min(adjusted_speed / 10.0, 1.0)
        
        return speed_score
    
    def _calculate_quality_score(self, model_info: ModelInfo, content_analysis: ContentAnalysis,
                                constraints: ProcessingConstraints) -> float:
        """Calculate quality score (0-1, higher = better quality)"""
        
        base_quality = model_info.quality_score / 10.0  # Normalize to 0-1
        
        # Adjust for content type compatibility
        content_bonus = self._get_content_type_bonus(model_info.type, content_analysis.content_type)
        
        # Check minimum quality requirement
        if model_info.quality_score < constraints.min_quality_score:
            return 0.0
        
        quality_score = min(base_quality * content_bonus, 1.0)
        return quality_score
    
    def _calculate_memory_score(self, model_info: ModelInfo, 
                              constraints: ProcessingConstraints) -> float:
        """Calculate memory efficiency score (0-1, higher = more efficient)"""
        
        # Normalize memory usage (assume 16GB as high usage)
        memory_usage_normalized = model_info.memory_usage_mb / 16000.0
        memory_score = max(0.0, 1.0 - memory_usage_normalized)
        
        return memory_score
    
    def _calculate_compatibility_score(self, model_info: ModelInfo, 
                                     content_analysis: ContentAnalysis,
                                     constraints: ProcessingConstraints) -> float:
        """Calculate content compatibility score"""
        
        compatibility = 0.5  # Base compatibility
        
        # Content type specific bonuses
        content_bonus = self._get_content_type_bonus(model_info.type, content_analysis.content_type)
        compatibility *= content_bonus
        
        # Scale factor compatibility
        if constraints.target_scale_factor in model_info.supported_scales:
            compatibility *= 1.2
        
        # Format compatibility
        supported_formats = model_info.supported_formats
        if any(fmt in ['.jpg', '.jpeg', '.png'] for fmt in supported_formats):
            compatibility *= 1.1
        
        return min(compatibility, 1.0)
    
    def _get_content_type_bonus(self, model_type: ModelType, content_type: ContentType) -> float:
        """Get content type compatibility bonus"""
        
        # Model type vs content type compatibility matrix
        compatibility_matrix = {
            ModelType.STABLE_DIFFUSION: {
                ContentType.PHOTO_REALISTIC: 1.2,
                ContentType.ANIME_ILLUSTRATION: 1.3,
                ContentType.GRAPHIC_DESIGN: 1.1,
                ContentType.LINE_ART: 0.9,
                ContentType.MIXED: 1.0,
                ContentType.UNKNOWN: 1.0
            },
            ModelType.REAL_ESRGAN: {
                ContentType.PHOTO_REALISTIC: 1.3,
                ContentType.ANIME_ILLUSTRATION: 1.1,
                ContentType.GRAPHIC_DESIGN: 0.9,
                ContentType.LINE_ART: 0.8,
                ContentType.MIXED: 1.0,
                ContentType.UNKNOWN: 1.0
            },
            ModelType.EDSR: {
                ContentType.PHOTO_REALISTIC: 1.1,
                ContentType.ANIME_ILLUSTRATION: 0.9,
                ContentType.GRAPHIC_DESIGN: 1.0,
                ContentType.LINE_ART: 1.2,
                ContentType.MIXED: 1.0,
                ContentType.UNKNOWN: 1.0
            }
        }
        
        return compatibility_matrix.get(model_type, {}).get(content_type, 1.0)
    
    def _get_priority_weights(self, priority: ProcessingPriority) -> Dict[str, float]:
        """Get scoring weights based on processing priority"""
        
        weights = {
            ProcessingPriority.SPEED: {
                'speed': 0.5,
                'quality': 0.2,
                'memory': 0.15,
                'compatibility': 0.15
            },
            ProcessingPriority.QUALITY: {
                'speed': 0.15,
                'quality': 0.5,
                'memory': 0.1,
                'compatibility': 0.25
            },
            ProcessingPriority.BALANCED: {
                'speed': 0.3,
                'quality': 0.3,
                'memory': 0.2,
                'compatibility': 0.2
            },
            ProcessingPriority.MEMORY_EFFICIENT: {
                'speed': 0.2,
                'quality': 0.2,
                'memory': 0.4,
                'compatibility': 0.2
            }
        }
        
        return weights[priority]
    
    def _estimate_processing_time(self, model_info: ModelInfo, 
                                content_analysis: ContentAnalysis) -> float:
        """Estimate processing time for model and content"""
        
        base_time = 30.0  # Base time in seconds
        
        if model_info.processing_speed_fps > 0:
            # Calculate based on FPS
            pixel_count = content_analysis.resolution[0] * content_analysis.resolution[1]
            frame_equivalent = pixel_count / (1920 * 1080)  # Normalize to 1080p
            base_time = frame_equivalent / model_info.processing_speed_fps
        
        # Adjust for complexity
        complexity_multiplier = 1.0 + (content_analysis.complexity_score * 0.5)
        estimated_time = base_time * complexity_multiplier
        
        return estimated_time
    
    def clear_cache(self):
        """Clear analysis cache"""
        self.content_analyzer.cache.clear()
        self.performance_cache.clear()


# Global model selector instance
model_selector = ModelSelector()