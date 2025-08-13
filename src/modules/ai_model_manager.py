"""
AI Model Manager
Manages multiple AI models for upscaling with unified interface
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

import torch
import psutil
import requests
from concurrent.futures import ThreadPoolExecutor

from config.settings import PATHS

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """AI model types"""
    STABLE_DIFFUSION = "stable_diffusion"
    REAL_ESRGAN = "real_esrgan"
    EDSR = "edsr"
    SRCNN = "srcnn"
    CUSTOM = "custom"


class ModelStatus(Enum):
    """Model loading status"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Model information and metadata"""
    id: str
    name: str
    type: ModelType
    version: str
    description: str
    file_path: str
    download_url: str = ""
    file_size_mb: float = 0.0
    memory_usage_mb: float = 0.0
    supported_scales: List[float] = field(default_factory=lambda: [1.5, 2.0])
    supported_formats: List[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg"])
    gpu_required: bool = True
    min_vram_mb: float = 2000.0
    processing_speed_fps: float = 0.0  # Frames per second
    quality_score: float = 0.0  # Quality benchmark score
    license: str = "unknown"
    author: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "version": self.version,
            "description": self.description,
            "file_path": self.file_path,
            "download_url": self.download_url,
            "file_size_mb": self.file_size_mb,
            "memory_usage_mb": self.memory_usage_mb,
            "supported_scales": self.supported_scales,
            "supported_formats": self.supported_formats,
            "gpu_required": self.gpu_required,
            "min_vram_mb": self.min_vram_mb,
            "processing_speed_fps": self.processing_speed_fps,
            "quality_score": self.quality_score,
            "license": self.license,
            "author": self.author,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """Create from dictionary"""
        return cls(
            id=data["id"],
            name=data["name"],
            type=ModelType(data["type"]),
            version=data["version"],
            description=data["description"],
            file_path=data["file_path"],
            download_url=data.get("download_url", ""),
            file_size_mb=data.get("file_size_mb", 0.0),
            memory_usage_mb=data.get("memory_usage_mb", 0.0),
            supported_scales=data.get("supported_scales", [1.5, 2.0]),
            supported_formats=data.get("supported_formats", [".png", ".jpg", ".jpeg"]),
            gpu_required=data.get("gpu_required", True),
            min_vram_mb=data.get("min_vram_mb", 2000.0),
            processing_speed_fps=data.get("processing_speed_fps", 0.0),
            quality_score=data.get("quality_score", 0.0),
            license=data.get("license", "unknown"),
            author=data.get("author", ""),
            tags=data.get("tags", [])
        )


class AIModelInterface(ABC):
    """Abstract interface for AI models"""
    
    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info
        self.status = ModelStatus.UNLOADED
        self.model = None
        self.device = "cpu"
        self.load_time = 0.0
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the model into memory"""
        pass
    
    @abstractmethod
    def unload_model(self):
        """Unload the model from memory"""
        pass
    
    @abstractmethod
    def process_image(self, image_path: str, output_path: str, 
                     scale_factor: float = 1.5, **kwargs) -> Dict[str, Any]:
        """Process a single image"""
        pass
    
    @abstractmethod
    def process_batch(self, image_paths: List[str], output_dir: str,
                     scale_factor: float = 1.5, **kwargs) -> List[Dict[str, Any]]:
        """Process multiple images"""
        pass
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.status == ModelStatus.LOADED
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if self.model is None:
            return 0.0
        
        try:
            if hasattr(self.model, 'get_memory_footprint'):
                return self.model.get_memory_footprint()
            elif torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
            else:
                return self.model_info.memory_usage_mb
        except:
            return self.model_info.memory_usage_mb


class StableDiffusionModel(AIModelInterface):
    """Stable Diffusion model wrapper"""
    
    def load_model(self) -> bool:
        """Load Stable Diffusion model"""
        try:
            self.status = ModelStatus.LOADING
            start_time = time.time()
            
            # Import Stable Diffusion components
            from diffusers import StableDiffusionUpscalePipeline
            
            # Determine device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model
            self.model = StableDiffusionUpscalePipeline.from_pretrained(
                self.model_info.file_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            # Enable memory efficient attention if available
            if hasattr(self.model, 'enable_attention_slicing'):
                self.model.enable_attention_slicing()
            
            self.load_time = time.time() - start_time
            self.status = ModelStatus.LOADED
            
            logger.info(f"Loaded Stable Diffusion model {self.model_info.name} in {self.load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion model: {e}")
            self.status = ModelStatus.ERROR
            return False
    
    def unload_model(self):
        """Unload Stable Diffusion model"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
                
                # Clear CUDA cache if using GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.status = ModelStatus.UNLOADED
                logger.info(f"Unloaded Stable Diffusion model {self.model_info.name}")
                
        except Exception as e:
            logger.error(f"Error unloading Stable Diffusion model: {e}")
    
    def process_image(self, image_path: str, output_path: str, 
                     scale_factor: float = 1.5, **kwargs) -> Dict[str, Any]:
        """Process single image with Stable Diffusion"""
        
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        try:
            from PIL import Image
            import numpy as np
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            original_size = image.size
            
            # Process with model
            prompt = kwargs.get("prompt", "high quality, detailed")
            num_inference_steps = kwargs.get("num_inference_steps", 20)
            guidance_scale = kwargs.get("guidance_scale", 7.5)
            
            with torch.no_grad():
                upscaled_image = self.model(
                    prompt=prompt,
                    image=image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
            
            # Save result
            upscaled_image.save(output_path)
            
            return {
                "success": True,
                "input_size": original_size,
                "output_size": upscaled_image.size,
                "scale_achieved": upscaled_image.size[0] / original_size[0],
                "processing_time": 0.0  # Would measure actual time
            }
            
        except Exception as e:
            logger.error(f"Stable Diffusion processing error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_batch(self, image_paths: List[str], output_dir: str,
                     scale_factor: float = 1.5, **kwargs) -> List[Dict[str, Any]]:
        """Process multiple images"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            output_path = Path(output_dir) / f"{Path(image_path).stem}_upscaled.png"
            result = self.process_image(str(image_path), str(output_path), scale_factor, **kwargs)
            results.append(result)
            
            # Progress callback if provided
            if "progress_callback" in kwargs:
                progress = (i + 1) / len(image_paths) * 100
                kwargs["progress_callback"](progress, f"Processed {i+1}/{len(image_paths)}")
        
        return results


class RealESRGANModel(AIModelInterface):
    """Real-ESRGAN model wrapper"""
    
    def load_model(self) -> bool:
        """Load Real-ESRGAN model"""
        try:
            self.status = ModelStatus.LOADING
            start_time = time.time()
            
            # Import Real-ESRGAN
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            # Determine device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=2
            )
            
            self.model = RealESRGANer(
                scale=2,
                model_path=self.model_info.file_path,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True if self.device == "cuda" else False,
                gpu_id=0 if self.device == "cuda" else None
            )
            
            self.load_time = time.time() - start_time
            self.status = ModelStatus.LOADED
            
            logger.info(f"Loaded Real-ESRGAN model {self.model_info.name} in {self.load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Real-ESRGAN model: {e}")
            self.status = ModelStatus.ERROR
            return False
    
    def unload_model(self):
        """Unload Real-ESRGAN model"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.status = ModelStatus.UNLOADED
                logger.info(f"Unloaded Real-ESRGAN model {self.model_info.name}")
                
        except Exception as e:
            logger.error(f"Error unloading Real-ESRGAN model: {e}")
    
    def process_image(self, image_path: str, output_path: str, 
                     scale_factor: float = 2.0, **kwargs) -> Dict[str, Any]:
        """Process single image with Real-ESRGAN"""
        
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        try:
            import cv2
            import numpy as np
            
            # Load image
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            original_size = (image.shape[1], image.shape[0])
            
            # Process with model
            upscaled_image, _ = self.model.enhance(image, outscale=scale_factor)
            
            # Save result
            cv2.imwrite(output_path, upscaled_image)
            
            return {
                "success": True,
                "input_size": original_size,
                "output_size": (upscaled_image.shape[1], upscaled_image.shape[0]),
                "scale_achieved": upscaled_image.shape[1] / original_size[0],
                "processing_time": 0.0
            }
            
        except Exception as e:
            logger.error(f"Real-ESRGAN processing error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_batch(self, image_paths: List[str], output_dir: str,
                     scale_factor: float = 2.0, **kwargs) -> List[Dict[str, Any]]:
        """Process multiple images"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            output_path = Path(output_dir) / f"{Path(image_path).stem}_realesrgan.png"
            result = self.process_image(str(image_path), str(output_path), scale_factor, **kwargs)
            results.append(result)
            
            if "progress_callback" in kwargs:
                progress = (i + 1) / len(image_paths) * 100
                kwargs["progress_callback"](progress, f"Processed {i+1}/{len(image_paths)}")
        
        return results


class AIModelManager:
    """Manages multiple AI models with unified interface"""
    
    def __init__(self):
        self.models_dir = PATHS["temp_dir"] / "ai_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.available_models: Dict[str, ModelInfo] = {}
        self.loaded_models: Dict[str, AIModelInterface] = {}
        
        # Model loading executor
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Callbacks
        self.model_load_callback: Optional[Callable] = None
        self.progress_callback: Optional[Callable] = None
        
        # Registry file
        self.registry_file = self.models_dir / "model_registry.json"
        
        # Load model registry
        self._initialize_default_models()
        self.load_model_registry()
    
    def _initialize_default_models(self):
        """Initialize default model definitions"""
        
        # Stable Diffusion models
        sd_models = [
            ModelInfo(
                id="stable_diffusion_2_1_upscale",
                name="Stable Diffusion 2.1 Upscale",
                type=ModelType.STABLE_DIFFUSION,
                version="2.1",
                description="High-quality image upscaling with Stable Diffusion",
                file_path="stabilityai/stable-diffusion-x4-upscaler",
                download_url="https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler",
                file_size_mb=5200.0,
                memory_usage_mb=6000.0,
                supported_scales=[4.0],
                gpu_required=True,
                min_vram_mb=8000.0,
                quality_score=8.5,
                license="CreativeML Open RAIL-M",
                author="Stability AI",
                tags=["stable_diffusion", "upscale", "high_quality"]
            ),
            ModelInfo(
                id="stable_diffusion_1_5_upscale",
                name="Stable Diffusion 1.5 Upscale",
                type=ModelType.STABLE_DIFFUSION,
                version="1.5",
                description="Stable Diffusion 1.5 based upscaling",
                file_path="runwayml/stable-diffusion-v1-5",
                download_url="https://huggingface.co/runwayml/stable-diffusion-v1-5",
                file_size_mb=4200.0,
                memory_usage_mb=5000.0,
                supported_scales=[2.0, 4.0],
                gpu_required=True,
                min_vram_mb=6000.0,
                quality_score=8.0,
                license="CreativeML Open RAIL-M",
                author="Runway ML",
                tags=["stable_diffusion", "upscale", "versatile"]
            )
        ]
        
        # Real-ESRGAN models
        realesrgan_models = [
            ModelInfo(
                id="realesrgan_x4plus",
                name="Real-ESRGAN x4plus",
                type=ModelType.REAL_ESRGAN,
                version="0.2.5",
                description="Real-ESRGAN for real-world image restoration",
                file_path="RealESRGAN_x4plus.pth",
                download_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus.pth",
                file_size_mb=67.0,
                memory_usage_mb=1500.0,
                supported_scales=[4.0],
                gpu_required=False,
                min_vram_mb=2000.0,
                processing_speed_fps=2.5,
                quality_score=7.8,
                license="BSD-3-Clause",
                author="Tencent ARC",
                tags=["realesrgan", "fast", "photo"]
            ),
            ModelInfo(
                id="realesrgan_x2plus",
                name="Real-ESRGAN x2plus",
                type=ModelType.REAL_ESRGAN,
                version="0.2.5",
                description="Real-ESRGAN 2x upscaling for faster processing",
                file_path="RealESRGAN_x2plus.pth",
                download_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x2plus.pth",
                file_size_mb=67.0,
                memory_usage_mb=1200.0,
                supported_scales=[2.0],
                gpu_required=False,
                min_vram_mb=1500.0,
                processing_speed_fps=5.0,
                quality_score=7.5,
                license="BSD-3-Clause",
                author="Tencent ARC",
                tags=["realesrgan", "fast", "2x"]
            )
        ]
        
        # Add all default models
        for model in sd_models + realesrgan_models:
            self.available_models[model.id] = model
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models"""
        return list(self.available_models.values())
    
    def get_models_by_type(self, model_type: ModelType) -> List[ModelInfo]:
        """Get models filtered by type"""
        return [model for model in self.available_models.values() 
                if model.type == model_type]
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information"""
        return self.available_models.get(model_id)
    
    def is_model_loaded(self, model_id: str) -> bool:
        """Check if model is loaded"""
        return model_id in self.loaded_models and self.loaded_models[model_id].is_loaded()
    
    def load_model(self, model_id: str, async_load: bool = True) -> bool:
        """Load a model (async by default)"""
        
        if model_id not in self.available_models:
            logger.error(f"Model {model_id} not found in registry")
            return False
        
        if self.is_model_loaded(model_id):
            logger.info(f"Model {model_id} already loaded")
            return True
        
        model_info = self.available_models[model_id]
        
        # Check system requirements
        if not self._check_system_requirements(model_info):
            logger.error(f"System requirements not met for model {model_id}")
            return False
        
        # Create model instance
        model_instance = self._create_model_instance(model_info)
        if not model_instance:
            logger.error(f"Failed to create model instance for {model_id}")
            return False
        
        if async_load:
            # Load asynchronously
            future = self.executor.submit(self._load_model_sync, model_id, model_instance)
            return True  # Return immediately, check status later
        else:
            # Load synchronously
            return self._load_model_sync(model_id, model_instance)
    
    def _load_model_sync(self, model_id: str, model_instance: AIModelInterface) -> bool:
        """Synchronously load model"""
        try:
            if self.progress_callback:
                self.progress_callback("loading", model_id, 0)
            
            success = model_instance.load_model()
            
            if success:
                self.loaded_models[model_id] = model_instance
                logger.info(f"Successfully loaded model {model_id}")
                
                if self.model_load_callback:
                    self.model_load_callback("loaded", model_id, model_instance)
                
                if self.progress_callback:
                    self.progress_callback("loaded", model_id, 100)
            else:
                logger.error(f"Failed to load model {model_id}")
                if self.model_load_callback:
                    self.model_load_callback("error", model_id, None)
                
                if self.progress_callback:
                    self.progress_callback("error", model_id, 0)
            
            return success
            
        except Exception as e:
            logger.error(f"Exception loading model {model_id}: {e}")
            if self.model_load_callback:
                self.model_load_callback("error", model_id, None)
            return False
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model"""
        
        if model_id not in self.loaded_models:
            return True  # Already unloaded
        
        try:
            model_instance = self.loaded_models[model_id]
            model_instance.unload_model()
            del self.loaded_models[model_id]
            
            logger.info(f"Unloaded model {model_id}")
            
            if self.model_load_callback:
                self.model_load_callback("unloaded", model_id, None)
            
            return True
            
        except Exception as e:
            logger.error(f"Error unloading model {model_id}: {e}")
            return False
    
    def unload_all_models(self):
        """Unload all loaded models"""
        for model_id in list(self.loaded_models.keys()):
            self.unload_model(model_id)
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model IDs"""
        return [model_id for model_id, model in self.loaded_models.items() 
                if model.is_loaded()]
    
    def process_image(self, model_id: str, image_path: str, output_path: str,
                     scale_factor: float = 1.5, **kwargs) -> Dict[str, Any]:
        """Process image with specified model"""
        
        if not self.is_model_loaded(model_id):
            return {
                "success": False,
                "error": f"Model {model_id} not loaded"
            }
        
        model = self.loaded_models[model_id]
        return model.process_image(image_path, output_path, scale_factor, **kwargs)
    
    def process_batch(self, model_id: str, image_paths: List[str], output_dir: str,
                     scale_factor: float = 1.5, **kwargs) -> List[Dict[str, Any]]:
        """Process multiple images with specified model"""
        
        if not self.is_model_loaded(model_id):
            return [{
                "success": False,
                "error": f"Model {model_id} not loaded"
            }] * len(image_paths)
        
        model = self.loaded_models[model_id]
        return model.process_batch(image_paths, output_dir, scale_factor, **kwargs)
    
    def _create_model_instance(self, model_info: ModelInfo) -> Optional[AIModelInterface]:
        """Create model instance based on type"""
        
        if model_info.type == ModelType.STABLE_DIFFUSION:
            return StableDiffusionModel(model_info)
        elif model_info.type == ModelType.REAL_ESRGAN:
            return RealESRGANModel(model_info)
        else:
            logger.error(f"Unsupported model type: {model_info.type}")
            return None
    
    def _check_system_requirements(self, model_info: ModelInfo) -> bool:
        """Check if system meets model requirements"""
        
        # Check memory
        available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
        if available_memory < model_info.memory_usage_mb:
            logger.warning(f"Insufficient RAM: need {model_info.memory_usage_mb}MB, have {available_memory}MB")
            return False
        
        # Check GPU requirements
        if model_info.gpu_required:
            if not torch.cuda.is_available():
                logger.warning(f"GPU required for model {model_info.id} but CUDA not available")
                return False
            
            # Check VRAM
            if torch.cuda.is_available():
                vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                if vram_mb < model_info.min_vram_mb:
                    logger.warning(f"Insufficient VRAM: need {model_info.min_vram_mb}MB, have {vram_mb}MB")
                    return False
        
        return True
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for model compatibility"""
        
        info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_mb": psutil.virtual_memory().total / 1024 / 1024,
            "memory_available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": 0,
            "gpu_names": [],
            "vram_total_mb": 0
        }
        
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpu_names"].append(props.name)
                if i == 0:  # Primary GPU
                    info["vram_total_mb"] = props.total_memory / 1024 / 1024
        
        return info
    
    def save_model_registry(self):
        """Save model registry to file"""
        try:
            registry_data = {
                "models": {model_id: model_info.to_dict() 
                          for model_id, model_info in self.available_models.items()},
                "version": "1.0",
                "last_updated": time.time()
            }
            
            with open(self.registry_file, "w") as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def load_model_registry(self):
        """Load model registry from file"""
        try:
            if not self.registry_file.exists():
                self.save_model_registry()
                return
            
            with open(self.registry_file, "r") as f:
                registry_data = json.load(f)
            
            # Load custom models (preserve defaults)
            for model_id, model_data in registry_data.get("models", {}).items():
                if model_id not in self.available_models:
                    self.available_models[model_id] = ModelInfo.from_dict(model_data)
            
            logger.info(f"Loaded model registry with {len(self.available_models)} models")
            
        except Exception as e:
            logger.warning(f"Failed to load model registry: {e}")
    
    def shutdown(self):
        """Shutdown model manager"""
        logger.info("Shutting down AI Model Manager...")
        
        # Unload all models
        self.unload_all_models()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Save registry
        self.save_model_registry()
        
        logger.info("AI Model Manager shutdown complete")


# Global model manager instance
model_manager = AIModelManager()