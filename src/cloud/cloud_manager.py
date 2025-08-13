"""
Cloud Processing Manager
Manages cloud resources and hybrid local/cloud processing
"""

import os
import json
import time
import asyncio
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
import hashlib

import requests
import boto3
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient

from config.settings import PATHS

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers"""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    CUSTOM = "custom"


class InstanceType(Enum):
    """Cloud instance types"""
    # AWS
    AWS_P3_XLARGE = "p3.xlarge"      # 1x V100, 4 vCPUs, 61GB RAM
    AWS_P3_2XLARGE = "p3.2xlarge"   # 1x V100, 8 vCPUs, 122GB RAM
    AWS_P4D_XLARGE = "p4d.xlarge"   # 1x A100, 4 vCPUs, 96GB RAM
    AWS_G4DN_XLARGE = "g4dn.xlarge" # 1x T4, 4 vCPUs, 16GB RAM
    
    # Azure
    AZURE_NC6S_V3 = "Standard_NC6s_v3"      # 1x V100, 6 vCPUs, 112GB RAM
    AZURE_NC12S_V3 = "Standard_NC12s_v3"    # 2x V100, 12 vCPUs, 224GB RAM
    AZURE_ND40RS_V2 = "Standard_ND40rs_v2"  # 8x V100, 40 vCPUs, 672GB RAM
    
    # GCP
    GCP_N1_HIGHMEM_4_K80 = "n1-highmem-4-k80"     # 1x K80, 4 vCPUs, 26GB RAM
    GCP_N1_HIGHMEM_8_T4 = "n1-highmem-8-t4"       # 1x T4, 8 vCPUs, 52GB RAM
    GCP_A2_HIGHGPU_1G = "a2-highgpu-1g"           # 1x A100, 12 vCPUs, 85GB RAM


class InstanceStatus(Enum):
    """Cloud instance status"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    TERMINATED = "terminated"
    ERROR = "error"


@dataclass
class CloudInstanceInfo:
    """Cloud instance information"""
    instance_id: str
    provider: CloudProvider
    instance_type: InstanceType
    region: str
    status: InstanceStatus
    public_ip: Optional[str] = None
    private_ip: Optional[str] = None
    created_at: Optional[float] = None
    cost_per_hour: float = 0.0
    gpu_count: int = 1
    gpu_type: str = "unknown"
    vram_gb: float = 0.0
    vcpus: int = 4
    ram_gb: float = 16.0
    storage_gb: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "instance_id": self.instance_id,
            "provider": self.provider.value,
            "instance_type": self.instance_type.value,
            "region": self.region,
            "status": self.status.value,
            "public_ip": self.public_ip,
            "private_ip": self.private_ip,
            "created_at": self.created_at,
            "cost_per_hour": self.cost_per_hour,
            "gpu_count": self.gpu_count,
            "gpu_type": self.gpu_type,
            "vram_gb": self.vram_gb,
            "vcpus": self.vcpus,
            "ram_gb": self.ram_gb,
            "storage_gb": self.storage_gb
        }


@dataclass
class CloudJob:
    """Cloud processing job"""
    job_id: str
    input_files: List[str]
    output_directory: str
    processing_params: Dict[str, Any]
    instance_info: CloudInstanceInfo
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    
    @property
    def processing_time(self) -> Optional[float]:
        """Get processing time in seconds"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "job_id": self.job_id,
            "input_files": self.input_files,
            "output_directory": self.output_directory,
            "processing_params": self.processing_params,
            "instance_info": self.instance_info.to_dict(),
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
            "estimated_cost": self.estimated_cost,
            "actual_cost": self.actual_cost
        }


class CloudProviderInterface:
    """Base interface for cloud providers"""
    
    def __init__(self, provider: CloudProvider, credentials: Dict[str, Any]):
        self.provider = provider
        self.credentials = credentials
        self.client = None
    
    async def authenticate(self) -> bool:
        """Authenticate with cloud provider"""
        raise NotImplementedError
    
    async def list_instances(self) -> List[CloudInstanceInfo]:
        """List available instances"""
        raise NotImplementedError
    
    async def create_instance(self, instance_type: InstanceType, 
                            region: str, **kwargs) -> CloudInstanceInfo:
        """Create new instance"""
        raise NotImplementedError
    
    async def start_instance(self, instance_id: str) -> bool:
        """Start instance"""
        raise NotImplementedError
    
    async def stop_instance(self, instance_id: str) -> bool:
        """Stop instance"""
        raise NotImplementedError
    
    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate instance"""
        raise NotImplementedError
    
    async def get_instance_status(self, instance_id: str) -> InstanceStatus:
        """Get instance status"""
        raise NotImplementedError
    
    async def upload_files(self, instance_id: str, local_files: List[str],
                          remote_directory: str) -> bool:
        """Upload files to instance"""
        raise NotImplementedError
    
    async def download_files(self, instance_id: str, remote_files: List[str],
                           local_directory: str) -> bool:
        """Download files from instance"""
        raise NotImplementedError
    
    async def execute_command(self, instance_id: str, command: str,
                            working_directory: str = "/home/ubuntu") -> Tuple[int, str, str]:
        """Execute command on instance"""
        raise NotImplementedError
    
    def estimate_cost(self, instance_type: InstanceType, 
                     processing_time_hours: float) -> float:
        """Estimate processing cost"""
        raise NotImplementedError


class AWSProvider(CloudProviderInterface):
    """AWS cloud provider implementation"""
    
    def __init__(self, credentials: Dict[str, Any]):
        super().__init__(CloudProvider.AWS, credentials)
        
        # Pricing per hour (approximate, as of 2024)
        self.pricing = {
            InstanceType.AWS_P3_XLARGE: 3.06,
            InstanceType.AWS_P3_2XLARGE: 6.12,
            InstanceType.AWS_P4D_XLARGE: 3.912,
            InstanceType.AWS_G4DN_XLARGE: 0.526
        }
        
        # Instance specifications
        self.instance_specs = {
            InstanceType.AWS_P3_XLARGE: {
                "gpu_count": 1, "gpu_type": "V100", "vram_gb": 16.0,
                "vcpus": 4, "ram_gb": 61.0, "storage_gb": 25.0
            },
            InstanceType.AWS_P3_2XLARGE: {
                "gpu_count": 1, "gpu_type": "V100", "vram_gb": 16.0,
                "vcpus": 8, "ram_gb": 122.0, "storage_gb": 25.0
            },
            InstanceType.AWS_P4D_XLARGE: {
                "gpu_count": 1, "gpu_type": "A100", "vram_gb": 40.0,
                "vcpus": 4, "ram_gb": 96.0, "storage_gb": 600.0
            },
            InstanceType.AWS_G4DN_XLARGE: {
                "gpu_count": 1, "gpu_type": "T4", "vram_gb": 16.0,
                "vcpus": 4, "ram_gb": 16.0, "storage_gb": 125.0
            }
        }
    
    async def authenticate(self) -> bool:
        """Authenticate with AWS"""
        try:
            self.ec2_client = boto3.client(
                'ec2',
                aws_access_key_id=self.credentials.get('access_key_id'),
                aws_secret_access_key=self.credentials.get('secret_access_key'),
                region_name=self.credentials.get('region', 'us-east-1')
            )
            
            self.ssm_client = boto3.client(
                'ssm',
                aws_access_key_id=self.credentials.get('access_key_id'),
                aws_secret_access_key=self.credentials.get('secret_access_key'),
                region_name=self.credentials.get('region', 'us-east-1')
            )
            
            # Test authentication
            await asyncio.get_event_loop().run_in_executor(
                None, self.ec2_client.describe_regions
            )
            
            logger.info("AWS authentication successful")
            return True
            
        except Exception as e:
            logger.error(f"AWS authentication failed: {e}")
            return False
    
    async def list_instances(self) -> List[CloudInstanceInfo]:
        """List AWS instances"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.ec2_client.describe_instances
            )
            
            instances = []
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    instance_info = self._parse_aws_instance(instance)
                    if instance_info:
                        instances.append(instance_info)
            
            return instances
            
        except Exception as e:
            logger.error(f"Error listing AWS instances: {e}")
            return []
    
    async def create_instance(self, instance_type: InstanceType, 
                            region: str, **kwargs) -> CloudInstanceInfo:
        """Create AWS instance"""
        try:
            # AMI with CUDA and ML frameworks pre-installed
            ami_id = kwargs.get('ami_id', 'ami-0c02fb55956c7d316')  # Deep Learning AMI
            key_name = kwargs.get('key_name', 'default-key')
            security_groups = kwargs.get('security_groups', ['default'])
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.ec2_client.run_instances(
                    ImageId=ami_id,
                    MinCount=1,
                    MaxCount=1,
                    InstanceType=instance_type.value,
                    KeyName=key_name,
                    SecurityGroups=security_groups,
                    TagSpecifications=[
                        {
                            'ResourceType': 'instance',
                            'Tags': [
                                {'Key': 'Name', 'Value': 'UpScale-Worker'},
                                {'Key': 'Purpose', 'Value': 'AI-Processing'}
                            ]
                        }
                    ]
                )
            )
            
            instance_data = response['Instances'][0]
            return self._parse_aws_instance(instance_data)
            
        except Exception as e:
            logger.error(f"Error creating AWS instance: {e}")
            raise
    
    def _parse_aws_instance(self, instance_data: Dict) -> Optional[CloudInstanceInfo]:
        """Parse AWS instance data"""
        try:
            instance_type_str = instance_data['InstanceType']
            instance_type = None
            
            # Find matching instance type
            for it in InstanceType:
                if it.value == instance_type_str:
                    instance_type = it
                    break
            
            if not instance_type:
                return None
            
            specs = self.instance_specs.get(instance_type, {})
            
            # Map AWS state to our status
            state_mapping = {
                'pending': InstanceStatus.STARTING,
                'running': InstanceStatus.RUNNING,
                'stopping': InstanceStatus.STOPPING,
                'stopped': InstanceStatus.STOPPED,
                'terminated': InstanceStatus.TERMINATED
            }
            
            status = state_mapping.get(instance_data['State']['Name'], InstanceStatus.ERROR)
            
            return CloudInstanceInfo(
                instance_id=instance_data['InstanceId'],
                provider=CloudProvider.AWS,
                instance_type=instance_type,
                region=instance_data['Placement']['AvailabilityZone'][:-1],
                status=status,
                public_ip=instance_data.get('PublicIpAddress'),
                private_ip=instance_data.get('PrivateIpAddress'),
                created_at=instance_data['LaunchTime'].timestamp(),
                cost_per_hour=self.pricing.get(instance_type, 0.0),
                **specs
            )
            
        except Exception as e:
            logger.error(f"Error parsing AWS instance: {e}")
            return None
    
    async def start_instance(self, instance_id: str) -> bool:
        """Start AWS instance"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.ec2_client.start_instances(InstanceIds=[instance_id])
            )
            return True
        except Exception as e:
            logger.error(f"Error starting AWS instance {instance_id}: {e}")
            return False
    
    async def stop_instance(self, instance_id: str) -> bool:
        """Stop AWS instance"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.ec2_client.stop_instances(InstanceIds=[instance_id])
            )
            return True
        except Exception as e:
            logger.error(f"Error stopping AWS instance {instance_id}: {e}")
            return False
    
    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate AWS instance"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.ec2_client.terminate_instances(InstanceIds=[instance_id])
            )
            return True
        except Exception as e:
            logger.error(f"Error terminating AWS instance {instance_id}: {e}")
            return False
    
    async def execute_command(self, instance_id: str, command: str,
                            working_directory: str = "/home/ubuntu") -> Tuple[int, str, str]:
        """Execute command on AWS instance using SSM"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.ssm_client.send_command(
                    InstanceIds=[instance_id],
                    DocumentName="AWS-RunShellScript",
                    Parameters={
                        'commands': [f'cd {working_directory} && {command}'],
                        'workingDirectory': [working_directory]
                    }
                )
            )
            
            command_id = response['Command']['CommandId']
            
            # Wait for command completion
            for _ in range(30):  # Wait up to 30 seconds
                await asyncio.sleep(1)
                
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.ssm_client.get_command_invocation(
                        CommandId=command_id,
                        InstanceId=instance_id
                    )
                )
                
                if result['Status'] in ['Success', 'Failed']:
                    return_code = 0 if result['Status'] == 'Success' else 1
                    stdout = result.get('StandardOutputContent', '')
                    stderr = result.get('StandardErrorContent', '')
                    return return_code, stdout, stderr
            
            # Timeout
            return 1, '', 'Command timeout'
            
        except Exception as e:
            logger.error(f"Error executing command on AWS instance: {e}")
            return 1, '', str(e)
    
    def estimate_cost(self, instance_type: InstanceType, 
                     processing_time_hours: float) -> float:
        """Estimate AWS processing cost"""
        hourly_rate = self.pricing.get(instance_type, 0.0)
        return hourly_rate * processing_time_hours


class CloudManager:
    """Manages cloud resources and hybrid processing"""
    
    def __init__(self):
        self.providers: Dict[CloudProvider, CloudProviderInterface] = {}
        self.active_instances: Dict[str, CloudInstanceInfo] = {}
        self.active_jobs: Dict[str, CloudJob] = {}
        
        # Configuration
        self.config_file = PATHS["temp_dir"] / "cloud_config.json"
        self.jobs_file = PATHS["temp_dir"] / "cloud_jobs.json"
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Callbacks
        self.job_callback: Optional[Callable] = None
        self.instance_callback: Optional[Callable] = None
        
        # Load configuration
        self.load_configuration()
        
        logger.info("Cloud Manager initialized")
    
    def add_provider(self, provider: CloudProvider, credentials: Dict[str, Any]) -> bool:
        """Add cloud provider"""
        try:
            if provider == CloudProvider.AWS:
                provider_instance = AWSProvider(credentials)
            else:
                logger.error(f"Unsupported provider: {provider}")
                return False
            
            self.providers[provider] = provider_instance
            logger.info(f"Added {provider.value} provider")
            return True
            
        except Exception as e:
            logger.error(f"Error adding provider {provider}: {e}")
            return False
    
    async def authenticate_providers(self) -> Dict[CloudProvider, bool]:
        """Authenticate all configured providers"""
        results = {}
        
        for provider, provider_instance in self.providers.items():
            try:
                success = await provider_instance.authenticate()
                results[provider] = success
                logger.info(f"{provider.value} authentication: {'Success' if success else 'Failed'}")
            except Exception as e:
                logger.error(f"Error authenticating {provider}: {e}")
                results[provider] = False
        
        return results
    
    async def list_all_instances(self) -> Dict[CloudProvider, List[CloudInstanceInfo]]:
        """List instances from all providers"""
        results = {}
        
        for provider, provider_instance in self.providers.items():
            try:
                instances = await provider_instance.list_instances()
                results[provider] = instances
                
                # Update active instances cache
                for instance in instances:
                    if instance.status == InstanceStatus.RUNNING:
                        self.active_instances[instance.instance_id] = instance
                        
            except Exception as e:
                logger.error(f"Error listing instances from {provider}: {e}")
                results[provider] = []
        
        return results
    
    async def create_optimal_instance(self, requirements: Dict[str, Any]) -> Optional[CloudInstanceInfo]:
        """Create optimal instance based on requirements"""
        
        min_vram = requirements.get('min_vram_gb', 8.0)
        max_cost_per_hour = requirements.get('max_cost_per_hour', 10.0)
        preferred_provider = requirements.get('preferred_provider', CloudProvider.AWS)
        preferred_region = requirements.get('preferred_region', 'us-east-1')
        
        # Find suitable instance types
        suitable_types = []
        
        if preferred_provider == CloudProvider.AWS and preferred_provider in self.providers:
            aws_provider = self.providers[preferred_provider]
            
            for instance_type in [InstanceType.AWS_G4DN_XLARGE, InstanceType.AWS_P3_XLARGE, 
                                InstanceType.AWS_P3_2XLARGE, InstanceType.AWS_P4D_XLARGE]:
                
                specs = aws_provider.instance_specs.get(instance_type, {})
                cost = aws_provider.pricing.get(instance_type, float('inf'))
                
                if specs.get('vram_gb', 0) >= min_vram and cost <= max_cost_per_hour:
                    suitable_types.append((instance_type, cost, specs))
        
        if not suitable_types:
            logger.error("No suitable instance types found")
            return None
        
        # Sort by cost (cheapest first)
        suitable_types.sort(key=lambda x: x[1])
        best_type, _, _ = suitable_types[0]
        
        # Create instance
        try:
            provider_instance = self.providers[preferred_provider]
            instance = await provider_instance.create_instance(
                best_type, preferred_region, **requirements
            )
            
            self.active_instances[instance.instance_id] = instance
            
            if self.instance_callback:
                self.instance_callback("created", instance)
            
            logger.info(f"Created instance {instance.instance_id} ({best_type.value})")
            return instance
            
        except Exception as e:
            logger.error(f"Error creating instance: {e}")
            return None
    
    async def submit_cloud_job(self, input_files: List[str], 
                             processing_params: Dict[str, Any],
                             instance_requirements: Dict[str, Any] = None) -> Optional[str]:
        """Submit job for cloud processing"""
        
        # Generate job ID
        job_id = f"job_{int(time.time() * 1000)}_{len(self.active_jobs)}"
        
        # Create or find suitable instance
        instance = None
        
        if instance_requirements:
            instance = await self.create_optimal_instance(instance_requirements)
        else:
            # Use existing instance or create default
            running_instances = [inst for inst in self.active_instances.values() 
                               if inst.status == InstanceStatus.RUNNING]
            if running_instances:
                instance = running_instances[0]  # Use first available
            else:
                # Create default instance
                default_requirements = {
                    'min_vram_gb': 8.0,
                    'max_cost_per_hour': 5.0,
                    'preferred_provider': CloudProvider.AWS
                }
                instance = await self.create_optimal_instance(default_requirements)
        
        if not instance:
            logger.error("No suitable instance available for job")
            return None
        
        # Create job
        output_directory = f"/tmp/upscale_output_{job_id}"
        
        job = CloudJob(
            job_id=job_id,
            input_files=input_files,
            output_directory=output_directory,
            processing_params=processing_params,
            instance_info=instance
        )
        
        self.active_jobs[job_id] = job
        
        # Start job processing (would be implemented)
        asyncio.create_task(self._process_cloud_job(job))
        
        if self.job_callback:
            self.job_callback("submitted", job)
        
        logger.info(f"Submitted cloud job {job_id}")
        return job_id
    
    async def _process_cloud_job(self, job: CloudJob):
        """Process job on cloud instance"""
        try:
            job.status = "starting"
            job.started_at = time.time()
            
            if self.job_callback:
                self.job_callback("started", job)
            
            # Get provider instance
            provider_instance = self.providers[job.instance_info.provider]
            
            # Ensure instance is running
            if job.instance_info.status != InstanceStatus.RUNNING:
                await provider_instance.start_instance(job.instance_info.instance_id)
                
                # Wait for instance to start
                for _ in range(60):  # Wait up to 60 seconds
                    await asyncio.sleep(1)
                    status = await provider_instance.get_instance_status(job.instance_info.instance_id)
                    if status == InstanceStatus.RUNNING:
                        break
                else:
                    raise Exception("Instance failed to start")
            
            # Upload input files
            job.status = "uploading"
            if self.job_callback:
                self.job_callback("uploading", job)
                
            upload_success = await provider_instance.upload_files(
                job.instance_info.instance_id,
                job.input_files,
                "/tmp/upscale_input"
            )
            
            if not upload_success:
                raise Exception("Failed to upload input files")
            
            # Execute processing command
            job.status = "processing"
            if self.job_callback:
                self.job_callback("processing", job)
            
            # Build processing command
            command = self._build_processing_command(job)
            
            return_code, stdout, stderr = await provider_instance.execute_command(
                job.instance_info.instance_id,
                command
            )
            
            if return_code != 0:
                raise Exception(f"Processing failed: {stderr}")
            
            # Download results
            job.status = "downloading"
            if self.job_callback:
                self.job_callback("downloading", job)
            
            download_success = await provider_instance.download_files(
                job.instance_info.instance_id,
                [job.output_directory],
                str(PATHS["output_dir"] / "cloud" / job.job_id)
            )
            
            if not download_success:
                raise Exception("Failed to download results")
            
            # Job completed
            job.status = "completed"
            job.completed_at = time.time()
            
            # Calculate cost
            if job.processing_time:
                job.actual_cost = provider_instance.estimate_cost(
                    job.instance_info.instance_type,
                    job.processing_time / 3600.0
                )
            
            if self.job_callback:
                self.job_callback("completed", job)
            
            logger.info(f"Cloud job {job.job_id} completed in {job.processing_time:.2f}s")
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = time.time()
            
            if self.job_callback:
                self.job_callback("failed", job)
            
            logger.error(f"Cloud job {job.job_id} failed: {e}")
    
    def _build_processing_command(self, job: CloudJob) -> str:
        """Build processing command for cloud execution"""
        
        params = job.processing_params
        model = params.get('model_id', 'realesrgan_x4plus')
        scale_factor = params.get('scale_factor', 2.0)
        
        # Simple command - would be more sophisticated in reality
        command = f"""
        cd /home/ubuntu/upscale_app && \\
        python -m src.enhanced_upscale_app \\
            --input /tmp/upscale_input \\
            --output {job.output_directory} \\
            --model {model} \\
            --scale {scale_factor} \\
            --batch
        """
        
        return command.strip()
    
    def get_job_status(self, job_id: str) -> Optional[CloudJob]:
        """Get job status"""
        return self.active_jobs.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel cloud job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = "cancelled"
            
            if self.job_callback:
                self.job_callback("cancelled", job)
            
            logger.info(f"Cancelled cloud job {job_id}")
            return True
        return False
    
    def estimate_job_cost(self, input_files: List[str], 
                         processing_params: Dict[str, Any],
                         instance_type: InstanceType) -> float:
        """Estimate job processing cost"""
        
        # Simple estimation based on file count and complexity
        file_count = len(input_files)
        complexity_factor = processing_params.get('complexity_factor', 1.0)
        
        # Estimate processing time (rough approximation)
        estimated_hours = (file_count * 0.1 * complexity_factor) / 60.0  # Convert minutes to hours
        
        # Find provider for instance type
        for provider_instance in self.providers.values():
            if hasattr(provider_instance, 'pricing') and instance_type in provider_instance.pricing:
                return provider_instance.estimate_cost(instance_type, estimated_hours)
        
        return 0.0
    
    def get_cloud_stats(self) -> Dict[str, Any]:
        """Get cloud processing statistics"""
        
        active_instances = sum(1 for inst in self.active_instances.values() 
                             if inst.status == InstanceStatus.RUNNING)
        
        job_stats = {
            "total": len(self.active_jobs),
            "running": sum(1 for job in self.active_jobs.values() if job.status == "processing"),
            "completed": sum(1 for job in self.active_jobs.values() if job.status == "completed"),
            "failed": sum(1 for job in self.active_jobs.values() if job.status == "failed")
        }
        
        total_cost = sum(job.actual_cost for job in self.active_jobs.values() 
                        if job.actual_cost > 0)
        
        return {
            "providers": len(self.providers),
            "active_instances": active_instances,
            "total_instances": len(self.active_instances),
            "jobs": job_stats,
            "total_cost": total_cost
        }
    
    def load_configuration(self):
        """Load cloud configuration"""
        try:
            if self.config_file.exists():
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                
                # Load provider configurations
                for provider_name, credentials in config.get("providers", {}).items():
                    provider = CloudProvider(provider_name)
                    self.add_provider(provider, credentials)
                
                logger.info("Cloud configuration loaded")
        except Exception as e:
            logger.warning(f"Error loading cloud configuration: {e}")
    
    def save_configuration(self):
        """Save cloud configuration"""
        try:
            config = {
                "providers": {},
                "version": "1.0",
                "last_updated": time.time()
            }
            
            # Note: Don't save credentials to file for security
            # This would save non-sensitive configuration only
            
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving cloud configuration: {e}")
    
    def shutdown(self):
        """Shutdown cloud manager"""
        logger.info("Shutting down Cloud Manager...")
        
        # Cancel all active jobs
        for job_id in list(self.active_jobs.keys()):
            self.cancel_job(job_id)
        
        # Save configuration
        self.save_configuration()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Cloud Manager shutdown complete")


# Global cloud manager instance
cloud_manager = CloudManager()