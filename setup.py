#!/usr/bin/env python3
"""
Setup script for UpScale App
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text(encoding="utf-8").splitlines()
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

setup(
    name="upscale-app",
    version="0.1.0",
    description="AI-powered video upscaling tool using Stable Diffusion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="UpScaleApp Team",
    author_email="team@upscaleapp.com",
    url="https://github.com/SumihisaYutani/UpScaleAppProject",
    packages=find_packages(),
    package_dir={"": "src"},
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "upscale=main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    keywords="video upscaling ai stable-diffusion machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/SumihisaYutani/UpScaleAppProject/issues",
        "Source": "https://github.com/SumihisaYutani/UpScaleAppProject",
    },
)