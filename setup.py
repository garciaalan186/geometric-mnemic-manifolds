#!/usr/bin/env python3
"""
Setup script for Geometric Mnemic Manifolds package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="geometric-mnemic-manifolds",
    version="1.0.0",
    author="Alan Garcia",
    author_email="your.email@example.com",
    description="A Foveated Architecture for Autonoetic Memory in LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/garciaalan186/geometric-mnemic-manifolds",
    project_urls={
        "Bug Tracker": "https://github.com/garciaalan186/geometric-mnemic-manifolds/issues",
        "Documentation": "https://github.com/garciaalan186/geometric-mnemic-manifolds#readme",
        "Source Code": "https://github.com/garciaalan186/geometric-mnemic-manifolds",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "viz": [
            "matplotlib>=3.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gmm-prototype=examples.run_prototype:main",
            "gmm-benchmark=benchmarks.run_benchmarks:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "artificial intelligence",
        "memory systems",
        "geometric topology",
        "autonoetic memory",
        "vector databases",
        "RAG systems",
        "Kronecker sequences",
        "hierarchical compression",
        "machine learning",
        "LLMs",
    ],
)
