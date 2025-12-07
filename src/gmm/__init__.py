"""
Geometric Mnemic Manifolds - A Foveated Architecture for Autonoetic Memory in LLMs

This package implements a novel memory architecture that uses:
- Kronecker sequences for low-discrepancy geometric positioning
- Hierarchical skip-list compression (3 layers)
- Foveated retrieval with O(1) analytical addressing
- Telegraphic compression for pattern extraction
"""

from .core import Engram, GeometricMnemicManifold
from .geometry import SpiralPosition, KroneckerSpiral
from .hierarchy import HierarchicalCompressor
from .storage import EngramSerializer
from .query import ManifoldRetriever
from .synthesis import SyntheticBiographyGenerator

__version__ = '1.0.0'

__all__ = [
    'Engram',
    'GeometricMnemicManifold',
    'SpiralPosition',
    'KroneckerSpiral',
    'HierarchicalCompressor',
    'EngramSerializer',
    'ManifoldRetriever',
    'SyntheticBiographyGenerator'
]
