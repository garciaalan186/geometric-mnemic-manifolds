"""
Kronecker spiral positioning engine.

Implements low-discrepancy geometric positioning using Kronecker sequences
for organizing memories on a hypersphere with exponential temporal decay.
"""

import numpy as np
from .position import SpiralPosition


class KroneckerSpiral:
    """
    Generates low-discrepancy positions on a hypersphere using Kronecker sequences.

    This implements the core geometric positioning algorithm from the paper,
    utilizing square roots of primes to achieve quasi-random angular distribution
    with exponential radial expansion for temporal foveation.

    Attributes:
        dimensions: Embedding space dimensionality
        lambda_decay: Exponential decay constant for radial expansion
        primes: First N prime numbers for Kronecker basis
        alpha: Square roots of primes (linearly independent irrationals)
    """

    # Class constant: first primes for Kronecker sequence
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    def __init__(self, dimensions: int = 2, lambda_decay: float = 0.01):
        """
        Initialize the Kronecker spiral.

        Args:
            dimensions: Embedding space dimensions (default: 2 for visualization)
            lambda_decay: Exponential decay constant controlling radial expansion rate

        Raises:
            ValueError: If dimensions exceeds available primes
        """
        if dimensions > len(self.PRIMES):
            raise ValueError(
                f"Dimensions ({dimensions}) exceeds available primes ({len(self.PRIMES)})"
            )

        self.dimensions = dimensions
        self.lambda_decay = lambda_decay

        # Alpha vector: square roots of primes (linearly independent irrationals)
        self.alpha = np.array([np.sqrt(p) for p in self.PRIMES[:dimensions]])

    def position(self, k: int) -> SpiralPosition:
        """
        Calculate position of the k-th engram on the spiral.

        Uses Kronecker sequence for angular distribution and exponential
        function for radial expansion, implementing the geometric encoding
        described in the paper.

        Args:
            k: Index in the sequence (0 = most recent)

        Returns:
            SpiralPosition with all coordinate information
        """
        # Kronecker sequence for angular distribution
        # u_k = {k * alpha} where {} denotes fractional part
        u = np.array([(k * self.alpha[d]) % 1 for d in range(self.dimensions)])

        # For 2D visualization, use first component
        theta = 2 * np.pi * u[0]

        # Exponential radial expansion: r(k) = e^(λk)
        radius = np.exp(self.lambda_decay * k)

        # Cartesian coordinates
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        # Determine hierarchical layer based on k
        # Layer boundaries from paper: k_fovea=10, k_para=64
        if k < 10:
            layer = 0  # Fovea (raw memories)
        elif k < 64:
            layer = 1  # Para-fovea (pattern summaries)
        else:
            layer = 2  # Periphery (abstract concepts)

        return SpiralPosition(
            k=k,
            theta=theta,
            radius=radius,
            x=x,
            y=y,
            layer=layer
        )

    def distance(self, pos1: SpiralPosition, pos2: SpiralPosition) -> float:
        """
        Calculate geometric distance between two positions.

        Incorporates both spatial and temporal components with adaptive weighting:
        - Recent memories: spatial distance dominates
        - Distant memories: temporal distance dominates

        Args:
            pos1: First position
            pos2: Second position

        Returns:
            Weighted distance combining spatial and temporal components
        """
        # Euclidean spatial distance
        spatial = np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)

        # Temporal distance
        temporal = abs(pos1.k - pos2.k)

        # Adaptive weighting: spatial matters more for nearby, temporal for distant
        weight = np.exp(-0.01 * temporal)

        return spatial * weight + temporal * (1 - weight)

    def get_layer_boundaries(self) -> dict:
        """
        Get the k-value boundaries for each hierarchical layer.

        Returns:
            Dictionary mapping layer names to (min_k, max_k) tuples
        """
        return {
            'fovea': (0, 9),
            'para_fovea': (10, 63),
            'periphery': (64, float('inf'))
        }
