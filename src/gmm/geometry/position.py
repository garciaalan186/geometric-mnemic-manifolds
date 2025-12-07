"""
Geometric position data structure for the Mnemic Manifold.

This module defines the SpiralPosition dataclass representing a point
on the Kronecker spiral in the geometric memory space.
"""

from dataclasses import dataclass


@dataclass
class SpiralPosition:
    """
    Position of an engram on the geometric manifold.

    Attributes:
        k: Index in the temporal sequence (0 = most recent)
        theta: Angular position in radians
        radius: Radial distance from origin (temporal decay)
        x: Cartesian x-coordinate
        y: Cartesian y-coordinate
        layer: Hierarchical layer (0=fovea, 1=para-fovea, 2=periphery)
    """
    k: int
    theta: float
    radius: float
    x: float
    y: float
    layer: int

    def __repr__(self) -> str:
        return (f"SpiralPosition(k={self.k}, layer={self.layer}, "
                f"r={self.radius:.3f}, θ={self.theta:.3f})")
