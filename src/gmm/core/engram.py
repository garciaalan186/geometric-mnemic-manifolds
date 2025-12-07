"""
Engram data structure - immutable memory states.

This module defines the Engram class representing a frozen phenomenological
moment in the agent's experience, as described in the paper.
"""

import pickle
from typing import Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class Engram:
    """
    Immutable memory state representing a frozen phenomenological moment.

    An Engram is the fundamental unit of memory in the GMM architecture.
    Unlike standard RAG chunks, an Engram is executable - it can be "woken up"
    as an ephemeral clone to engage in active reasoning.

    Attributes:
        id: Unique identifier for this engram
        timestamp: Creation time (Unix timestamp)
        context_window: Serialized text/token content
        embedding: Vector representation in embedding space
        metadata: Additional information (dict)
        layer: Hierarchical layer (0=raw, 1=pattern, 2=abstract)
    """
    id: int
    timestamp: float
    context_window: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    layer: int = 0  # Default to raw (Layer 0)

    def __post_init__(self):
        """Validate engram after initialization."""
        if self.layer not in {0, 1, 2}:
            raise ValueError(f"Layer must be 0, 1, or 2, got {self.layer}")

        if not isinstance(self.embedding, np.ndarray):
            raise TypeError(f"Embedding must be numpy array, got {type(self.embedding)}")

    def serialize(self) -> bytes:
        """
        Serialize engram for persistent storage.

        Returns:
            Pickled bytes representation of the engram
        """
        data = {
            'id': self.id,
            'timestamp': self.timestamp,
            'context_window': self.context_window,
            'embedding': self.embedding.tolist(),  # Convert to list for JSON-compatibility
            'metadata': self.metadata,
            'layer': self.layer
        }
        return pickle.dumps(data)

    @staticmethod
    def deserialize(data: bytes) -> 'Engram':
        """
        Deserialize engram from storage.

        Args:
            data: Pickled bytes representation

        Returns:
            Reconstructed Engram instance
        """
        obj = pickle.loads(data)
        obj['embedding'] = np.array(obj['embedding'])
        return Engram(**obj)

    @property
    def is_raw(self) -> bool:
        """Check if this is a raw (Layer 0) engram."""
        return self.layer == 0

    @property
    def is_pattern(self) -> bool:
        """Check if this is a pattern (Layer 1) engram."""
        return self.layer == 1

    @property
    def is_abstract(self) -> bool:
        """Check if this is an abstract (Layer 2) engram."""
        return self.layer == 2

    def __repr__(self) -> str:
        layer_name = {0: 'Raw', 1: 'Pattern', 2: 'Abstract'}[self.layer]
        preview = self.context_window[:50] + '...' if len(self.context_window) > 50 else self.context_window
        return f"Engram(id={self.id}, layer={layer_name}, preview='{preview}')"
