"""
Hierarchical compression for the skip-list structure.

Implements the telegraphic compression operator Λ for building
pattern and abstract layers from raw engrams.
"""

from typing import List
import numpy as np
from ..core.engram import Engram


class HierarchicalCompressor:
    """
    Builds the hierarchical skip-list structure from raw engrams.

    Implements the telegraphic compression operator Λ described in the paper,
    which minimizes token count while preserving entities and causal links.

    Attributes:
        beta1: Compression ratio for Layer 1 (patterns)
        beta2: Compression ratio for Layer 2 (abstracts)
    """

    # Class constants for compression ratios (from paper)
    DEFAULT_BETA1 = 64  # 1 pattern per 64 raw engrams
    DEFAULT_BETA2 = 16  # 1 abstract per 16 patterns

    # Function words to remove in telegraphic compression
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
    }

    def __init__(self, beta1: int = None, beta2: int = None):
        """
        Initialize the hierarchical compressor.

        Args:
            beta1: Compression ratio for Layer 1 (default: 64)
            beta2: Compression ratio for Layer 2 (default: 16)
        """
        self.beta1 = beta1 or self.DEFAULT_BETA1
        self.beta2 = beta2 or self.DEFAULT_BETA2

    def generate_seed_context(self, text: str) -> str:
        """
        Generate a compressed seed summary from a context string.
        used for daisy-chaining engrams.
        
        Args:
            text: Input text
            
        Returns:
            Compressed summary string
        """
        # Reuse telegraphic compression logic for now
        # Limit to fewer tokens for a seed (e.g. 50 chars or 10 words)
        compressed = self._telegraphic_compress([text])
        # Force strict truncation for seeds to avoid context bloat
        words = compressed.split()
        return " ".join(words[:15])

    def compress_to_layer1(self, engrams: List[Engram]) -> List[Engram]:
        """
        Compress raw engrams into pattern-level summaries.

        Groups every beta1 raw engrams into a single pattern engram using
        telegraphic compression to preserve key entities and patterns.

        Args:
            engrams: List of Layer 0 (raw) engrams

        Returns:
            List of Layer 1 (pattern) engrams
        """
        layer1_engrams = []

        for i in range(0, len(engrams), self.beta1):
            chunk = engrams[i:i + self.beta1]

            # Telegraphic compression: extract key entities and patterns
            compressed_context = self._telegraphic_compress(
                [e.context_window for e in chunk]
            )

            # Average embedding (could be replaced with learned compression)
            avg_embedding = np.mean([e.embedding for e in chunk], axis=0)

            # Create pattern engram
            pattern = Engram(
                id=i // self.beta1,
                timestamp=chunk[0].timestamp,  # Use earliest timestamp
                context_window=compressed_context,
                embedding=avg_embedding,
                metadata={
                    'children': [e.id for e in chunk],
                    'span': (chunk[0].id, chunk[-1].id),
                    'compression_ratio': self.beta1
                },
                layer=1
            )
            layer1_engrams.append(pattern)

        return layer1_engrams

    def compress_to_layer2(self, layer1_engrams: List[Engram]) -> List[Engram]:
        """
        Compress pattern engrams into abstract semantic concepts.

        Groups every beta2 pattern engrams into a single abstract engram.

        Args:
            layer1_engrams: List of Layer 1 (pattern) engrams

        Returns:
            List of Layer 2 (abstract) engrams
        """
        layer2_engrams = []

        for i in range(0, len(layer1_engrams), self.beta2):
            chunk = layer1_engrams[i:i + self.beta2]

            # Extract high-level themes
            compressed_context = self._abstract_compress(
                [e.context_window for e in chunk]
            )

            # Average embedding
            avg_embedding = np.mean([e.embedding for e in chunk], axis=0)

            # Create abstract engram
            abstract = Engram(
                id=i // self.beta2,
                timestamp=chunk[0].timestamp,
                context_window=compressed_context,
                embedding=avg_embedding,
                metadata={
                    'children': [e.id for e in chunk],
                    'patterns': len(chunk),
                    'compression_ratio': self.beta1 * self.beta2
                },
                layer=2
            )
            layer2_engrams.append(abstract)

        return layer2_engrams

    def _telegraphic_compress(self, texts: List[str]) -> str:
        """
        Telegraphic compression: remove function words, keep content.

        This is a simplified version. Production would use:
        - NER for entity extraction
        - Dependency parsing for causal links
        - Learned compression models

        Args:
            texts: List of text strings to compress

        Returns:
            Compressed text preserving entities and key content
        """
        # Combine texts
        combined = " ".join(texts)

        # Simple heuristic: keep capitalized words (entities) and important keywords
        words = combined.split()

        # Filter out stop words but keep entities (capitalized)
        compressed_words = [
            w for w in words
            if w.lower() not in self.STOP_WORDS or w[0].isupper()
        ]

        # Limit length to prevent excessive growth
        max_tokens = 200
        return " ".join(compressed_words[:max_tokens])

    def _abstract_compress(self, texts: List[str]) -> str:
        """
        Abstract compression: extract themes and concepts.

        This is highly simplified. Production would use:
        - Topic modeling (LDA, NMF)
        - Semantic clustering
        - LLM-based summarization

        Args:
            texts: List of text strings to compress

        Returns:
            Abstract representation of themes
        """
        # Combine texts
        combined = " ".join(texts)
        words = combined.split()

        # Extract unique important terms (very crude approximation)
        # In production, this would be semantic theme extraction
        unique_terms = list(set([
            w for w in words
            if len(w) > 4  # Filter short words
        ]))[:50]  # Limit to top concepts

        return " ".join(unique_terms)

    def get_layer_sizes(self, raw_count: int) -> dict:
        """
        Calculate expected layer sizes for a given raw engram count.

        Args:
            raw_count: Number of raw (Layer 0) engrams

        Returns:
            Dictionary with layer sizes
        """
        layer1_size = (raw_count + self.beta1 - 1) // self.beta1  # Ceiling division
        layer2_size = (layer1_size + self.beta2 - 1) // self.beta2

        return {
            'layer0': raw_count,
            'layer1': layer1_size,
            'layer2': layer2_size,
            'total_nodes': raw_count + layer1_size + layer2_size,
            'compression_l1': raw_count / max(layer1_size, 1),
            'compression_l2': raw_count / max(layer2_size, 1)
        }
