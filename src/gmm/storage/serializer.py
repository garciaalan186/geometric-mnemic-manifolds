"""
Persistent storage for engrams.

Handles serialization and deserialization of engrams to/from disk,
managing the storage directory and file operations.
"""

from pathlib import Path
from typing import Optional
from ..core.engram import Engram


class EngramSerializer:
    """
    Handles persistent storage of engrams.

    Manages serialization to disk using a directory-based storage system.
    Each engram is stored as a separate file for efficient access and
    modification.

    Attributes:
        storage_path: Directory path for storing engram files
    """

    DEFAULT_STORAGE_DIR = "./gmm_storage"

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the engram serializer.

        Args:
            storage_path: Path to storage directory (creates if doesn't exist)
        """
        self.storage_path = storage_path or Path(self.DEFAULT_STORAGE_DIR)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save(self, engram: Engram) -> Path:
        """
        Save an engram to disk.

        Args:
            engram: Engram instance to save

        Returns:
            Path to the saved file
        """
        filepath = self._get_filepath(engram)
        with open(filepath, 'wb') as f:
            f.write(engram.serialize())
        return filepath

    def load(self, engram_id: int, layer: int = 0) -> Optional[Engram]:
        """
        Load an engram from disk.

        Args:
            engram_id: ID of the engram to load
            layer: Layer of the engram (0, 1, or 2)

        Returns:
            Loaded Engram instance, or None if not found
        """
        filepath = self.storage_path / f"engram_{layer}_{engram_id}.pkl"

        if not filepath.exists():
            return None

        with open(filepath, 'rb') as f:
            return Engram.deserialize(f.read())

    def delete(self, engram: Engram) -> bool:
        """
        Delete an engram from disk.

        Args:
            engram: Engram instance to delete

        Returns:
            True if deleted, False if not found
        """
        filepath = self._get_filepath(engram)

        if filepath.exists():
            filepath.unlink()
            return True

        return False

    def get_storage_size(self) -> float:
        """
        Calculate total storage size in MB.

        Returns:
            Total size of all engram files in megabytes
        """
        total_bytes = sum(f.stat().st_size for f in self.storage_path.glob("*.pkl"))
        return total_bytes / (1024 * 1024)

    def clear_storage(self) -> int:
        """
        Remove all stored engrams.

        Returns:
            Number of files deleted
        """
        files = list(self.storage_path.glob("engram_*.pkl"))
        count = len(files)

        for filepath in files:
            filepath.unlink()

        return count

    def _get_filepath(self, engram: Engram) -> Path:
        """
        Generate filepath for an engram.

        Args:
            engram: Engram instance

        Returns:
            Path object for the engram file
        """
        return self.storage_path / f"engram_{engram.layer}_{engram.id}.pkl"
