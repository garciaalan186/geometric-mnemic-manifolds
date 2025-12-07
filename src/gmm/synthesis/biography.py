"""
Synthetic longitudinal biography generator.

Generates synthetic life histories for testing without data contamination.
Uses phonotactically neutral entities and unique physics to ensure retrieval
is from the manifold, not from pre-training memorization.
"""

from typing import List, Tuple
import numpy as np


class SyntheticBiographyGenerator:
    """
    Generates synthetic life histories for contamination-free testing.

    Creates consistent, long-horizon life logs in a universe with unique
    physical laws and phonotactically neutral entities (e.g., "Banet", "Mison").
    This ensures that successful retrieval demonstrates manifold functionality,
    not latent weight activation from pre-training.

    Attributes:
        entities: List of neutral entity names
        actions: List of possible actions
        locations: List of location names
        rng: Random number generator (for reproducibility)
    """

    # Phonotactically neutral entity names
    DEFAULT_ENTITIES = [
        "Banet", "Mison", "Toral", "Kevar", "Rilax", "Nophen",
        "Quorix", "Zeltan", "Morfin", "Plexar"
    ]

    DEFAULT_ACTIONS = [
        "acquired", "sold", "discovered", "invented", "lost",
        "traded", "modified", "damaged", "repaired", "studied"
    ]

    DEFAULT_LOCATIONS = [
        "Nexara", "Valdor", "Crimson Plains", "Azure District",
        "Obsidian Tower", "Crystal Bay"
    ]

    def __init__(self, seed: int = 42):
        """
        Initialize the biography generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.entities = self.DEFAULT_ENTITIES.copy()
        self.actions = self.DEFAULT_ACTIONS.copy()
        self.locations = self.DEFAULT_LOCATIONS.copy()

        # Set random seed for reproducibility
        self.rng = np.random.RandomState(seed)

    def generate_event(self, day: int) -> str:
        """
        Generate a single life event.

        Args:
            day: Day number in the biography

        Returns:
            Event description string
        """
        entity = self.rng.choice(self.entities)
        action = self.rng.choice(self.actions)
        location = self.rng.choice(self.locations)
        value = self.rng.randint(10, 1000)

        return f"Day {day}: I {action} the {entity} in {location} for {value} credits."

    def generate_biography(self, num_days: int = 1000) -> List[str]:
        """
        Generate a complete synthetic biography.

        Args:
            num_days: Number of days/events to generate

        Returns:
            List of event description strings
        """
        return [self.generate_event(day) for day in range(num_days)]

    def generate_passkey_dataset(
        self,
        num_events: int = 10000,
        passkey_depth: int = 5000
    ) -> Tuple[List[str], str, int]:
        """
        Generate dataset with a "needle in haystack" passkey.

        Creates a biography with a unique passkey inserted at a specific depth,
        used for testing retrieval accuracy (Needle in the Spiral benchmark).

        Args:
            num_events: Total number of events to generate
            passkey_depth: Position to insert the passkey

        Returns:
            Tuple of (events, passkey, passkey_index)
        """
        events = self.generate_biography(num_events)

        # Insert unique passkey
        passkey_code = self.rng.randint(100000, 999999)
        passkey = f"PASSKEY: The secret code is {passkey_code}"
        events[passkey_depth] = passkey

        return events, passkey, passkey_depth

    def add_custom_entity(self, entity: str) -> None:
        """Add a custom entity to the generator."""
        if entity not in self.entities:
            self.entities.append(entity)

    def add_custom_action(self, action: str) -> None:
        """Add a custom action to the generator."""
        if action not in self.actions:
            self.actions.append(action)

    def add_custom_location(self, location: str) -> None:
        """Add a custom location to the generator."""
        if location not in self.locations:
            self.locations.append(location)
