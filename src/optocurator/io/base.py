"""Base abstract classes for I/O readers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

from optocurator.models import StimulusProtocol, UnitData


class BaseSpikeReader(ABC):
    """Abstract base reader for electrophysiology spike sorting data."""

    @abstractmethod
    def load(
        self, folder_path: Union[str, Path], session_id: Optional[str] = None
    ) -> List[UnitData]:
        """Load spike sorted data from the given folder.

        Args:
            folder_path: Directory containing spike sorted outputs.
            session_id: Optional session identifier. If not provided, inferred from path.

        Returns:
            List of UnitData objects.
        """
        pass


class BaseStimulusReader(ABC):
    """Abstract base reader for optical stimulus timestamp files."""

    @abstractmethod
    def load(
        self, file_path: Union[str, Path], sampling_rate: float = 30000.0
    ) -> StimulusProtocol:
        """Load stimulus pulse onsets and offsets from a file.

        Args:
            file_path: Path to stimulus file (.npy or .mat).
            sampling_rate: Recording sampling rate in Hz to convert sample indices to seconds.

        Returns:
            StimulusProtocol containing pulse onsets and offsets in seconds.
        """
        pass

    @abstractmethod
    def discover(self, folder_path: Union[str, Path]) -> List[Path]:
        """Discover stimulus files within a directory.

        Args:
            folder_path: Directory to search for pulse files.

        Returns:
            List of matching Path objects.
        """
        pass
