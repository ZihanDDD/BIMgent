from typing import Any, Dict

from BIMgent.utils.singleton import Singleton


class LocalMemory(metaclass=Singleton):
    """Process-wide key/value working area shared by every pipeline stage.

    Stages write intermediate results (floorplan paths, floorplan metadata,
    planner output, current task, ...) with :meth:`update_info_history` and
    read them back through :attr:`working_area`.
    """

    def __init__(self) -> None:
        self.working_area: Dict[str, Any] = {}

    def update_info_history(self, data: Dict[str, Any]) -> None:
        self.working_area.update(data)

    def clear(self) -> None:
        """Clear all memory state for a fresh run."""
        self.working_area = {}
