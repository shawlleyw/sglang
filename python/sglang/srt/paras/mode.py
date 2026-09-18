"""Shared execution modes for ParaS state and internal APIs."""

from enum import Enum


class ParaSMode(Enum):
    """Use members internally, values for tensor names, and names for metrics."""

    EP = "ep"
    TP = "tp"
