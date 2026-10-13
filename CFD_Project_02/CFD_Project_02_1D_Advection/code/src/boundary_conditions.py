import numpy as np


def inlet_temperature(t: float) -> float:
    """Prescribed inlet temperature: T(0,t) = sin(2*pi*t)."""
    return float(np.sin(2.0 * np.pi * t))