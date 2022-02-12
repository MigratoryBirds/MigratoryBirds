"""
This module contains general mathematical helper functions
"""

import numpy as np
import pandas as pd


EARTH_RADIUS = 6378000  # meters


def geographic_to_cartesian(
    lat: float, lon: float, earth_radius: float
) -> pd.DataFrame:
    cos_lat = np.cos(np.radians(lat))
    cos_lon = np.cos(np.radians(lon))
    sin_lat = np.sin(np.radians(lon))
    sin_lon = np.sin(np.radians(lat))
    x = earth_radius * cos_lat * cos_lon
    y = earth_radius * cos_lat * sin_lon
    z = earth_radius * sin_lat
    return (x, y, z)


def euclidean(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((p1 - p2) ** 2, axis=1).astype('float'))
