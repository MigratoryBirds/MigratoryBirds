import numpy as np
import pandas as pd


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


def euclidean(p1: tuple[float], p2: tuple[float]) -> float:
    return np.sqrt(sum([(x1 - x2) ** 2 for x1, x2 in zip(p1, p2)]))
