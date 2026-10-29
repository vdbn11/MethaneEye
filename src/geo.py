import math
from typing import Iterable, Tuple

EARTH_RADIUS_M = 6371000.0


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def destination(point: Tuple[float, float], bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    lat, lon = map(math.radians, point)
    bearing = math.radians(bearing_deg)
    angular = distance_m / EARTH_RADIUS_M
    lat2 = math.asin(math.sin(lat) * math.cos(angular) + math.cos(lat) * math.sin(angular) * math.cos(bearing))
    lon2 = lon + math.atan2(
        math.sin(bearing) * math.sin(angular) * math.cos(lat),
        math.cos(angular) - math.sin(lat) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def bearing_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def offset_meters(point: Tuple[float, float], east_m: float, north_m: float) -> Tuple[float, float]:
    lat, lon = point
    dlat = north_m / 111320.0
    dlon = east_m / (111320.0 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def local_offsets(point: Tuple[float, float], other: Tuple[float, float]) -> Tuple[float, float]:
    north = (other[0] - point[0]) * 111320.0
    east = (other[1] - point[1]) * 111320.0 * math.cos(math.radians(point[0]))
    return east, north


def path_length_m(points: Iterable[Tuple[float, float]]) -> float:
    pts = list(points)
    return sum(haversine_m(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def pixel_to_ground(
    drone: Tuple[float, float],
    altitude_m: float,
    heading_deg: float,
    image_size: Tuple[int, int],
    pixel: Tuple[float, float],
    hfov_deg: float = 40.0,
) -> Tuple[float, float]:
    width, height = image_size
    ground_width = 2.0 * altitude_m * math.tan(math.radians(hfov_deg) / 2.0)
    ground_height = ground_width * height / width

    dx = (pixel[0] - width / 2.0) / width * ground_width
    dy = (pixel[1] - height / 2.0) / height * ground_height

    heading = math.radians(heading_deg)
    east = dx * math.cos(heading) - dy * math.sin(heading)
    north = dx * math.sin(heading) + dy * math.cos(heading)
    return offset_meters(drone, east, -north)


def interpolate(a: Tuple[float, float], b: Tuple[float, float], fraction: float) -> Tuple[float, float]:
    return a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction
