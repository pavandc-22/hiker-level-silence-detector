import math
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

EARTH_RADIUS_KM = 6371.0
GPX_NAMESPACE = {"gpx": "http://www.topografix.com/GPX/1/1"}

MAX_GAP_SECONDS = 600
MAX_HIKING_SPEED_KMH = 12.0


def calculate_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return EARTH_RADIUS_KM * 2 * math.atan2(
        math.sqrt(a), math.sqrt(1 - a)
    )


def load_gpx(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    observations = []

    for point in root.findall(".//gpx:trkpt", GPX_NAMESPACE):
        lat = point.get("lat")
        lon = point.get("lon")

        time_element = point.find("gpx:time", GPX_NAMESPACE)
        ele_element = point.find("gpx:ele", GPX_NAMESPACE)

        if not lat or not lon or time_element is None:
            continue

        timestamp = datetime.fromisoformat(
            time_element.text.replace("Z", "+00:00")
        )

        elevation = (
            float(ele_element.text)
            if ele_element is not None and ele_element.text
            else None
        )

        observations.append({
            "latitude": float(lat),
            "longitude": float(lon),
            "elevation": elevation,
            "timestamp": timestamp
        })

    observations.sort(key=lambda p: p["timestamp"])

    for i, point in enumerate(observations):
        point["distance_km"] = 0.0
        point["speed_kmh"] = 0.0
        point["valid_interval"] = False
        point["time_gap"] = False

        if i == 0:
            continue

        previous = observations[i - 1]

        seconds = (
            point["timestamp"] - previous["timestamp"]
        ).total_seconds()

        if seconds <= 0:
            continue

        if seconds > MAX_GAP_SECONDS:
            point["time_gap"] = True
            continue

        distance = calculate_distance(
            previous["latitude"],
            previous["longitude"],
            point["latitude"],
            point["longitude"]
        )

        speed = distance / (seconds / 3600)

        if speed > MAX_HIKING_SPEED_KMH:
            continue

        point["distance_km"] = distance
        point["speed_kmh"] = speed
        point["valid_interval"] = True

    return observations


def load_gps_dataset(folder_path):
    results = {}

    for file_path in sorted(Path(folder_path).rglob("*.gpx")):
        try:
            observations = load_gpx(file_path)

            if len(observations) >= 2:
                results[file_path.name] = observations

        except (ET.ParseError, ValueError, OSError) as error:
            print(f"Skipping {file_path.name}: {error}")

    return results


def get_track_summary(observations):
    if not observations:
        return {}

    valid = [
        p for p in observations if p["valid_interval"]
    ]

    total_distance = sum(
        p["distance_km"] for p in valid
    )

    active_seconds = sum(
        (
            observations[i]["timestamp"]
            - observations[i - 1]["timestamp"]
        ).total_seconds()
        for i, p in enumerate(observations)
        if i > 0 and p["valid_interval"]
    )

    speeds = [p["speed_kmh"] for p in valid]

    elevations = [
        p["elevation"] for p in observations
        if p["elevation"] is not None
    ]

    time_gaps = sum(
        1 for p in observations if p["time_gap"]
    )

    rejected_intervals = sum(
        1 for i, p in enumerate(observations)
        if i > 0
        and not p["valid_interval"]
        and not p["time_gap"]
    )

    return {
        "gps_points": len(observations),
        "valid_intervals": len(valid),
        "time_gaps": time_gaps,
        "rejected_intervals": rejected_intervals,
        "distance_km": round(total_distance, 2),
        "active_duration_hours": round(
            active_seconds / 3600, 2
        ),
        "average_speed_kmh": round(
            total_distance / (active_seconds / 3600), 2
        ) if active_seconds > 0 else 0,
        "max_speed_kmh": round(max(speeds), 2) if speeds else 0,
        "elevation_min_m": round(min(elevations), 2) if elevations else None,
        "elevation_max_m": round(max(elevations), 2) if elevations else None
    }
