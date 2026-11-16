"""
Hybrid real-GPS + simulated-communication dataset generator.

REAL:
- GPS coordinates
- GPS timestamps
- Elevation
- Recorded route movement speed

SIMULATED:
- Hiker check-in events
- RF link quality
- Communication scenarios

No real emergency labels or individual hiker trajectories are inferred.
"""

import argparse
import random
from bisect import bisect_left
from datetime import timedelta
from pathlib import Path

import pandas as pd

from gps_processor import load_gpx


HIKERS = ["H1", "H2", "H3", "H4"]

SCENARIOS = [
    "shared_coverage_gap",
    "individual_communication_anomaly",
    "group_co_silence",
]

CHECKIN_INTERVAL_MINUTES = 5
GPS_MATCH_TOLERANCE_SECONDS = 150


def nearest_observation(observations, timestamps, scheduled_time):
    """Find the nearest recorded GPS point without interpolation."""

    index = bisect_left(timestamps, scheduled_time)

    candidates = []

    if index < len(observations):
        candidates.append(observations[index])

    if index > 0:
        candidates.append(observations[index - 1])

    if not candidates:
        return None, None

    nearest = min(
        candidates,
        key=lambda point: abs(
            (point["timestamp"] - scheduled_time).total_seconds()
        )
    )

    difference = abs(
        (nearest["timestamp"] - scheduled_time).total_seconds()
    )

    if difference > GPS_MATCH_TOLERANCE_SECONDS:
        return None, difference

    return nearest, difference


def generate_hybrid_dataset(
    gpx_path,
    scenario,
    seed=42,
    output_dir="data/hybrid"
):
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    observations = load_gpx(gpx_path)

    if len(observations) < 2:
        raise ValueError("GPX file does not contain enough GPS observations.")

    timestamps = [point["timestamp"] for point in observations]

    start_time = timestamps[0]
    end_time = timestamps[-1]

    scheduled_times = []
    current_time = start_time

    while current_time <= end_time:
        scheduled_times.append(current_time)
        current_time += timedelta(minutes=CHECKIN_INTERVAL_MINUTES)

    if len(scheduled_times) < 3:
        raise ValueError("Track is too short for scheduled event generation.")

    rng = random.Random(seed)

    window_start = int(len(scheduled_times) * 0.35)
    window_end = max(
        window_start + 1,
        int(len(scheduled_times) * 0.65)
    )

    rows = []

    for event_index, scheduled_time in enumerate(scheduled_times):

        gps_point, match_difference = nearest_observation(
            observations,
            timestamps,
            scheduled_time
        )

        in_scenario_window = (
            window_start <= event_index < window_end
        )

        for hiker_id in HIKERS:

            # Communication events are simulated independently
            # of the recorded GPS movement.
            checkin_received = True

            if in_scenario_window:
                if scenario in (
                    "shared_coverage_gap",
                    "group_co_silence"
                ):
                    checkin_received = False

                elif (
                    scenario == "individual_communication_anomaly"
                    and hiker_id == "H4"
                ):
                    checkin_received = False

            if scenario == "shared_coverage_gap" and in_scenario_window:
                link_quality = round(rng.uniform(0.10, 0.25), 3)
            elif scenario == "group_co_silence" and in_scenario_window:
                link_quality = round(rng.uniform(0.65, 0.85), 3)
            else:
                link_quality = round(rng.uniform(0.75, 0.95), 3)

            gps_available = gps_point is not None

            rows.append({
                "scheduled_timestamp": scheduled_time,
                "hiker_id": hiker_id,
                "group_id": "SIMULATED-GROUP-01",

                # Real GPS route context; shared by the simulated group.
                "gps_timestamp": (
                    gps_point["timestamp"] if gps_available else None
                ),
                "latitude": (
                    gps_point["latitude"] if gps_available else None
                ),
                "longitude": (
                    gps_point["longitude"] if gps_available else None
                ),
                "elevation_m": (
                    gps_point["elevation"] if gps_available else None
                ),
                "group_route_speed_kmh": (
                    gps_point["speed_kmh"]
                    if gps_available and gps_point["valid_interval"]
                    else None
                ),
                "gps_valid_interval": (
                    gps_point["valid_interval"] if gps_available else False
                ),
                "gps_time_gap": (
                    gps_point["time_gap"] if gps_available else False
                ),
                "gps_match_available": gps_available,
                "gps_match_difference_seconds": match_difference,

                # Simulated communication fields.
                "simulated_checkin_received": checkin_received,
                "simulated_link_quality": link_quality,
                "simulation_scenario": scenario,
                "simulation_window_active": in_scenario_window,

                # Data provenance.
                "gps_source_type": "real_recorded_gpx",
                "communication_source_type": "simulated",
                "movement_source_type": "shared_real_route_context",
                "simulation_seed": seed,
            })

    dataframe = pd.DataFrame(rows)

    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_folder
        / f"{Path(gpx_path).stem}_{scenario}.csv"
    )

    dataframe.to_csv(output_path, index=False)

    print("\nHYBRID DATASET GENERATED")
    print("-----------------------")
    print(f"GPX source: {Path(gpx_path).name}")
    print(f"Scenario: {scenario}")
    print(f"Scheduled events: {len(scheduled_times)}")
    print(f"Simulated hikers: {len(HIKERS)}")
    print(f"Total rows: {len(dataframe)}")
    print(f"GPS matches: {dataframe['gps_match_available'].sum()}")
    print(
        "Simulated missed check-ins:",
        int((~dataframe["simulated_checkin_received"]).sum())
    )
    print(f"Output: {output_path}")

    return dataframe, output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a real-GPS and simulated-communication dataset."
    )

    parser.add_argument(
        "--track",
        default="10799.gpx",
        help="GPX filename inside the real GPS dataset folder."
    )

    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default="shared_coverage_gap"
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    gps_folder = (
        Path(__file__).resolve().parent
        / "data"
        / "real_gps"
        / "tracks-BalearicIslands-Spain"
    )

    gpx_path = gps_folder / args.track

    if not gpx_path.exists():
        raise FileNotFoundError(f"GPX file not found: {gpx_path}")

    generate_hybrid_dataset(
        gpx_path=gpx_path,
        scenario=args.scenario,
        seed=args.seed
    )
