import base64
import csv
import json
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from src.config import GAS_CONFIDENCE_THRESHOLD, GAS_RATIO_THRESHOLD, REPORTS_DIR, SITE_LAT, SITE_LON
from src.geo import bearing_deg, destination, haversine_m, interpolate, pixel_to_ground
from src.model import FrameLibrary, GasDetector, encode_jpeg
from src.optimizer import gwo_localize, optimize_route


SECTOR_META = [
    {"key": "N", "letter": "N", "name": "North", "color": "#3aa0ff"},
    {"key": "E", "letter": "E", "name": "East", "color": "#2ecc71"},
    {"key": "S", "letter": "S", "name": "South", "color": "#ffb066"},
    {"key": "W", "letter": "W", "name": "West", "color": "#ff5a4d"},
]


class Drone:
    def __init__(self, meta, depot, nodes, edges, leaks):
        self.meta = meta
        self.key = meta["key"]
        self.letter = meta["letter"]
        self.name = meta["name"]
        self.color = meta["color"]
        self.depot = depot
        self.nodes = nodes
        self.edges = edges
        self.edge_lengths = [haversine_m(nodes[a], nodes[b]) for a, b in edges]
        self.waypoints = [interpolate(nodes[a], nodes[b], 0.5) for a, b in edges]
        self.leaks = leaks
        self.wp_indices = list(range(len(self.waypoints)))
        self.route = []
        self.reset()

    def reset(self):
        base = getattr(self, "base_route", None)
        self.route = list(base) if base else list(range(len(self.waypoints)))
        self.position = self.depot
        self.heading = 0.0
        self.battery = 100.0
        self.phase = "idle"
        self.launched = False
        self.hold = False
        self.current_leg = 0
        self.flown_m = 0.0
        self.coverage = set()
        self.scans = []
        self.pending = False
        self.capture_id = 0
        self.camera_image = None
        self.camera_result = None
        self.camera_size = None
        self.camera_pose = None
        self.inspect_timer = 0.0
        self.trail = []
        self.last_trail_t = -1e9
        self.low_battery_logged = False
        self.verify_waypoints = []
        for leak in self.leaks:
            leak["detected"] = False
            leak["detection"] = None


class SimulationManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.detector = GasDetector()
        self.frames = FrameLibrary()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.running = False
        self.speed = 2.0
        self.time_scale = 8.0
        self.mode = "ga"
        self.cruise_speed = 18.0
        self.inspect_duration = 5.0
        self.detect_radius = 38.0
        self.auto_mode = True
        self.repair_duration = 60.0
        self.crew_speed = 20.0
        self._verification_started = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self._build_scenario()
        for drone in self.drones:
            drone.wp_indices = list(range(len(drone.waypoints)))
        self._optimize()
        self._apply_routes()
        self.reset(silent=True)

    def _build_scenario(self):
        self.depot = (25.9400, 49.6500)

        # 4 independent pipeline networks
        net_defs = {
            "N": {
                "nodes": [
                    (25.9550, 49.6460),
                    (25.9700, 49.6520),
                    (25.9850, 49.6580),
                    (25.9990, 49.6640),
                    (25.9550, 49.6640),
                    (25.9700, 49.6700),
                    (25.9850, 49.6760),
                    (25.9990, 49.6820),
                    (25.9900, 49.6880),
                ],
                "edges": [
                    (0, 1), (1, 2), (2, 3),
                    (4, 5), (5, 6), (6, 7),
                    (0, 4), (2, 6), (6, 8),
                ],
                "leaks": [
                    {"edge": 1, "offset": 7.0, "severity_kg_h": 650.0, "frame_index": 0},
                    {"edge": 6, "offset": -5.0, "severity_kg_h": 280.0, "frame_index": 2},
                    {"edge": 8, "offset": 6.0, "severity_kg_h": 340.0, "frame_index": 4},
                ]
            },
            "E": {
                "nodes": [
                    (25.9380, 49.6800),
                    (25.9430, 49.6970),
                    (25.9480, 49.7140),
                    (25.9530, 49.7310),
                    (25.9500, 49.6800),
                    (25.9550, 49.6970),
                    (25.9600, 49.7140),
                    (25.9650, 49.7310),
                    (25.9420, 49.7220),
                ],
                "edges": [
                    (0, 1), (1, 2), (2, 3),
                    (4, 5), (5, 6), (6, 7),
                    (0, 4), (2, 6), (5, 8),
                ],
                "leaks": [
                    {"edge": 2, "offset": -6.0, "severity_kg_h": 420.0, "frame_index": 6},
                    {"edge": 5, "offset": 5.0, "severity_kg_h": 310.0, "frame_index": 8},
                    {"edge": 7, "offset": -4.0, "severity_kg_h": 230.0, "frame_index": 3},
                ]
            },
            "S": {
                "nodes": [
                    (25.9280, 49.6460),
                    (25.9150, 49.6520),
                    (25.9020, 49.6580),
                    (25.8890, 49.6640),
                    (25.9280, 49.6660),
                    (25.9150, 49.6720),
                    (25.9020, 49.6780),
                    (25.8890, 49.6840),
                    (25.8960, 49.6900),
                ],
                "edges": [
                    (0, 1), (1, 2), (2, 3),
                    (4, 5), (5, 6), (6, 7),
                    (0, 4), (2, 6), (6, 8),
                ],
                "leaks": [
                    {"edge": 1, "offset": 8.0, "severity_kg_h": 540.0, "frame_index": 1},
                    {"edge": 4, "offset": -6.0, "severity_kg_h": 190.0, "frame_index": 5},
                    {"edge": 8, "offset": 5.0, "severity_kg_h": 300.0, "frame_index": 7},
                ]
            },
            "W": {
                "nodes": [
                    (25.9380, 49.6220),
                    (25.9430, 49.6040),
                    (25.9480, 49.5860),
                    (25.9530, 49.5680),
                    (25.9500, 49.6220),
                    (25.9550, 49.6040),
                    (25.9600, 49.5860),
                    (25.9650, 49.5680),
                    (25.9370, 49.5800),
                ],
                "edges": [
                    (0, 1), (1, 2), (2, 3),
                    (4, 5), (5, 6), (6, 7),
                    (0, 4), (2, 6), (5, 8),
                ],
                "leaks": [
                    {"edge": 1, "offset": 6.0, "severity_kg_h": 380.0, "frame_index": 0},
                    {"edge": 5, "offset": -5.0, "severity_kg_h": 240.0, "frame_index": 2},
                    {"edge": 8, "offset": 4.0, "severity_kg_h": 210.0, "frame_index": 4},
                ]
            },
        }
        self._net_defs = net_defs

        self.drones = []
        self.leaks = []
        leak_counter = 1

        for meta in SECTOR_META:
            def_item = net_defs[meta["key"]]
            nodes = def_item["nodes"]
            edges = def_item["edges"]
            drone_leaks = []
            for lspec in def_item["leaks"]:
                e_idx = lspec["edge"]
                a, b = edges[e_idx]
                midpoint = interpolate(nodes[a], nodes[b], 0.5)
                heading = bearing_deg(nodes[a], nodes[b])
                true_point = destination(midpoint, (heading + 90.0) % 360.0, lspec["offset"])
                leak_obj = {
                    "id": leak_counter,
                    "drone": meta["key"],
                    "edge": e_idx,
                    "lat": true_point[0],
                    "lon": true_point[1],
                    "severity_kg_h": lspec["severity_kg_h"],
                    "frame_index": lspec["frame_index"],
                    "detected": False,
                    "detection": None,
                    "status": "idle",
                    "detected_at": None,
                    "repaired_at": None,
                    "verified_at": None,
                }
                drone_leaks.append(leak_obj)
                self.leaks.append(leak_obj)
                leak_counter += 1

            drone = Drone(meta, self.depot, nodes, edges, drone_leaks)
            self.drones.append(drone)

        # Pipeline GeoJSON for map display
        self.pipeline_geojson = []
        for drone in self.drones:
            for index, (a, b) in enumerate(drone.edges):
                self.pipeline_geojson.append({
                    "drone": drone.key,
                    "color": drone.color,
                    "from": list(drone.nodes[a]),
                    "to": list(drone.nodes[b]),
                    "length_m": drone.edge_lengths[index],
                })
        self.total_pipeline_m = sum(d_len for drone in self.drones for d_len in drone.edge_lengths)

        self.crew_bases = [
            {"id": "Crew Alpha", "lat": self.depot[0], "lon": self.depot[1]},
            {"id": "Crew Bravo", "lat": self.depot[0] + 0.02, "lon": self.depot[1] + 0.02},
        ]

    def _optimize(self, seed=7, generations=140, mutation_rate=0.25):
        self.ga_fleet = {}
        for offset, drone in enumerate(self.drones):
            result = optimize_route(
                drone.waypoints,
                self.depot,
                population_size=80,
                generations=generations,
                mutation_rate=mutation_rate,
                seed=seed + offset * 17,
                cruise_speed_mps=self.cruise_speed,
            )
            self.ga_fleet[drone.key] = result
        self.ga = self._aggregate_ga()

    def _aggregate_ga(self):
        distance = sum(result["distance_m"] for result in self.ga_fleet.values())
        baseline = sum(result["baseline_distance_m"] for result in self.ga_fleet.values())
        sectors = []
        for drone in self.drones:
            result = self.ga_fleet[drone.key]
            sectors.append(
                {
                    "key": drone.key,
                    "letter": drone.letter,
                    "name": drone.name,
                    "color": drone.color,
                    "distance_km": result["distance_m"] / 1000.0,
                    "baseline_distance_km": result["baseline_distance_m"] / 1000.0,
                    "improvement_pct": result["improvement_pct"],
                    "flight_time_min": result["flight_time_min"],
                    "waypoints": len(drone.waypoints),
                    "convergence": result["convergence"],
                }
            )
        first = self.ga_fleet[self.drones[0].key]
        return {
            "distance_m": distance,
            "baseline_distance_m": baseline,
            "improvement_pct": (baseline - distance) / baseline * 100.0 if baseline else 0.0,
            "flight_time_min": sum(result["flight_time_min"] for result in self.ga_fleet.values()),
            "baseline_time_min": sum(result["baseline_time_min"] for result in self.ga_fleet.values()),
            "convergence": first["convergence"],
            "generations": first["generations"],
            "population": first["population"],
            "sectors": sectors,
        }

    def _apply_routes(self):
        for drone in self.drones:
            result = self.ga_fleet[drone.key]
            order = result["order"] if self.mode == "ga" else result["baseline_order"]
            drone.route = [drone.wp_indices[position] for position in order]
            drone.base_route = list(drone.route)

    @property
    def phase(self):
        launched = [drone for drone in self.drones if drone.launched]
        if not launched:
            return "idle"
        phases = [drone.phase for drone in launched]
        if all(phase == "done" for phase in phases):
            return "done"
        for candidate in ("transit", "inspect", "return"):
            if candidate in phases:
                return candidate
        return "idle"

    @property
    def route(self):
        primary = next((drone for drone in self.drones if drone.launched), self.drones[0])
        return primary.route

    def drone_by_key(self, key):
        wanted = str(key or "").strip().upper()
        for drone in self.drones:
            if drone.key == wanted or drone.letter == wanted:
                return drone
        return None

    def reset(self, silent: bool = False):
        with self.lock:
            self.running = False
            self.sim_time = 0.0
            self.detections = []
            self.false_alarms = 0
            self.work_orders = []
            self.logs = []
            self.crew_state = [
                {**base, "next_free_min": 0.0, "current_lat": base["lat"], "current_lon": base["lon"]}
                for base in self.crew_bases
            ]
            self._persisted = False
            self._frame_cache = {}
            self._verification_started = False
            self._verify_warned = False
            self._covered_ever = {drone.key: set() for drone in self.drones}
            lane = 0.0026
            self.vehicles = []
            for index, base in enumerate(self.crew_bases):
                side = -1 if index == 0 else 1
                home_lat = self.depot[0] - 0.0007 if side < 0 else self.depot[0] + 0.0007
                home_lon = self.depot[1] + side * lane
                self.vehicles.append(
                    {
                        "id": "Unit-%d" % (index + 1),
                        "name": base["id"],
                        "lat": home_lat,
                        "lon": home_lon,
                        "home_lat": home_lat,
                        "home_lon": home_lon,
                        "status": "idle",
                        "leak_id": None,
                        "timer": 0.0,
                    }
                )
            frames_dir = REPORTS_DIR / "frames"
            if frames_dir.exists():
                for item in frames_dir.glob("*.jpg"):
                    try:
                        item.unlink()
                    except OSError:
                        pass
            for index, drone in enumerate(self.drones):
                drone.reset()
                drone.camera_image = self.frames.clear_frame(index)
                drone.camera_result = self._zero_result()
                drone.camera_size = drone.camera_image.size
                drone.camera_pose = {"lat": drone.depot[0], "lon": drone.depot[1], "alt": 25.0, "heading": 0.0}
                drone.heading = bearing_deg(self.depot, drone.waypoints[drone.route[0]]) if drone.route else 0.0
            for leak in self.leaks:
                leak["detected"] = False
                leak["detection"] = None
                leak["status"] = "idle"
                leak["detected_at"] = None
                leak["repaired_at"] = None
                leak["verified_at"] = None
            if not silent:
                self._log("Mission reset. Fleet standing by at depot.")
            total_wps = sum(len(d.waypoints) for d in self.drones)
            self._log("Fleet ready: 4 drones (N/E/S/W) with separate pipeline networks (%d total waypoints)." % total_wps)
            self._log("Optimizer ready: GA per sector. GWO source refinement enabled.")

    def start(self):
        with self.lock:
            if self.phase == "done":
                if self._has_open_work():
                    for drone in self.drones:
                        if drone.launched:
                            drone.hold = False
                    self.running = True
                    self._dispatch_queued()
                    self._log("Resuming pending repair/verification work.")
                    return
                self.reset(silent=True)
            if self.phase == "idle":
                self.launch_all()
            else:
                for drone in self.drones:
                    if drone.launched:
                        drone.hold = False
                self.running = True
                self._log("Simulation resumed.")

    def _has_open_work(self):
        if any(leak["status"] in ("open", "queued", "enroute", "repairing") for leak in self.leaks):
            return True
        if any(unit["status"] != "idle" for unit in self.vehicles):
            return True
        if any(leak["status"] == "repaired" for leak in self.leaks):
            return True
        return False

    def launch_all(self):
        with self.lock:
            launched = 0
            for drone in self.drones:
                if drone.phase == "idle":
                    drone.launched = True
                    drone.hold = False
                    drone.phase = "transit"
                    drone.current_leg = 0
                    launched += 1
            if launched:
                for drone in self.drones:
                    drone.hold = False
                self.running = True
                self._log("Fleet launched: %d drones airborne | route=%s | speed=%.1fx" % (
                    launched,
                    self.mode.upper(),
                    self.speed,
                ))
            elif any(drone.launched for drone in self.drones):
                for drone in self.drones:
                    drone.hold = False
                self.running = True
                self._log("Fleet relaunched.")

    def launch(self, key):
        with self.lock:
            drone = self.drone_by_key(key)
            if drone is None:
                self._log("Unknown drone '%s'. Use N, E, S or W." % key)
                return False
            if drone.phase == "done":
                self._log("[%s] already complete - reset the mission to relaunch." % drone.letter)
                return False
            drone.hold = False
            if drone.phase == "idle":
                drone.launched = True
                drone.phase = "transit"
                drone.current_leg = 0
                self._log("[%s] %s drone launched | %d waypoints." % (
                    drone.letter, drone.name, len(drone.waypoints)))
            else:
                drone.launched = True
                self._log("[%s] resumed." % drone.letter)
            self.running = True
            return True

    def hold(self, key):
        with self.lock:
            drone = self.drone_by_key(key)
            if drone is None:
                self._log("Unknown drone '%s'. Use N, E, S or W." % key)
                return None
            if not drone.launched:
                self._log("[%s] is still on the pad - launch it first." % drone.letter)
                return None
            drone.hold = not drone.hold
            self._log("[%s] hold %s." % (drone.letter, "ON" if drone.hold else "OFF"))
            active = [item for item in self.drones if item.launched and item.phase != "done"]
            if active:
                self.running = any(not item.hold for item in active)
            return drone.hold

    def pause(self):
        with self.lock:
            self.running = False
            self._log("Simulation paused.")

    def set_speed(self, speed: float):
        with self.lock:
            self.speed = max(0.5, min(8.0, float(speed)))

    def set_mode(self, mode: str):
        with self.lock:
            if mode not in ("ga", "baseline"):
                return
            if any(drone.launched for drone in self.drones):
                self._log("Route mode can only be changed before launch or after reset.")
                return
            self.mode = mode
            self._apply_routes()
            for drone in self.drones:
                drone.current_leg = 0
            self._log("Route mode set to %s." % mode.upper())

    def rerun_ga(self):
        with self.lock:
            if any(drone.launched for drone in self.drones):
                self._log("Reset the mission before re-optimizing routes.")
                return self.ga
            seed = random.randint(1, 99999)
            self._optimize(seed=seed, generations=160, mutation_rate=0.3)
            self._apply_routes()
            self._log("GA re-optimized (4 separate networks): %.2f km vs baseline %.2f km (%.1f%% shorter)." % (
                self.ga["distance_m"] / 1000.0,
                self.ga["baseline_distance_m"] / 1000.0,
                self.ga["improvement_pct"],
            ))
            return self.ga

    def _zero_result(self):
        return {
            "mask": None,
            "ratio": 0.0,
            "confidence": 0.0,
            "centroid": (240, 148),
            "box": None,
            "inference_ms": 0.0,
            "model": self.detector.checkpoint.name,
        }

    def _loop(self):
        last = time.perf_counter()
        while True:
            time.sleep(0.05)
            now = time.perf_counter()
            dt_real = min(1.0, now - last)
            last = now
            with self.lock:
                if self.running:
                    self.tick(dt_real)

    def tick(self, dt_real: float):
        dt = dt_real * self.time_scale * self.speed
        for drone in self.drones:
            if not drone.launched or drone.hold or drone.phase in ("idle", "done"):
                continue
            self._tick_drone(drone, dt)
        for drone in self.drones:
            if hasattr(self, "_covered_ever"):
                self._covered_ever[drone.key] |= drone.coverage
        self._tick_vehicles(dt)
        self.sim_time += dt

        if self.auto_mode:
            self._auto_dispatch()
            self._auto_verify()

        launched = [drone for drone in self.drones if drone.launched]
        drones_done = bool(launched) and all(drone.phase == "done" for drone in launched)
        vehicles_busy = any(unit["status"] != "idle" for unit in self.vehicles)
        pending = any(leak["status"] in ("open", "enroute", "repairing", "queued") for leak in self.leaks)
        if drones_done and not vehicles_busy and not pending:
            if self.running:
                self.running = False
                repaired = sum(1 for leak in self.leaks if leak["status"] in ("repaired", "closed"))
                closed = sum(1 for leak in self.leaks if leak["status"] == "closed")
                self._log("Mission complete | coverage %.1f%% | %d leaks found | %d repaired | %d verified | %d camera readings." % (
                    self._coverage_pct(),
                    len(self.detections),
                    repaired,
                    closed,
                    sum(len(drone.scans) for drone in self.drones),
                ))
            if not self._persisted:
                self.persist_run()

    def _tick_drone(self, drone, dt):
        if drone.phase == "transit":
            if not drone.route:
                drone.phase = "done"
                return
            target = drone.waypoints[drone.route[drone.current_leg]]
            self._move_towards(drone, target, dt)
            drone.battery -= dt * 0.030
            distance = haversine_m(drone.position, target)
            if distance <= 2.0 or distance <= self.cruise_speed * dt:
                drone.position = target
                drone.phase = "inspect"
                drone.inspect_timer = 0.0
                self._start_inspection(drone, drone.route[drone.current_leg])
        elif drone.phase == "inspect":
            drone.inspect_timer += dt
            drone.battery -= dt * 0.020
            if drone.inspect_timer >= self.inspect_duration:
                waypoint_index = drone.route[drone.current_leg]
                drone.coverage.add(waypoint_index)
                self._complete_verification(drone, waypoint_index)
                drone.current_leg += 1
                if drone.current_leg >= len(drone.route):
                    drone.phase = "return"
                    if drone.verify_waypoints:
                        self._log("[%s] verification pass complete. Returning to depot." % drone.letter)
                    else:
                        self._log("[%s] sector network scan complete. Returning to depot." % drone.letter)
                    drone.verify_waypoints = []
                else:
                    drone.phase = "transit"
        elif drone.phase == "return":
            self._move_towards(drone, self.depot, dt)
            drone.battery -= dt * 0.030
            if haversine_m(drone.position, self.depot) <= 2.0:
                drone.position = self.depot
                drone.phase = "done"
                self._log("[%s] landed at depot | coverage %.1f%% | battery %.0f%% | %d readings." % (
                    drone.letter,
                    self._drone_coverage_pct(drone),
                    drone.battery,
                    len(drone.scans),
                ))

        drone.battery = max(0.0, min(100.0, drone.battery))
        if drone.phase in ("transit", "inspect") and drone.battery < 12.0 and not drone.low_battery_logged:
            drone.phase = "return"
            drone.low_battery_logged = True
            self._log("[%s] Low battery: aborting route and returning to depot." % drone.letter)

        if drone.phase in ("transit", "inspect", "return") and self.sim_time - drone.last_trail_t >= 1.0:
            drone.trail.append([round(drone.position[0], 6), round(drone.position[1], 6)])
            drone.last_trail_t = self.sim_time
            if len(drone.trail) > 800:
                del drone.trail[:-800]

    def _move_towards(self, drone, target, dt):
        distance = haversine_m(drone.position, target)
        if distance < 0.5:
            return
        step = min(distance, self.cruise_speed * dt)
        drone.heading = bearing_deg(drone.position, target)
        drone.position = destination(drone.position, drone.heading, step)
        drone.flown_m += step

    def set_automation(self, enabled):
        with self.lock:
            self.auto_mode = bool(enabled)
            self._log("Automation set to %s." % ("AUTO" if self.auto_mode else "MANUAL"))

    def _open_leaks(self):
        return [leak for leak in self.leaks if leak["status"] == "open"]

    def _open_work(self):
        return [leak for leak in self.leaks if leak["status"] in ("queued", "open")]

    def _order_for(self, leak_id):
        return next((order for order in self.work_orders if order["leak_id"] == leak_id), None)

    def _idle_vehicle(self):
        return next((unit for unit in self.vehicles if unit["status"] == "idle"), None)

    def dispatch_leak(self, leak_id):
        with self.lock:
            leak = next((item for item in self.leaks if item["id"] == int(leak_id)), None)
            if leak is None:
                return {"ok": False, "queued": False, "message": "unknown leak"}
            if leak["status"] == "idle":
                return {"ok": False, "queued": False, "message": "leak not detected yet"}
            if leak["status"] not in ("open", "queued"):
                return {"ok": False, "queued": False, "message": "already in progress (%s)" % leak["status"]}
            unit = self._idle_vehicle()
            if unit is None:
                if leak["status"] != "queued":
                    leak["status"] = "queued"
                    order = self._order_for(leak["id"])
                    if order:
                        order["status"] = "Queued"
                    self._log("All repair units busy - leak #%d queued and will be dispatched next." % leak["id"])
                return {"ok": True, "queued": True, "message": "queued"}
            self._assign_unit(unit, leak)
            return {"ok": True, "queued": False, "message": "%s dispatched" % unit["name"]}

    def _assign_unit(self, unit, leak):
        unit["status"] = "enroute"
        unit["leak_id"] = leak["id"]
        leak["status"] = "enroute"
        order = self._order_for(leak["id"])
        if order:
            order["status"] = "En route"
            order["dispatched_s"] = round(self.sim_time, 1)
        self.running = True
        distance_km = haversine_m((unit["lat"], unit["lon"]), (leak["lat"], leak["lon"])) / 1000.0
        self._log("%s dispatched to leak #%d | %.1f km away." % (unit["name"], leak["id"], distance_km))

    def _tick_vehicles(self, dt):
        for unit in self.vehicles:
            if unit["status"] == "enroute":
                leak = next((item for item in self.leaks if item["id"] == unit["leak_id"]), None)
                if leak is None:
                    unit["status"] = "idle"
                    unit["leak_id"] = None
                    continue
                target = (leak["lat"], leak["lon"])
                position = (unit["lat"], unit["lon"])
                distance = haversine_m(position, target)
                step = self.crew_speed * dt
                if distance <= max(20.0, step):
                    unit["lat"], unit["lon"] = target
                    unit["status"] = "repairing"
                    unit["timer"] = 0.0
                    if leak["status"] == "enroute":
                        leak["status"] = "repairing"
                    order = self._order_for(leak["id"])
                    if order:
                        order["status"] = "Repairing"
                    self._log("%s on site at leak #%d - crew working." % (unit["name"], leak["id"]))
                else:
                    heading = bearing_deg(position, target)
                    unit["lat"], unit["lon"] = destination(position, heading, step)
            elif unit["status"] == "repairing":
                unit["timer"] += dt
                if unit["timer"] >= self.repair_duration:
                    leak = next((item for item in self.leaks if item["id"] == unit["leak_id"]), None)
                    if leak is not None and leak["status"] == "repairing":
                        leak["status"] = "repaired"
                        leak["repaired_at"] = round(self.sim_time, 1)
                        order = self._order_for(leak["id"])
                        if order:
                            order["status"] = "Repaired"
                            order["repaired_s"] = round(self.sim_time, 1)
                        detected_at = leak["detected_at"] or leak["repaired_at"]
                        self._log("%s sealed leak #%d | response %.1f min." % (
                            unit["name"], leak["id"], (leak["repaired_at"] - detected_at) / 60.0))
                    unit["status"] = "returning" if not self._open_work() else "idle"
                    unit["leak_id"] = None
                    self._dispatch_queued()
            elif unit["status"] == "returning":
                home = (unit["home_lat"], unit["home_lon"])
                position = (unit["lat"], unit["lon"])
                distance = haversine_m(position, home)
                step = self.crew_speed * dt
                if distance <= max(12.0, step):
                    unit["lat"], unit["lon"] = home
                    unit["status"] = "idle"
                else:
                    heading = bearing_deg(position, home)
                    unit["lat"], unit["lon"] = destination(position, heading, step)

    def _auto_dispatch(self):
        while True:
            pending = [leak for leak in self.leaks if leak["status"] in ("queued", "open")]
            if not pending:
                return
            unit = self._idle_vehicle()
            if unit is None:
                return
            pending.sort(key=lambda item: (0 if item["status"] == "queued" else 1, item["id"]))
            self._assign_unit(unit, pending[0])

    def _dispatch_queued(self):
        while True:
            queued = [leak for leak in self.leaks if leak["status"] == "queued"]
            unit = self._idle_vehicle()
            if not queued or unit is None:
                return
            self._assign_unit(unit, sorted(queued, key=lambda item: item["id"])[0])

    def _auto_verify(self):
        detected = [leak for leak in self.leaks if leak["detected"]]
        if not detected:
            return
        if any(leak["status"] not in ("repaired", "closed") for leak in detected):
            return
        if not any(leak["status"] == "repaired" for leak in detected):
            return
        busy = [drone for drone in self.drones if drone.launched and drone.phase != "done"]
        if busy:
            return
        if self._coverage_pct() < 99.5:
            if not self._verify_warned:
                self._verify_warned = True
                self._log("Verification postponed: inspection coverage is %.1f%% (needs a full scan)." % self._coverage_pct())
            return
        self.start_verification()

    def start_verification(self):
        with self.lock:
            busy = [drone for drone in self.drones if drone.launched and drone.phase not in ("done", "idle")]
            if busy:
                self._log("Wait for the inspection flight to finish before verifying.")
                return False
            pending = [leak for leak in self.leaks if leak["status"] == "repaired"]
            if not pending:
                self._log("No repaired leaks waiting for verification.")
                return False
            coverage_now = self._coverage_pct()
            if coverage_now < 99.5:
                self._log("Cannot verify: inspection coverage is %.1f%% - finish the full scan first." % coverage_now)
                return False
            self._verification_started = True
            self._persisted = False
            for leak in pending:
                leak["status"] = "verify"
                order = self._order_for(leak["id"])
                if order:
                    order["status"] = "Verifying"
                drone = self.drone_by_key(leak["drone"])
                if drone is not None:
                    drone.verify_waypoints.append((leak["edge"], leak["id"]))
            for drone in self.drones:
                if not drone.verify_waypoints:
                    continue
                drone.launched = True
                drone.hold = False
                drone.phase = "transit"
                drone.current_leg = 0
                drone.route = [item[0] for item in drone.verify_waypoints]
                drone.battery = max(drone.battery, 40.0)
                drone.low_battery_logged = False
            self.running = True
            self._log("Verification pass launched: %d leaks to re-scan." % len(pending))
            return True

    def _complete_verification(self, drone, waypoint_index):
        for item in list(drone.verify_waypoints):
            if item[0] != waypoint_index:
                continue
            drone.verify_waypoints.remove(item)
            leak = next((entry for entry in self.leaks if entry["id"] == item[1]), None)
            if leak is None:
                continue
            leak["status"] = "closed"
            leak["verified_at"] = round(self.sim_time, 1)
            order = self._order_for(leak["id"])
            if order:
                order["status"] = "Closed"
                order["verified_s"] = round(self.sim_time, 1)
            self._log("[%s] verification: leak #%d confirmed sealed." % (drone.letter, leak["id"]))

    def _start_inspection(self, drone, waypoint_index):
        verifying = any(item[0] == waypoint_index for item in drone.verify_waypoints)
        nearest = None
        nearest_distance = self.detect_radius
        if not verifying:
            for leak in drone.leaks:
                if leak["detected"]:
                    continue
                distance = haversine_m(drone.waypoints[waypoint_index], (leak["lat"], leak["lon"]))
                if distance <= nearest_distance:
                    nearest = leak
                    nearest_distance = distance

        if nearest:
            image = self.frames.gas_frame(nearest["frame_index"])
            label = "CANDIDATE PLUME"
        elif verifying:
            image = self.frames.clear_frame(waypoint_index)
            label = "VERIFICATION PASS"
        else:
            image = self.frames.clear_frame(waypoint_index)
            label = "ROUTINE SCAN"

        drone.camera_image = image
        drone.camera_result = self._zero_result()
        drone.camera_size = image.size
        drone.camera_pose = {
            "lat": drone.position[0],
            "lon": drone.position[1],
            "alt": 25.0,
            "heading": drone.heading,
        }
        drone.capture_id += 1
        capture_id = drone.capture_id
        pose = dict(drone.camera_pose)

        # Generate frame data for scan storage so it can be viewed later when clicking purple points
        telemetry = {
            "lat": drone.position[0],
            "lon": drone.position[1],
            "alt": 25.0,
            "heading": drone.heading,
            "battery": drone.battery,
            "mode": "PLUME TRACK" if nearest else "SCAN",
        }
        annotated_preview = self.detector.annotate(image, drone.camera_result, telemetry)
        preview_data = encode_jpeg(annotated_preview)
        frame_url = self._save_frame(drone, capture_id, preview_data, version=1)

        scan = {
            "id": "%s#%03d" % (drone.key, capture_id),
            "drone": drone.key,
            "letter": drone.letter,
            "color": drone.color,
            "capture_id": capture_id,
            "t": round(self.sim_time, 1),
            "lat": round(drone.position[0], 6),
            "lon": round(drone.position[1], 6),
            "alt": 25.0,
            "heading": round(drone.heading, 1),
            "waypoint": waypoint_index + 1,
            "label": label,
            "leak_id": nearest["id"] if nearest else None,
            "ratio": None,
            "confidence": None,
            "inference_ms": None,
            "detected": None,
            "frame_url": frame_url,
        }
        drone.scans.append(scan)

        self._log(
            "[%s] Inspection %d/%d at %.6f, %.6f | %s" % (
                drone.letter,
                len(drone.coverage) + 1,
                len(drone.route),
                drone.position[0],
                drone.position[1],
                label,
            )
        )
        if nearest is None:
            self._log("[%s] Stage-1 anomaly gate: clear frame, no segmentation request." % drone.letter)
            return
        if not drone.pending:
            drone.pending = True
            self.executor.submit(
                self._infer_and_store, drone.key, image.copy(), pose, nearest, capture_id, scan
            )

    def _infer_and_store(self, drone_key, image, pose, leak, capture_id, scan):
        try:
            result = self.detector.predict(image)
        except Exception as error:
            result = self._zero_result()
            result["error"] = str(error)
        with self.lock:
            drone = self.drone_by_key(drone_key)
            if drone is None:
                return
            drone.pending = False
            if capture_id == drone.capture_id:
                drone.camera_result = result
                self._handle_result(drone, result, pose, leak, capture_id, scan)

    def _handle_result(self, drone, result, pose, leak, capture_id, scan):
        ratio = result["ratio"]
        confidence = result["confidence"]
        threshold = GAS_RATIO_THRESHOLD
        if scan is not None:
            scan["ratio"] = round(ratio, 5)
            scan["confidence"] = round(confidence, 4)
            scan["inference_ms"] = round(result["inference_ms"], 2)
            # update preview frame on disk with the detection mask overlay
            telemetry = {
                "lat": drone.position[0],
                "lon": drone.position[1],
                "alt": 25.0,
                "heading": drone.heading,
                "battery": drone.battery,
                "mode": "PLUME TRACK" if ratio >= threshold else "SCAN",
            }
            annotated = self.detector.annotate(drone.camera_image, result, telemetry)
            scan["frame_url"] = self._save_frame(drone, capture_id, encode_jpeg(annotated), version=2)

        if ratio >= threshold and confidence >= GAS_CONFIDENCE_THRESHOLD:
            ground_point = pixel_to_ground(
                (pose["lat"], pose["lon"]),
                pose["alt"],
                pose["heading"],
                drone.camera_size,
                result["centroid"],
            )
            jitter_lat = random.gauss(0.0, 0.9)
            jitter_lon = random.gauss(0.0, 0.9)
            detected_point = (
                ground_point[0] + jitter_lat / 111320.0,
                ground_point[1] + jitter_lon / (111320.0 * math.cos(math.radians(ground_point[0]))),
            )
            severity_estimate = max(15.0, min(900.0, ratio * 3000.0 + 20.0))
            detection = {
                "id": len(self.detections) + 1,
                "leak_id": leak["id"] if leak else None,
                "drone": drone.key,
                "letter": drone.letter,
                "drone_name": drone.name,
                "lat": detected_point[0],
                "lon": detected_point[1],
                "confidence": confidence,
                "ratio": ratio,
                "severity_kg_h": severity_estimate,
                "inference_ms": result["inference_ms"],
                "time_s": self.sim_time,
                "status": "detected",
                "refined": None,
                "error_m": None,
                "frame_url": scan.get("frame_url") if scan else None,
            }
            if leak:
                leak["detected"] = True
                leak["detection"] = detection
                leak["status"] = "open"
                leak["detected_at"] = round(self.sim_time, 1)
                error = haversine_m(detected_point, (leak["lat"], leak["lon"]))
                detection["error_m"] = error
                gwo = gwo_localize(detected_point, (leak["lat"], leak["lon"]))
                detection["refined"] = {
                    "lat": gwo["point"][0],
                    "lon": gwo["point"][1],
                    "error_m": gwo["error_after_m"],
                    "error_before_m": gwo["error_before_m"],
                    "convergence_m": gwo["convergence_m"],
                }
                if scan is not None:
                    scan["detected"] = True
                self._log(
                    "[%s] LEAK #%d detected | %.1f kg/h est. | GPS error %.1f m | confidence %.0f%%" % (
                        drone.letter,
                        leak["id"],
                        severity_estimate,
                        error,
                        confidence * 100,
                    )
                )
                self._log(
                    "[%s] GWO refined source: %.1f m -> %.1f m error." % (
                        drone.letter, gwo["error_before_m"], gwo["error_after_m"])
                )
                self._create_work_order(detection, leak, drone)
            else:
                if scan is not None:
                    scan["detected"] = False
                self.false_alarms += 1
                self._log("[%s] Unverified plume candidate flagged; no asset match. Logged for review." % drone.letter)
            self.detections.append(detection)
        elif scan is not None:
            scan["detected"] = False

    def _create_work_order(self, detection, leak, drone):
        best_crew = None
        best_eta = float("inf")
        for crew in self.crew_state:
            distance_km = haversine_m((crew["current_lat"], crew["current_lon"]), (detection["refined"]["lat"], detection["refined"]["lon"])) / 1000.0
            eta = crew["next_free_min"] + distance_km / 50.0 * 60.0 + 45.0
            if eta < best_eta:
                best_eta = eta
                best_crew = crew
        severity = detection["severity_kg_h"]
        priority = "P1" if severity >= 400 else "P2" if severity >= 150 else "P3"
        best_crew["next_free_min"] = best_eta
        best_crew["current_lat"] = detection["refined"]["lat"]
        best_crew["current_lon"] = detection["refined"]["lon"]
        order = {
            "id": "WO-%03d" % (len(self.work_orders) + 1),
            "leak_id": leak["id"],
            "drone": drone.letter,
            "lat": detection["refined"]["lat"],
            "lon": detection["refined"]["lon"],
            "severity_kg_h": severity,
            "priority": priority,
            "crew": best_crew["id"],
            "eta_min": best_eta,
            "status": "Assigned",
            "created_s": round(self.sim_time, 1),
            "dispatched_s": None,
            "repaired_s": None,
            "verified_s": None,
        }
        self.work_orders.append(order)
        self._log("%s queued for leak #%d (found by %s drone) | %s | ETA %.0f min." % (
            best_crew["id"], leak["id"], drone.letter, priority, best_eta))
        if self.auto_mode:
            self._auto_dispatch()

    def _avg_response_min(self):
        responses = [
            leak["repaired_at"] - leak["detected_at"]
            for leak in self.leaks
            if leak["repaired_at"] and leak["detected_at"]
        ]
        if not responses:
            return 0.0
        return sum(responses) / len(responses) / 60.0

    def _covered_count(self):
        total = 0
        for drone in self.drones:
            ever = self._covered_ever.get(drone.key, set()) if hasattr(self, "_covered_ever") else set()
            total += len(drone.coverage | ever)
        return total

    def _coverage_pct(self):
        total_wps = sum(len(drone.waypoints) for drone in self.drones)
        covered_wps = self._covered_count()
        return covered_wps / total_wps * 100.0 if total_wps else 0.0

    def _drone_coverage_pct(self, drone):
        if not drone.waypoints:
            return 0.0
        return len(drone.coverage) / len(drone.waypoints) * 100.0

    def _log(self, message):
        self.logs.append({"t": round(self.sim_time, 1), "message": message})
        self.logs = self.logs[-60:]

    def _save_frame(self, drone, capture_id, data_url, version=1):
        try:
            frames_dir = REPORTS_DIR / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            name = "%s_%03d.jpg" % (drone.key, capture_id)
            payload = data_url.split(",", 1)[1] if "," in data_url else data_url
            (frames_dir / name).write_bytes(base64.b64decode(payload))
            return "/frames/%s?v=%d" % (name, version)
        except Exception as error:
            self._log("Frame save failed: %s" % error)
            return None

    def _camera_frame(self, drone):
        telemetry = {
            "lat": drone.position[0],
            "lon": drone.position[1],
            "alt": 25.0,
            "heading": drone.heading,
            "battery": drone.battery,
            "mode": "PLUME TRACK" if drone.camera_result["ratio"] >= GAS_RATIO_THRESHOLD else "SCAN",
        }
        cache_key = (
            drone.capture_id,
            round(drone.battery, 1),
            round(drone.heading, 1),
            round(drone.position[0], 6),
            round(drone.position[1], 6),
            round(drone.camera_result["ratio"], 5),
        )
        cached = self._frame_cache.get(drone.key)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        annotated = self.detector.annotate(drone.camera_image, drone.camera_result, telemetry)
        frame_data = encode_jpeg(annotated)
        self._frame_cache[drone.key] = (cache_key, frame_data)
        return frame_data

    def _drone_payload(self, drone):
        route_points = [[self.depot[0], self.depot[1]]]
        for index in drone.route:
            point = drone.waypoints[index]
            route_points.append([point[0], point[1]])
        route_points.append([self.depot[0], self.depot[1]])
        ga_result = self.ga_fleet[drone.key]
        return {
            "key": drone.key,
            "letter": drone.letter,
            "name": drone.name,
            "color": drone.color,
            "phase": drone.phase,
            "launched": drone.launched,
            "hold": drone.hold,
            "verifying": bool(drone.verify_waypoints),
            "lat": drone.position[0],
            "lon": drone.position[1],
            "alt": 25.0,
            "heading": drone.heading,
            "battery": drone.battery,
            "speed_mps": self.cruise_speed,
            "current_leg": drone.current_leg,
            "total_legs": len(drone.route),
            "distance_km": drone.flown_m / 1000.0,
            "coverage_pct": self._drone_coverage_pct(drone),
            "scans": len(drone.scans),
            "detections": sum(1 for detection in self.detections if detection.get("drone") == drone.key),
            "scan_points": drone.scans,
            "trail": drone.trail,
            "route": route_points,
            "camera": {
                "data": self._camera_frame(drone),
                "capture_id": drone.capture_id,
                "ratio": drone.camera_result["ratio"],
                "confidence": drone.camera_result["confidence"],
                "inference_ms": drone.camera_result["inference_ms"],
                "model": drone.camera_result["model"],
                "threshold": GAS_RATIO_THRESHOLD,
                "pose": drone.camera_pose,
            },
            "ga": {
                "distance_km": ga_result["distance_m"] / 1000.0,
                "baseline_distance_km": ga_result["baseline_distance_m"] / 1000.0,
                "improvement_pct": ga_result["improvement_pct"],
                "flight_time_min": ga_result["flight_time_min"],
                "waypoints": len(drone.waypoints),
            },
        }

    def snapshot(self):
        with self.lock:
            drones = [self._drone_payload(drone) for drone in self.drones]
            primary = next((payload for payload in drones if payload["launched"] and payload["phase"] != "done"),
                           drones[0])

            detected = [detection for detection in self.detections if detection["leak_id"] is not None]
            avg_error = sum(d["error_m"] for d in detected if d["error_m"] is not None) / len(detected) if detected else 0.0
            refined_errors = [d["refined"]["error_m"] for d in detected if d.get("refined")]
            avg_refined = sum(refined_errors) / len(refined_errors) if refined_errors else 0.0
            total_emissions = sum(d["severity_kg_h"] for d in detected)
            airborne = [payload for payload in drones if payload["launched"] and payload["phase"] != "done"]
            batteries = [payload["battery"] for payload in drones]
            repaired_leaks = [leak for leak in self.leaks if leak["status"] in ("repaired", "closed", "verify")]
            closed_leaks = [leak for leak in self.leaks if leak["status"] == "closed"]
            pending_leaks = [leak for leak in self.leaks if leak["status"] in ("open", "enroute", "repairing", "queued")]
            responses = [
                leak["repaired_at"] - leak["detected_at"]
                for leak in self.leaks
                if leak["repaired_at"] and leak["detected_at"]
            ]
            avoided_kg = sum(
                leak["severity_kg_h"] * max(0.0, self.sim_time - leak["repaired_at"]) / 3600.0
                for leak in self.leaks
                if leak["repaired_at"]
            )

            waypoints_all = []
            for drone in self.drones:
                for index, point in enumerate(drone.waypoints):
                    waypoints_all.append({
                        "drone": drone.key,
                        "index": index,
                        "lat": point[0],
                        "lon": point[1],
                        "covered": index in drone.coverage,
                        "risk": max([leak["severity_kg_h"] for leak in drone.leaks if leak["edge"] == index] or [0]),
                    })

            return {
                "running": self.running,
                "phase": self.phase,
                "mode": self.mode,
                "speed": self.speed,
                "sim_time": self.sim_time,
                "auto_mode": self.auto_mode,
                "drone": primary,
                "drones": drones,
                "vehicles": [
                    {
                        "id": unit["id"],
                        "name": unit["name"],
                        "lat": unit["lat"],
                        "lon": unit["lon"],
                        "status": unit["status"],
                        "leak_id": unit["leak_id"],
                    }
                    for unit in self.vehicles
                ],
                "pipeline": {
                    "nodes": [list(node) for drone in self.drones for node in drone.nodes],
                    "edges": self.pipeline_geojson,
                    "total_km": self.total_pipeline_m / 1000.0,
                },
                "waypoints": waypoints_all,
                "route": primary["route"],
                "leaks": [
                    {
                        "id": leak["id"],
                        "drone": leak["drone"],
                        "lat": leak["lat"],
                        "lon": leak["lon"],
                        "severity_kg_h": leak["severity_kg_h"],
                        "detected": leak["detected"],
                        "status": leak["status"],
                        "detected_at": leak["detected_at"],
                        "repaired_at": leak["repaired_at"],
                        "verified_at": leak["verified_at"],
                        "detection_id": leak["detection"]["id"] if leak["detection"] else None,
                    }
                    for leak in self.leaks
                ],
                "detections": self.detections,
                "work_orders": self.work_orders,
                "camera": primary["camera"],
                "stats": {
                    "coverage_pct": self._coverage_pct(),
                    "points_covered": self._covered_count(),
                    "points_total": sum(len(drone.waypoints) for drone in self.drones),
                    "distance_km": sum(drone.flown_m for drone in self.drones) / 1000.0,
                    "flight_time_min": self.sim_time / 60.0,
                    "detections": len(detected),
                    "false_alarms": self.false_alarms,
                    "avg_error_m": avg_error,
                    "avg_refined_error_m": avg_refined,
                    "emissions_kg_h": total_emissions,
                    "leaks_total": len(self.leaks),
                    "battery": sum(batteries) / len(batteries) if batteries else 100.0,
                    "scans": sum(len(drone.scans) for drone in self.drones),
                    "airborne": len(airborne),
                    "launched": sum(1 for payload in drones if payload["launched"]),
                    "repaired": len(repaired_leaks),
                    "closed": len(closed_leaks),
                    "pending": len(pending_leaks),
                    "avg_response_min": (sum(responses) / len(responses) / 60.0) if responses else 0.0,
                    "avoided_kg": avoided_kg,
                },
                "ga": {
                    "distance_km": self.ga["distance_m"] / 1000.0,
                    "baseline_distance_km": self.ga["baseline_distance_m"] / 1000.0,
                    "improvement_pct": self.ga["improvement_pct"],
                    "flight_time_min": self.ga["flight_time_min"],
                    "baseline_time_min": self.ga["baseline_time_min"],
                    "convergence": self.ga["convergence"],
                    "generations": self.ga["generations"],
                    "population": self.ga["population"],
                    "sectors": self.ga["sectors"],
                },
                "model_info": self._model_info(),
                "logs": self.logs[-18:],
            }

    def persist_run(self):
        try:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            readings = sorted(
                [scan for drone in self.drones for scan in drone.scans],
                key=lambda scan: (scan["t"], scan["drone"]),
            )
            detected = [detection for detection in self.detections if detection["leak_id"] is not None]
            refined_errors = [d["refined"]["error_m"] for d in detected if d.get("refined")]
            summary = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "phase": self.phase,
                "sim_time_s": round(self.sim_time, 1),
                "stats": {
                    "coverage_pct": round(self._coverage_pct(), 2),
                    "distance_km": round(sum(drone.flown_m for drone in self.drones) / 1000.0, 3),
                    "flight_time_min": round(self.sim_time / 60.0, 2),
                    "detections": len(detected),
                    "false_alarms": self.false_alarms,
                    "avg_refined_error_m": round(
                        sum(refined_errors) / len(refined_errors), 2) if refined_errors else 0.0,
                    "emissions_kg_h": round(sum(d["severity_kg_h"] for d in detected), 1),
                    "leaks_total": len(self.leaks),
                    "camera_readings": len(readings),
                    "leaks_detected": sum(1 for leak in self.leaks if leak["detected"]),
                    "leaks_repaired": sum(1 for leak in self.leaks if leak["status"] in ("repaired", "closed", "verify")),
                    "leaks_closed": sum(1 for leak in self.leaks if leak["status"] == "closed"),
                    "leaks_pending": sum(1 for leak in self.leaks if leak["status"] in ("open", "enroute", "repairing", "queued")),
                    "avg_response_min": round(self._avg_response_min(), 2),
                    "avoided_kg": round(
                        sum(
                            leak["severity_kg_h"] * max(0.0, self.sim_time - leak["repaired_at"]) / 3600.0
                            for leak in self.leaks
                            if leak["repaired_at"]
                        ), 1),
                },
                "ga": self.ga,
                "model": self._model_info(),
                "leaks": [
                    {
                        "id": leak["id"],
                        "drone": leak["drone"],
                        "severity_kg_h": leak["severity_kg_h"],
                        "status": leak["status"],
                        "detected_at": leak["detected_at"],
                        "repaired_at": leak["repaired_at"],
                        "verified_at": leak["verified_at"],
                    }
                    for leak in self.leaks
                ],
                "drones": [
                    {
                        "key": drone.key,
                        "letter": drone.letter,
                        "name": drone.name,
                        "color": drone.color,
                        "sector_waypoints": len(drone.waypoints),
                        "route_mode": self.mode,
                        "distance_km": round(drone.flown_m / 1000.0, 3),
                        "sector_coverage_pct": round(self._drone_coverage_pct(drone), 1),
                        "camera_readings": len(drone.scans),
                        "detections": sum(1 for detection in self.detections if detection.get("drone") == drone.key),
                        "battery_pct": round(drone.battery, 1),
                        "ga": {
                            "distance_km": round(self.ga_fleet[drone.key]["distance_m"] / 1000.0, 3),
                            "baseline_distance_km": round(
                                self.ga_fleet[drone.key]["baseline_distance_m"] / 1000.0, 3),
                            "improvement_pct": round(self.ga_fleet[drone.key]["improvement_pct"], 2),
                        },
                    }
                    for drone in self.drones
                ],
                "detections": self.detections,
                "work_orders": self.work_orders,
                "logs": self.logs,
            }
            (REPORTS_DIR / "mission_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            scan_log = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "total": len(readings),
                "readings": readings,
            }
            (REPORTS_DIR / "scan_log.json").write_text(
                json.dumps(scan_log, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self._log("Mission data exported: reports/mission_summary.json, scan_log.json.")
        except Exception as error:
            self._log("Export failed: %s" % error)

    def _model_info(self):
        metrics = self.detector.metrics or {}
        test = metrics.get("test", {})
        return {
            "checkpoint": self.detector.checkpoint.name,
            "test_dice": test.get("dice_gas"),
            "test_iou": test.get("iou_gas"),
            "test_pixel_acc": test.get("pixel_acc"),
            "train_images": metrics.get("train_images"),
            "test_images": metrics.get("test_images"),
            "image_size": metrics.get("image_size", [384, 256]),
            "classes": ["background", "gas_day", "gas_night"],
        }
