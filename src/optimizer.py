import math
import random
from typing import List, Sequence, Tuple

from src.geo import haversine_m, offset_meters


def route_distance(order: Sequence[int], points: Sequence[Tuple[float, float]], depot: Tuple[float, float]) -> float:
    if not order:
        return 0.0
    total = haversine_m(depot, points[order[0]])
    for index in range(len(order) - 1):
        total += haversine_m(points[order[index]], points[order[index + 1]])
    total += haversine_m(points[order[-1]], depot)
    return total


def _ordered_crossover(parent_a: List[int], parent_b: List[int], rng: random.Random) -> List[int]:
    size = len(parent_a)
    start, end = sorted(rng.sample(range(size), 2))
    child = [-1] * size
    child[start : end + 1] = parent_a[start : end + 1]
    fill = [gene for gene in parent_b if gene not in child]
    cursor = 0
    for index in range(size):
        if child[index] == -1:
            child[index] = fill[cursor]
            cursor += 1
    return child


def _mutate(order: List[int], rng: random.Random) -> None:
    if rng.random() < 0.5 and len(order) > 3:
        a, b = sorted(rng.sample(range(len(order)), 2))
        order[a : b + 1] = reversed(order[a : b + 1])
    else:
        a, b = rng.sample(range(len(order)), 2)
        order[a], order[b] = order[b], order[a]


def optimize_route(
    points: Sequence[Tuple[float, float]],
    depot: Tuple[float, float],
    population_size: int = 80,
    generations: int = 140,
    mutation_rate: float = 0.25,
    seed: int = 7,
    cruise_speed_mps: float = 18.0,
) -> dict:
    rng = random.Random(seed)
    size = len(points)
    baseline_order = list(range(size))
    baseline_distance = route_distance(baseline_order, points, depot)

    population = [rng.sample(range(size), size) for _ in range(population_size)]
    best_order = min(population, key=lambda order: route_distance(order, points, depot))
    best_distance = route_distance(best_order, points, depot)
    convergence = []

    for _ in range(generations):
        scored = sorted(population, key=lambda order: route_distance(order, points, depot))
        next_population = [list(gene) for gene in scored[:2]]

        while len(next_population) < population_size:
            tournament = rng.sample(scored, min(3, len(scored)))
            parent_a = min(tournament, key=lambda order: route_distance(order, points, depot))
            tournament = rng.sample(scored, min(3, len(scored)))
            parent_b = min(tournament, key=lambda order: route_distance(order, points, depot))
            child = _ordered_crossover(parent_a, parent_b, rng)
            if rng.random() < mutation_rate:
                _mutate(child, rng)
            next_population.append(child)

        population = next_population
        candidate = min(population, key=lambda order: route_distance(order, points, depot))
        candidate_distance = route_distance(candidate, points, depot)
        if candidate_distance < best_distance:
            best_distance = candidate_distance
            best_order = list(candidate)
        convergence.append(round(best_distance / 1000.0, 4))

    best_order = _two_opt(best_order, points, depot)
    best_distance = route_distance(best_order, points, depot)
    improvement = (baseline_distance - best_distance) / baseline_distance * 100.0 if baseline_distance else 0.0

    return {
        "order": best_order,
        "baseline_order": baseline_order,
        "distance_m": best_distance,
        "baseline_distance_m": baseline_distance,
        "improvement_pct": improvement,
        "flight_time_min": best_distance / cruise_speed_mps / 60.0,
        "baseline_time_min": baseline_distance / cruise_speed_mps / 60.0,
        "convergence": convergence,
        "generations": generations,
        "population": population_size,
    }


def _two_opt(order: List[int], points: Sequence[Tuple[float, float]], depot: Tuple[float, float]) -> List[int]:
    improved = True
    best = list(order)
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                candidate = best[:i] + list(reversed(best[i : j + 1])) + best[j + 1 :]
                if route_distance(candidate, points, depot) + 1e-6 < route_distance(best, points, depot):
                    best = candidate
                    improved = True
    return best


def gwo_localize(
    initial_point: Tuple[float, float],
    true_point: Tuple[float, float],
    radius_m: float = 28.0,
    wolves: int = 12,
    iterations: int = 45,
    seed: int = 11,
) -> dict:
    rng = random.Random(seed)
    true_east, true_north = _local(initial_point, true_point)
    true_east, true_north = _clamp_radius(true_east, true_north, radius_m)

    def concentration(east: float, north: float) -> float:
        distance = math.hypot(east - true_east, north - true_north)
        signal = 1.0 / (1.0 + (distance / 4.0) ** 2)
        return signal + rng.gauss(0.0, 0.012)

    positions = [
        (rng.uniform(-radius_m, radius_m), rng.uniform(-radius_m, radius_m)) for _ in range(wolves)
    ]
    scores = [concentration(east, north) for east, north in positions]
    alpha_index = max(range(wolves), key=lambda index: scores[index])
    alpha_pos, alpha_score = positions[alpha_index], scores[alpha_index]
    beta_pos, beta_score = alpha_pos, alpha_score
    delta_pos, delta_score = alpha_pos, alpha_score
    convergence = []

    for iteration in range(iterations):
        a = 2.0 - 2.0 * iteration / max(iterations - 1, 1)
        for index in range(wolves):
            for leader_pos in (alpha_pos, beta_pos, delta_pos):
                r1, r2 = rng.random(), rng.random()
                a_vector = 2.0 * a * r1 - a
                c_vector = 2.0 * r2
                distance = abs(c_vector * leader_pos[0] - positions[index][0]), abs(
                    c_vector * leader_pos[1] - positions[index][1]
                )
                positions[index] = (
                    leader_pos[0] - a_vector * distance[0],
                    leader_pos[1] - a_vector * distance[1],
                )
            positions[index] = _clamp_radius(positions[index][0], positions[index][1], radius_m)
            scores[index] = concentration(*positions[index])

        ranked = sorted(range(wolves), key=lambda index: scores[index], reverse=True)
        alpha_pos, alpha_score = positions[ranked[0]], scores[ranked[0]]
        beta_pos, beta_score = positions[ranked[1]], scores[ranked[1]]
        delta_pos, delta_score = positions[ranked[2]], scores[ranked[2]]
        convergence.append(round(math.hypot(alpha_pos[0] - true_east, alpha_pos[1] - true_north), 3))

    refined = offset_meters(initial_point, alpha_pos[0], alpha_pos[1])
    return {
        "point": refined,
        "east_m": alpha_pos[0],
        "north_m": alpha_pos[1],
        "score": alpha_score,
        "radius_m": radius_m,
        "iterations": iterations,
        "wolves": wolves,
        "convergence_m": convergence,
        "error_before_m": haversine_m(initial_point, true_point),
        "error_after_m": haversine_m(refined, true_point),
    }


def _local(origin: Tuple[float, float], target: Tuple[float, float]) -> Tuple[float, float]:
    north = (target[0] - origin[0]) * 111320.0
    east = (target[1] - origin[1]) * 111320.0 * math.cos(math.radians(origin[0]))
    return east, north


def _clamp_radius(east: float, north: float, radius: float) -> Tuple[float, float]:
    distance = math.hypot(east, north)
    if distance <= radius:
        return east, north
    return east / distance * radius, north / distance * radius
