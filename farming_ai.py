import math
import random

import numpy as np

from settings import WORLD_HEIGHT, WORLD_WIDTH

FARMING_MODEL_VERSION = 4
FARMING_MODEL_NAME = f"diep_farming_ai_model_v{FARMING_MODEL_VERSION}"

TARGET_SCAN_RANGE = 1500.0
FIRE_RANGE = 850.0
IDEAL_MIN_DISTANCE = 240.0
IDEAL_MAX_DISTANCE = 620.0
DANGER_DISTANCE = 150.0
WALL_PROBE_RANGE = 420.0
EXPLORE_GOAL_RANGE = 1800.0
EXPLORE_GOAL_REACHED_DISTANCE = 180.0

FARMING_OBSERVATION_SIZE = 24
FARMING_ACTION_COUNT = 9

STAT_UPGRADE_PRIORITY = (5, 6, 4, 3, 7, 1, 0, 2)
TANK_UPGRADE_PRIORITY = (
    "Twin",
    "Machine Gun",
    "Sniper",
    "Flank Guard",
    "Triple Shot",
    "Gunner",
    "Destroyer",
    "Assassin",
    "Hunter",
    "Quad Tank",
    "Tri-Angle",
    "Twin Flank",
)

MOVE_DIRECTIONS = (
    (0, 0),
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (1, 1),
)

WALL_RAY_DIRECTIONS = (
    (1.0, 0.0),
    (0.7071, 0.7071),
    (0.0, 1.0),
    (-0.7071, 0.7071),
    (-1.0, 0.0),
    (-0.7071, -0.7071),
    (0.0, -1.0),
    (0.7071, -0.7071),
)


def clamp(value, low, high):
    return max(low, min(high, value))


def angle_delta(angle_a, angle_b):
    return abs((angle_a - angle_b + math.pi) % (math.pi * 2) - math.pi)


def line_of_sight(world, tank, target):
    start = (int(tank.x), int(tank.y))
    end = (int(target.x), int(target.y))
    for wall in world.walls:
        if wall.rect.clipline(start, end):
            return False
    return True


def find_farming_target(world, tank):
    nearest = None
    nearest_dist = float("inf")
    visible_best = None
    visible_best_dist = float("inf")

    for shape in world.shapes:
        if not shape.alive:
            continue

        distance = math.hypot(shape.x - tank.x, shape.y - tank.y)
        if distance < nearest_dist:
            nearest = shape
            nearest_dist = distance

        if distance <= TARGET_SCAN_RANGE and distance < visible_best_dist and line_of_sight(world, tank, shape):
            visible_best = shape
            visible_best_dist = distance

    if visible_best is not None:
        return visible_best, visible_best_dist, True

    return nearest, nearest_dist, False


def wall_probe_distance(world, tank, dir_x, dir_y):
    max_dist = WALL_PROBE_RANGE

    if dir_x > 0:
        max_dist = min(max_dist, (WORLD_WIDTH - tank.radius - tank.x) / dir_x)
    elif dir_x < 0:
        max_dist = min(max_dist, (tank.x - tank.radius) / -dir_x)

    if dir_y > 0:
        max_dist = min(max_dist, (WORLD_HEIGHT - tank.radius - tank.y) / dir_y)
    elif dir_y < 0:
        max_dist = min(max_dist, (tank.y - tank.radius) / -dir_y)

    max_dist = clamp(max_dist, 0.0, WALL_PROBE_RANGE)
    start = (int(tank.x), int(tank.y))
    end = (int(tank.x + dir_x * WALL_PROBE_RANGE), int(tank.y + dir_y * WALL_PROBE_RANGE))

    for wall in world.walls:
        clipped = wall.rect.clipline(start, end)
        if not clipped:
            continue

        hit_dist = min(math.hypot(point[0] - tank.x, point[1] - tank.y) for point in clipped)
        max_dist = min(max_dist, hit_dist)

    return clamp(max_dist, 0.0, WALL_PROBE_RANGE)


def _position_is_safe(world, x, y, radius):
    if x < radius or x > WORLD_WIDTH - radius or y < radius or y > WORLD_HEIGHT - radius:
        return False

    for wall in world.walls:
        if _circle_rect_collide(x, y, radius + 18, wall.rect):
            return False
    return True


def _circle_rect_collide(cx, cy, radius, rect):
    closest_x = max(rect.left, min(cx, rect.right))
    closest_y = max(rect.top, min(cy, rect.bottom))
    return (cx - closest_x) ** 2 + (cy - closest_y) ** 2 < radius ** 2


def _goal_score(world, tank, x, y):
    distance = math.hypot(x - tank.x, y - tank.y)
    if distance < 450:
        return -999.0

    clearances = [
        wall_probe_distance_at_point(world, x, y, tank.radius, dir_x, dir_y)
        for dir_x, dir_y in WALL_RAY_DIRECTIONS
    ]
    clearance_score = min(clearances) / WALL_PROBE_RANGE
    distance_score = clamp(distance / EXPLORE_GOAL_RANGE, 0.0, 1.0)
    return clearance_score * 1.8 + distance_score * 0.7 + random.random() * 0.15


def wall_probe_distance_at_point(world, x, y, radius, dir_x, dir_y):
    probe = type("Probe", (), {"x": x, "y": y, "radius": radius})()
    return wall_probe_distance(world, probe, dir_x, dir_y)


def choose_explore_goal(world, tank):
    best_goal = None
    best_score = -999.0

    for _ in range(48):
        angle = random.uniform(-math.pi, math.pi)
        distance = random.uniform(600.0, EXPLORE_GOAL_RANGE)
        x = clamp(tank.x + math.cos(angle) * distance, tank.radius, WORLD_WIDTH - tank.radius)
        y = clamp(tank.y + math.sin(angle) * distance, tank.radius, WORLD_HEIGHT - tank.radius)
        if not _position_is_safe(world, x, y, tank.radius):
            continue

        score = _goal_score(world, tank, x, y)
        if score > best_score:
            best_goal = (x, y)
            best_score = score

    if best_goal is not None:
        return best_goal

    best_ray = max(
        WALL_RAY_DIRECTIONS,
        key=lambda direction: wall_probe_distance(world, tank, direction[0], direction[1]),
    )
    x = clamp(tank.x + best_ray[0] * 900.0, tank.radius, WORLD_WIDTH - tank.radius)
    y = clamp(tank.y + best_ray[1] * 900.0, tank.radius, WORLD_HEIGHT - tank.radius)
    return x, y


def get_explore_goal(world, tank, force=False):
    goal = getattr(tank, "explore_goal", None)
    if goal is None or force:
        goal = choose_explore_goal(world, tank)
        tank.explore_goal = goal
        return goal

    distance = math.hypot(goal[0] - tank.x, goal[1] - tank.y)
    if distance < EXPLORE_GOAL_REACHED_DISTANCE:
        goal = choose_explore_goal(world, tank)
        tank.explore_goal = goal
    return goal


def maybe_replan_explore_goal(world, tank, moved_dist, target_visible):
    if target_visible:
        return getattr(tank, "explore_goal", None)

    stuck_ticks = getattr(tank, "nav_stuck_ticks", 0)
    if moved_dist < 0.35:
        stuck_ticks += 1
    else:
        stuck_ticks = max(0, stuck_ticks - 2)
    tank.nav_stuck_ticks = stuck_ticks

    return get_explore_goal(world, tank, force=stuck_ticks >= 18)


def _goal_features(tank, nav_goal):
    if nav_goal is None:
        return 0.0, 0.0, 1.0

    dx = nav_goal[0] - tank.x
    dy = nav_goal[1] - tank.y
    distance = math.hypot(dx, dy)
    return (
        clamp(dx / EXPLORE_GOAL_RANGE, -1.0, 1.0),
        clamp(dy / EXPLORE_GOAL_RANGE, -1.0, 1.0),
        clamp(distance / EXPLORE_GOAL_RANGE, 0.0, 1.0),
    )


def build_farming_observation(world, tank, nav_goal=None):
    if nav_goal is None:
        nav_goal = get_explore_goal(world, tank)

    target, target_dist, target_visible = find_farming_target(world, tank)
    if target is not None and math.isfinite(target_dist):
        dx = target.x - tank.x
        dy = target.y - tank.y
        target_angle = math.atan2(dy, dx)
        target_dist_norm = clamp(target_dist / TARGET_SCAN_RANGE, 0.0, 1.0)
        target_hp_ratio = clamp(target.hp / target.max_hp, 0.0, 1.0)
        target_dx_norm = clamp(dx / TARGET_SCAN_RANGE, -1.0, 1.0)
        target_dy_norm = clamp(dy / TARGET_SCAN_RANGE, -1.0, 1.0)
        target_id = str(id(target))
    else:
        target = None
        target_dist = float("inf")
        target_angle = tank.angle
        target_dist_norm = 1.0
        target_hp_ratio = 0.0
        target_dx_norm = 0.0
        target_dy_norm = 0.0
        target_id = None

    wall_probes = [
        wall_probe_distance(world, tank, dir_x, dir_y) / WALL_PROBE_RANGE
        for dir_x, dir_y in WALL_RAY_DIRECTIONS
    ]
    min_wall_probe = min(wall_probes) if wall_probes else 1.0
    goal_dx_norm, goal_dy_norm, goal_dist_norm = _goal_features(tank, nav_goal)
    stuck_ratio = clamp(getattr(tank, "nav_stuck_ticks", 0) / 24.0, 0.0, 1.0)

    obs = [
        tank.x / WORLD_WIDTH,
        tank.y / WORLD_HEIGHT,
        tank.hp / tank.max_hp,
        1.0 if tank.shoot_cooldown <= 0 else 0.0,
        target_dx_norm,
        target_dy_norm,
        target_dist_norm,
        target_hp_ratio,
        1.0 if target_visible else 0.0,
        math.cos(target_angle),
        math.sin(target_angle),
        goal_dx_norm,
        goal_dy_norm,
        goal_dist_norm,
        min_wall_probe,
        stuck_ratio,
        *wall_probes,
    ]

    info = {
        "target": target,
        "target_id": target_id,
        "target_dist": target_dist,
        "target_angle": target_angle,
        "target_visible": target_visible,
        "nav_goal": nav_goal,
        "goal_dist": math.hypot(nav_goal[0] - tank.x, nav_goal[1] - tank.y) if nav_goal else float("inf"),
        "wall_probes": wall_probes,
        "min_wall_probe": min_wall_probe,
        "stuck_ratio": stuck_ratio,
    }
    return np.array(obs, dtype=np.float32), info


def normalize_move_action(action):
    if isinstance(action, np.ndarray):
        action = action.item()
    return int(clamp(int(action), 0, FARMING_ACTION_COUNT - 1))


def movement_vector(action):
    return MOVE_DIRECTIONS[normalize_move_action(action)]


def movement_flags(action):
    move_x, move_y = movement_vector(action)
    return {
        "up": move_y < 0,
        "down": move_y > 0,
        "left": move_x < 0,
        "right": move_x > 0,
    }


def _projected_action_position(tank, action, scale=5.0):
    move_x, move_y = movement_vector(action)
    move_mag = math.hypot(move_x, move_y)
    if move_mag > 0.01:
        move_x /= move_mag
        move_y /= move_mag
    projected_distance = max(18.0, tank.speed * scale)
    return tank.x + move_x * projected_distance, tank.y + move_y * projected_distance


def action_is_blocked(world, tank, action):
    action = normalize_move_action(action)
    if action == 0:
        return False

    x, y = _projected_action_position(tank, action)
    if x < tank.radius or x > WORLD_WIDTH - tank.radius or y < tank.radius or y > WORLD_HEIGHT - tank.radius:
        return True

    for wall in world.wall_grid.get_nearby(tank):
        if _circle_rect_collide(x, y, tank.radius + 8.0, wall.rect):
            return True
    return False


def desired_farming_vector(info):
    target = info["target"]
    target_dist = info["target_dist"]
    target_angle = info["target_angle"]

    if target is not None and math.isfinite(target_dist) and info.get("target_visible"):
        if target_dist > IDEAL_MAX_DISTANCE:
            return math.cos(target_angle), math.sin(target_angle)
        if target_dist < IDEAL_MIN_DISTANCE:
            return -math.cos(target_angle), -math.sin(target_angle)
        return math.cos(target_angle + math.pi / 2), math.sin(target_angle + math.pi / 2)

    nav_goal = info.get("nav_goal")
    if nav_goal is None:
        return 0.0, 0.0

    dx = nav_goal[0] - getattr(info.get("tank_proxy", None), "x", 0.0)
    dy = nav_goal[1] - getattr(info.get("tank_proxy", None), "y", 0.0)
    # tank_proxy is only used by old callers; new callers should use desired_navigation_vector().
    if abs(dx) + abs(dy) < 1e-6:
        return 0.0, 0.0
    angle = math.atan2(dy, dx)
    return math.cos(angle), math.sin(angle)


def desired_navigation_vector(tank, info):
    target = info["target"]
    target_dist = info["target_dist"]
    target_angle = info["target_angle"]

    if target is not None and math.isfinite(target_dist) and info.get("target_visible"):
        if target_dist > IDEAL_MAX_DISTANCE:
            return math.cos(target_angle), math.sin(target_angle)
        if target_dist < IDEAL_MIN_DISTANCE:
            return -math.cos(target_angle), -math.sin(target_angle)
        return math.cos(target_angle + math.pi / 2), math.sin(target_angle + math.pi / 2)

    nav_goal = info.get("nav_goal")
    if nav_goal is None:
        return 0.0, 0.0
    dx = nav_goal[0] - tank.x
    dy = nav_goal[1] - tank.y
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        return 0.0, 0.0
    return dx / distance, dy / distance


def repair_move_action(world, tank, action, info):
    action = normalize_move_action(action)
    desired_x, desired_y = desired_navigation_vector(tank, info)
    should_move = info.get("target_visible") is False or info.get("target_dist", float("inf")) > IDEAL_MAX_DISTANCE

    if action != 0 and not action_is_blocked(world, tank, action):
        return action, False
    if action == 0 and not should_move:
        return action, False

    best_action = action
    best_score = -999.0
    for candidate in range(1, FARMING_ACTION_COUNT):
        if action_is_blocked(world, tank, candidate):
            continue

        move_x, move_y = movement_vector(candidate)
        move_mag = math.hypot(move_x, move_y)
        if move_mag < 0.01:
            continue

        alignment = 0.0
        if math.hypot(desired_x, desired_y) > 0.01:
            alignment = ((move_x * desired_x) + (move_y * desired_y)) / move_mag

        projected_x, projected_y = _projected_action_position(tank, candidate, scale=9.0)
        center_score = 1.0 - (
            abs(projected_x - WORLD_WIDTH / 2) / (WORLD_WIDTH / 2)
            + abs(projected_y - WORLD_HEIGHT / 2) / (WORLD_HEIGHT / 2)
        ) * 0.15
        score = alignment + center_score * 0.15 + random.random() * 0.02
        if score > best_score:
            best_action = candidate
            best_score = score

    if best_action == action:
        return action, False
    return best_action, True


def decode_farming_action(world, tank, action, info=None, repair=True):
    if info is None:
        _, info = build_farming_observation(world, tank)

    original_action = normalize_move_action(action)
    repaired_action = original_action
    repaired = False
    if repair:
        repaired_action, repaired = repair_move_action(world, tank, original_action, info)

    input_state = movement_flags(repaired_action)
    target = info["target"]
    target_visible = info["target_visible"]
    target_dist = info["target_dist"]

    input_state["aim_angle"] = info["target_angle"] if target is not None else tank.angle
    input_state["fire"] = bool(target_visible and target_dist <= FIRE_RANGE)
    input_state["_action_index"] = repaired_action
    input_state["_original_action_index"] = original_action
    input_state["_action_repaired"] = repaired
    input_state["_action_blocked"] = action_is_blocked(world, tank, original_action)
    return input_state


def apply_farming_upgrades(tank):
    while tank.stat_points > 0:
        upgraded = False
        for stat_index in STAT_UPGRADE_PRIORITY:
            before_points = tank.stat_points
            tank.upgrade_stat(stat_index)
            if tank.stat_points < before_points:
                upgraded = True
                break
        if not upgraded:
            break

    available = tank.available_tank_upgrades()
    for tank_type in TANK_UPGRADE_PRIORITY:
        if tank_type in available:
            tank.evolve_tank(tank_type)
            break
