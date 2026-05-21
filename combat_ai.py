import math

import numpy as np

from farming_ai import (
    MOVE_DIRECTIONS,
    WALL_PROBE_RANGE,
    WALL_RAY_DIRECTIONS,
    clamp,
    line_of_sight,
    movement_flags,
    movement_vector,
    normalize_move_action,
    wall_probe_distance,
)
from settings import WORLD_HEIGHT, WORLD_WIDTH

COMBAT_MODEL_VERSION = 1
COMBAT_MODEL_NAME = f"diep_combat_ai_model_v{COMBAT_MODEL_VERSION}"

COMBAT_TARGET_SCAN_RANGE = 1700.0
COMBAT_FIRE_RANGE = 1050.0
COMBAT_IDEAL_MIN_DISTANCE = 360.0
COMBAT_IDEAL_MAX_DISTANCE = 780.0
COMBAT_DANGER_DISTANCE = 210.0
BULLET_SCAN_RANGE = 650.0

COMBAT_OBSERVATION_SIZE = 26
COMBAT_ACTION_COUNT = len(MOVE_DIRECTIONS)

COMBAT_STAT_UPGRADE_PRIORITY = (5, 6, 3, 4, 7, 1, 0, 2)
COMBAT_TANK_UPGRADE_PRIORITY = (
    "Twin",
    "Machine Gun",
    "Sniper",
    "Triple Shot",
    "Gunner",
    "Destroyer",
    "Assassin",
    "Hunter",
    "Quad Tank",
    "Tri-Angle",
    "Twin Flank",
    "Flank Guard",
)


def find_combat_target(world, tank, candidates=None, max_range=COMBAT_TARGET_SCAN_RANGE):
    if candidates is None:
        candidates = world.players.values()

    nearest = None
    nearest_dist = float("inf")
    nearest_visible = False

    for target in candidates:
        if target is tank or not target.alive:
            continue
        if getattr(target, "entity_id", None) == getattr(tank, "entity_id", None):
            continue

        distance = math.hypot(target.x - tank.x, target.y - tank.y)
        if distance > max_range or distance >= nearest_dist:
            continue

        nearest = target
        nearest_dist = distance
        nearest_visible = line_of_sight(world, tank, target)

    return nearest, nearest_dist, nearest_visible


def _bullet_threat_score(tank, bullet):
    rel_x = tank.x - bullet.x
    rel_y = tank.y - bullet.y
    distance = math.hypot(rel_x, rel_y)
    if distance <= 0.001 or distance > BULLET_SCAN_RANGE:
        return 0.0, distance, 0.0

    bullet_speed = max(0.001, math.hypot(bullet.dx, bullet.dy))
    dir_x = bullet.dx / bullet_speed
    dir_y = bullet.dy / bullet_speed
    forward = (rel_x * dir_x) + (rel_y * dir_y)
    if forward <= 0:
        return 0.0, distance, 0.0

    lateral_sq = max(0.0, distance * distance - forward * forward)
    lateral = math.sqrt(lateral_sq)
    hit_width = tank.radius + bullet.radius + 55.0
    if lateral > hit_width:
        return 0.0, distance, lateral

    time_factor = 1.0 - clamp(forward / BULLET_SCAN_RANGE, 0.0, 1.0)
    lateral_factor = 1.0 - clamp(lateral / hit_width, 0.0, 1.0)
    return time_factor * lateral_factor, distance, lateral


def find_bullet_threat(world, tank):
    best_bullet = None
    best_score = 0.0
    best_dist = float("inf")
    best_lateral = 0.0

    for bullet in world.bullets:
        if not bullet.alive or bullet.owner_id == tank.entity_id:
            continue

        score, distance, lateral = _bullet_threat_score(tank, bullet)
        if score > best_score:
            best_bullet = bullet
            best_score = score
            best_dist = distance
            best_lateral = lateral

    return best_bullet, best_score, best_dist, best_lateral


def build_combat_observation(world, tank, candidates=None):
    target, target_dist, target_visible = find_combat_target(world, tank, candidates=candidates)
    if target is not None and math.isfinite(target_dist):
        dx = target.x - tank.x
        dy = target.y - tank.y
        target_angle = math.atan2(dy, dx)
        target_dx_norm = clamp(dx / COMBAT_TARGET_SCAN_RANGE, -1.0, 1.0)
        target_dy_norm = clamp(dy / COMBAT_TARGET_SCAN_RANGE, -1.0, 1.0)
        target_dist_norm = clamp(target_dist / COMBAT_TARGET_SCAN_RANGE, 0.0, 1.0)
        target_hp_ratio = clamp(target.hp / target.max_hp, 0.0, 1.0)
        target_ready = 1.0 if target.shoot_cooldown <= 0 else 0.0
        target_id = str(id(target))
    else:
        target = None
        target_dist = float("inf")
        target_angle = tank.angle
        target_dx_norm = 0.0
        target_dy_norm = 0.0
        target_dist_norm = 1.0
        target_hp_ratio = 0.0
        target_ready = 0.0
        target_id = None

    threat, threat_score, threat_dist, _ = find_bullet_threat(world, tank)
    if threat is not None:
        bullet_dx = clamp((threat.x - tank.x) / BULLET_SCAN_RANGE, -1.0, 1.0)
        bullet_dy = clamp((threat.y - tank.y) / BULLET_SCAN_RANGE, -1.0, 1.0)
        bullet_dist_norm = clamp(threat_dist / BULLET_SCAN_RANGE, 0.0, 1.0)
        bullet_speed = max(0.001, math.hypot(threat.dx, threat.dy))
        bullet_dir_x = clamp(threat.dx / bullet_speed, -1.0, 1.0)
        bullet_dir_y = clamp(threat.dy / bullet_speed, -1.0, 1.0)
    else:
        bullet_dx = 0.0
        bullet_dy = 0.0
        bullet_dist_norm = 1.0
        bullet_dir_x = 0.0
        bullet_dir_y = 0.0

    wall_probes = [
        wall_probe_distance(world, tank, dir_x, dir_y) / WALL_PROBE_RANGE
        for dir_x, dir_y in WALL_RAY_DIRECTIONS
    ]

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
        target_ready,
        math.cos(target_angle),
        math.sin(target_angle),
        bullet_dx,
        bullet_dy,
        bullet_dist_norm,
        bullet_dir_x,
        bullet_dir_y,
        clamp(threat_score, 0.0, 1.0),
        *wall_probes,
    ]

    info = {
        "target": target,
        "target_id": target_id,
        "target_dist": target_dist,
        "target_angle": target_angle,
        "target_visible": target_visible,
        "bullet_threat": threat,
        "bullet_threat_score": threat_score,
        "bullet_threat_dist": threat_dist,
        "wall_probes": wall_probes,
    }
    return np.array(obs, dtype=np.float32), info


def decode_combat_action(world, tank, action, info=None):
    if info is None:
        _, info = build_combat_observation(world, tank)

    action = normalize_move_action(action)
    input_state = movement_flags(action)
    target = info["target"]
    target_visible = info["target_visible"]
    target_dist = info["target_dist"]

    input_state["aim_angle"] = info["target_angle"] if target is not None else tank.angle
    input_state["fire"] = bool(target is not None and target_visible and target_dist <= COMBAT_FIRE_RANGE)
    return input_state


def apply_combat_upgrades(tank):
    while tank.stat_points > 0:
        upgraded = False
        for stat_index in COMBAT_STAT_UPGRADE_PRIORITY:
            before_points = tank.stat_points
            tank.upgrade_stat(stat_index)
            if tank.stat_points < before_points:
                upgraded = True
                break
        if not upgraded:
            break

    available = tank.available_tank_upgrades()
    for tank_type in COMBAT_TANK_UPGRADE_PRIORITY:
        if tank_type in available:
            tank.evolve_tank(tank_type)
            break


def desired_combat_vector(info, strafe_sign=1.0):
    threat = info.get("bullet_threat")
    threat_score = info.get("bullet_threat_score", 0.0)
    if threat is not None and threat_score > 0.05:
        bullet_speed = max(0.001, math.hypot(threat.dx, threat.dy))
        dir_x = threat.dx / bullet_speed
        dir_y = threat.dy / bullet_speed
        away_x = -dir_y * strafe_sign
        away_y = dir_x * strafe_sign
        return away_x, away_y

    target = info.get("target")
    target_dist = info.get("target_dist", float("inf"))
    target_angle = info.get("target_angle", 0.0)
    if target is None or not math.isfinite(target_dist):
        return 0.0, 0.0

    if target_dist > COMBAT_IDEAL_MAX_DISTANCE:
        return math.cos(target_angle), math.sin(target_angle)

    if target_dist < COMBAT_IDEAL_MIN_DISTANCE:
        return -math.cos(target_angle), -math.sin(target_angle)

    return math.cos(target_angle + (math.pi / 2) * strafe_sign), math.sin(target_angle + (math.pi / 2) * strafe_sign)


def scripted_combat_input(world, tank, target, tick_count=0, strafe_sign=1.0):
    _, info = build_combat_observation(world, tank, candidates=[target])
    desired_x, desired_y = desired_combat_vector(info, strafe_sign=strafe_sign)

    best_action = 0
    best_score = -999.0
    for index, (move_x, move_y) in enumerate(MOVE_DIRECTIONS):
        move_mag = math.hypot(move_x, move_y)
        desired_mag = math.hypot(desired_x, desired_y)
        if move_mag < 0.01 or desired_mag < 0.01:
            score = -0.05 if move_mag < 0.01 else 0.0
        else:
            score = ((move_x * desired_x) + (move_y * desired_y)) / (move_mag * desired_mag)
        if score > best_score:
            best_action = index
            best_score = score

    return decode_combat_action(world, tank, best_action, info)
