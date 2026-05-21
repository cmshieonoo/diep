import math
import random

import pygame

from settings import (
    BASE_TANK_RADIUS,
    BLUE,
    DARK_BLUE,
    DARK_RED,
    LEVEL_XP,
    MAX_STAT_LEVEL,
    PLAYER_BASE_HP,
    PLAYER_BASE_SPEED,
    PLAYER_BASE_RELOAD_TICKS,
    PLAYER_MIN_RELOAD_TICKS,
    PLAYER_RELOAD_REDUCTION_PER_STAT,
    PLAYER_SPEED_PER_STAT,
    RECOIL_FRICTION,
    RED,
    SHAPE_STATS,
    WORLD_HEIGHT,
    WORLD_WIDTH,
)
from tank_defs import build_barrel_polygon, get_barrel_anchor, get_tank_definition, get_tank_display_name


def default_input_state():
    return {
        "up": False,
        "down": False,
        "left": False,
        "right": False,
        "fire": False,
        "aim_angle": 0.0,
        "auto_fire": False,
        "auto_spin": False,
    }


def gains_stat_point_on_level(level):
    if 2 <= level <= 28:
        return True
    if level == 30:
        return True
    return 31 <= level <= 45 and level % 3 == 0


def stat_points_for_level(level):
    return sum(1 for current_level in range(2, level + 1) if gains_stat_point_on_level(current_level))


def resolve_circle_rect_collision(obj, rect):
    closest_x = max(rect.left, min(obj.x, rect.right))
    closest_y = max(rect.top, min(obj.y, rect.bottom))
    dist_x = obj.x - closest_x
    dist_y = obj.y - closest_y
    dist_sq = dist_x ** 2 + dist_y ** 2

    if dist_sq < obj.radius ** 2:
        if dist_sq == 0:
            distances = {
                "left": obj.x - rect.left,
                "right": rect.right - obj.x,
                "top": obj.y - rect.top,
                "bottom": rect.bottom - obj.y,
            }
            side = min(distances, key=distances.get)

            if side == "left":
                obj.x = rect.left - obj.radius
                horizontal = True
            elif side == "right":
                obj.x = rect.right + obj.radius
                horizontal = True
            elif side == "top":
                obj.y = rect.top - obj.radius
                horizontal = False
            else:
                obj.y = rect.bottom + obj.radius
                horizontal = False

            if hasattr(obj, "push_vx"):
                if horizontal:
                    obj.push_vx = 0
                else:
                    obj.push_vy = 0
            if hasattr(obj, "push_target_vx"):
                if horizontal:
                    obj.push_target_vx = 0
                else:
                    obj.push_target_vy = 0

            if hasattr(obj, "dx"):
                if horizontal:
                    obj.dx *= -1
                else:
                    obj.dy *= -1
            return True

        dist = math.sqrt(dist_sq)
        overlap = obj.radius - dist
        horizontal = abs(dist_x) > abs(dist_y)

        obj.x += (dist_x / dist) * overlap
        obj.y += (dist_y / dist) * overlap

        if hasattr(obj, "push_vx"):
            if horizontal:
                obj.push_vx = 0
            else:
                obj.push_vy = 0
        if hasattr(obj, "push_target_vx"):
            if horizontal:
                obj.push_target_vx = 0
            else:
                obj.push_target_vy = 0

        if hasattr(obj, "dx"):
            if horizontal:
                if (obj.x < closest_x and obj.dx > 0) or (obj.x > closest_x and obj.dx < 0):
                    obj.dx *= -1
            else:
                if (obj.y < closest_y and obj.dy > 0) or (obj.y > closest_y and obj.dy < 0):
                    obj.dy *= -1
        return True
    return False


def circle_rect_collide(cx, cy, radius, rect):
    closest_x = max(rect.left, min(cx, rect.right))
    closest_y = max(rect.top, min(cy, rect.bottom))
    return (cx - closest_x) ** 2 + (cy - closest_y) ** 2 < radius ** 2


def resolve_circle_collision(obj1, obj2):
    dx = obj1.x - obj2.x
    dy = obj1.y - obj2.y
    distance = math.hypot(dx, dy)
    min_dist = obj1.radius + obj2.radius

    if distance < min_dist and distance > 0:
        overlap = min_dist - distance
        push_x = (dx / distance) * (overlap / 2)
        push_y = (dy / distance) * (overlap / 2)

        obj1.x += push_x
        obj1.y += push_y
        obj2.x -= push_x
        obj2.y -= push_y

        if hasattr(obj1, "dx") and hasattr(obj2, "dx"):
            obj1.dx, obj2.dx = obj2.dx, obj1.dx
            obj1.dy, obj2.dy = obj2.dy, obj1.dy


class Wall:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)

    def to_state(self):
        return {
            "x": self.rect.x,
            "y": self.rect.y,
            "w": self.rect.width,
            "h": self.rect.height,
        }


class GameObject:
    def __init__(self, x, y, radius, color):
        self.x = float(x)
        self.y = float(y)
        self.radius = radius
        self.color = color
        self.alive = True

    def collides_with(self, other):
        return math.hypot(self.x - other.x, self.y - other.y) < (self.radius + other.radius)


class Player(GameObject):
    def __init__(
        self,
        x,
        y,
        name="Player",
        entity_id=None,
        color=BLUE,
        border_color=DARK_BLUE,
        is_bot=False,
    ):
        super().__init__(x, y, BASE_TANK_RADIUS, color)
        self.entity_id = entity_id
        self.name = name
        self.border_color = border_color
        self.is_bot = is_bot

        self.shoot_cooldown = 0
        self.push_vx = 0.0
        self.push_vy = 0.0
        self.push_target_vx = 0.0
        self.push_target_vy = 0.0
        self.frames_since_last_hit = 0
        self.auto_fire = False
        self.auto_spin = False
        self.immune_timer = 0
        self.damage_flash_timer = 0
        self.angle = 0.0

        self.spawn_tick = 0
        self.killer_name = ""
        self.time_alive = 0

        self._reset_progression()
        self.update_base_stats()
        self.hp = self.max_hp

    def _reset_progression(self):
        self.tank_type = "Basic"
        self.fire_cycle = 0
        self.level = 1
        self.xp = 0
        self.stats = [0] * 8
        self.stat_points = 0
        self.radius = BASE_TANK_RADIUS

    def _set_progression_for_level(self, level, xp=None):
        target_level = max(1, min(45, int(level)))
        self.tank_type = "Basic"
        self.fire_cycle = 0
        self.level = target_level
        self.xp = LEVEL_XP[target_level - 1] if xp is None and target_level > 1 else (0 if xp is None else int(xp))
        self.stats = [0] * 8
        self.stat_points = stat_points_for_level(target_level)
        self.radius = BASE_TANK_RADIUS * (1 + (target_level - 1) * 0.01)

    def update_base_stats(self):
        tank_def = get_tank_definition(self.tank_type)
        raw_max_hp = PLAYER_BASE_HP + (self.level * 2) + (self.stats[1] * 20)
        raw_speed = PLAYER_BASE_SPEED + (self.stats[7] * PLAYER_SPEED_PER_STAT)
        self.max_hp = raw_max_hp * tank_def["max_health_multiplier"]
        self.speed = raw_speed * tank_def["move_speed_multiplier"]

    def available_tank_upgrades(self):
        if not self.alive:
            return []

        tank_def = get_tank_definition(self.tank_type)
        if self.tank_type == "Basic" and self.level >= 30:
            return ["Smasher"]
        upgrade_level = tank_def.get("upgrade_level")
        if upgrade_level is None or self.level < upgrade_level:
            return []
        return list(tank_def.get("upgrades", []))

    def evolve_tank(self, new_tank_type):
        if new_tank_type not in self.available_tank_upgrades():
            return False

        hp_ratio = self.hp / self.max_hp if self.max_hp > 0 else 1.0
        self.tank_type = new_tank_type
        self.fire_cycle = 0
        self.update_base_stats()
        self.hp = max(1.0, min(self.max_hp, self.max_hp * hp_ratio))
        return True

    def upgrade_stat(self, index):
        if self.stat_points > 0 and 0 <= index < len(self.stats) and self.stats[index] < MAX_STAT_LEVEL:
            self.stats[index] += 1
            self.stat_points -= 1
            hp_ratio = self.hp / self.max_hp if self.max_hp > 0 else 1.0
            self.update_base_stats()
            self.hp = max(1.0, min(self.max_hp, self.max_hp * hp_ratio))

    def take_damage(self, amount):
        if not self.alive or self.immune_timer > 0:
            return

        self.hp -= amount
        self.frames_since_last_hit = 0
        self.damage_flash_timer = 10

    def apply_recoil(self, angle, scale=1.0):
        recoil_force = (1.5 + (self.stats[5] * 0.2)) * scale
        self.push_target_vx -= math.cos(angle) * recoil_force
        self.push_target_vy -= math.sin(angle) * recoil_force

    def get_reload_ticks(self):
        base_reload = max(
            PLAYER_MIN_RELOAD_TICKS,
            PLAYER_BASE_RELOAD_TICKS - (self.stats[6] * PLAYER_RELOAD_REDUCTION_PER_STAT),
        )
        return max(PLAYER_MIN_RELOAD_TICKS, int(round(base_reload * get_tank_definition(self.tank_type)["reload_multiplier"])))

    def _selected_barrels(self):
        tank_def = get_tank_definition(self.tank_type)
        barrels = tank_def["barrels"]
        if tank_def["fire_mode"] == "cycle":
            groups = sorted({barrel.get("cycle_group", 0) for barrel in barrels})
            group = groups[self.fire_cycle % len(groups)]
            self.fire_cycle = (self.fire_cycle + 1) % len(groups)
            return [barrel for barrel in barrels if barrel.get("cycle_group", 0) == group]
        return list(barrels)

    def build_shots(self):
        tank_def = get_tank_definition(self.tank_type)
        selected_barrels = self._selected_barrels()
        shots = []
        barrel_scale = self.radius / BASE_TANK_RADIUS

        for barrel in selected_barrels:
            spread = tank_def["spread"] + barrel.get("spread", 0.0)
            shot_angle = self.angle + barrel.get("angle_offset", 0.0)
            if spread > 0:
                shot_angle += random.uniform(-spread, spread)

            base_x, base_y, _ = get_barrel_anchor(self.x, self.y, self.angle, barrel, scale=barrel_scale)
            muzzle_distance = (barrel.get("length", 50) + barrel.get("muzzle_extra", 8.0)) * barrel_scale
            spawn_x = base_x + math.cos(shot_angle) * muzzle_distance
            spawn_y = base_y + math.sin(shot_angle) * muzzle_distance

            shots.append(
                {
                    "x": spawn_x,
                    "y": spawn_y,
                    "angle": shot_angle,
                    "speed_multiplier": tank_def["bullet_speed_multiplier"] * barrel.get("bullet_speed_multiplier", 1.0),
                    "damage_multiplier": tank_def["bullet_damage_multiplier"] * barrel.get("bullet_damage_multiplier", 1.0),
                    "life_multiplier": tank_def["bullet_life_multiplier"] * barrel.get("bullet_life_multiplier", 1.0),
                    "size_multiplier": tank_def["bullet_size_multiplier"] * barrel.get("bullet_size_multiplier", 1.0),
                    "penetration_bonus": tank_def["penetration_bonus"] + barrel.get("penetration_bonus", 0),
                    "recoil_multiplier": tank_def["recoil_multiplier"] * barrel.get("recoil_multiplier", 1.0),
                }
            )

        return shots

    def update(self, input_state, walls):
        if not self.alive:
            return

        input_state = input_state or default_input_state()

        if self.damage_flash_timer > 0:
            self.damage_flash_timer -= 1

        moved = input_state["up"] or input_state["down"] or input_state["left"] or input_state["right"]
        if self.immune_timer > 0:
            self.immune_timer -= 1
            if moved:
                self.immune_timer = 0

        self.frames_since_last_hit += 1
        regen_delay = max(60, 1800 - (self.stats[0] * 230))
        if self.hp < self.max_hp:
            if self.frames_since_last_hit > regen_delay:
                self.hp += self.max_hp * 0.01
            self.hp = min(self.hp, self.max_hp)

        self.auto_fire = bool(input_state.get("auto_fire", self.auto_fire))
        self.auto_spin = bool(input_state.get("auto_spin", self.auto_spin))

        if self.auto_spin:
            self.angle += 0.05
        else:
            self.angle = float(input_state.get("aim_angle", self.angle))

        move_x = (1 if input_state["right"] else 0) - (1 if input_state["left"] else 0)
        move_y = (1 if input_state["down"] else 0) - (1 if input_state["up"] else 0)

        self.push_vx += (self.push_target_vx - self.push_vx) * 0.24
        self.push_vy += (self.push_target_vy - self.push_vy) * 0.24
        self.push_target_vx *= 0.62
        self.push_target_vy *= 0.62

        self.x += (move_x * self.speed) + self.push_vx
        self.y += (move_y * self.speed) + self.push_vy

        self.push_vx *= 0.88
        self.push_vy *= 0.88

        self.x = max(self.radius, min(WORLD_WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(WORLD_HEIGHT - self.radius, self.y))

        for wall in walls:
            resolve_circle_rect_collision(self, wall.rect)

        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

    def wants_to_fire(self, input_state):
        if not self.alive or self.shoot_cooldown > 0:
            return False
        return bool(input_state.get("fire")) or self.auto_fire

    def add_xp(self, amount):
        if self.level >= 45:
            return

        self.xp += amount
        while self.level < 45 and self.xp >= LEVEL_XP[self.level]:
            self.level += 1
            if gains_stat_point_on_level(self.level):
                self.stat_points += 1
            self.radius += 0.3
            hp_ratio = self.hp / self.max_hp if self.max_hp > 0 else 1.0
            self.update_base_stats()
            self.hp = max(1.0, min(self.max_hp, self.max_hp * hp_ratio))

    def mark_dead(self, killer_name, time_alive_seconds):
        self.alive = False
        self.hp = 0
        self.push_vx = 0
        self.push_vy = 0
        self.push_target_vx = 0
        self.push_target_vy = 0
        self.shoot_cooldown = 0
        self.immune_timer = 0
        self.killer_name = killer_name
        self.time_alive = time_alive_seconds

    def respawn(self, x, y, spawn_tick):
        respawn_level = max(1, self.level // 2)
        self.x = float(x)
        self.y = float(y)
        self.alive = True
        self._set_progression_for_level(respawn_level)
        self.update_base_stats()
        self.hp = self.max_hp
        self.push_vx = 0.0
        self.push_vy = 0.0
        self.push_target_vx = 0.0
        self.push_target_vy = 0.0
        self.shoot_cooldown = 0
        self.immune_timer = 180
        self.damage_flash_timer = 0
        self.frames_since_last_hit = 0
        self.spawn_tick = spawn_tick
        self.killer_name = ""
        self.time_alive = 0
        self.auto_fire = False
        self.auto_spin = False
        self.angle = 0.0

    def to_state(self):
        return {
            "id": self.entity_id,
            "name": self.name,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "radius": round(self.radius, 2),
            "angle": round(self.angle, 4),
            "color": list(self.color),
            "border_color": list(self.border_color),
            "hp": round(self.hp, 2),
            "max_hp": round(self.max_hp, 2),
            "xp": int(self.xp),
            "level": self.level,
            "stats": list(self.stats),
            "stat_points": self.stat_points,
            "auto_fire": self.auto_fire,
            "auto_spin": self.auto_spin,
            "alive": self.alive,
            "is_bot": self.is_bot,
            "killer_name": self.killer_name,
            "time_alive": self.time_alive,
            "tank_type": self.tank_type,
            "tank_display_name": get_tank_display_name(self.tank_type),
            "available_tank_upgrades": self.available_tank_upgrades(),
        }


class Enemy(Player):
    def __init__(self, x, y, entity_id=None):
        super().__init__(
            x,
            y,
            name=f"Bot {random.randint(10, 99)}",
            entity_id=entity_id,
            color=RED,
            border_color=DARK_RED,
            is_bot=True,
        )
        self.shoot_intent = False
        self.state_timer = 0
        self.move_vx = 0.0
        self.move_vy = 0.0
        self.wander_angle = random.uniform(0, math.pi * 2)

    def update_ai(self, walls, living_players, shapes, bullets):
        if not self.alive:
            return

        while True:
            options = self.available_tank_upgrades()
            if not options:
                break
            self.evolve_tank(random.choice(options))

        while self.stat_points > 0:
            self.upgrade_stat(random.randint(0, 7))

        dodge_vx = 0.0
        dodge_vy = 0.0
        is_dodging = False

        for bullet in bullets:
            if bullet.owner is self or not bullet.alive:
                continue

            distance = math.hypot(self.x - bullet.x, self.y - bullet.y)
            if distance < 300:
                is_dodging = True
                force = 300 - distance
                angle_from_bullet = math.atan2(self.y - bullet.y, self.x - bullet.x)
                dodge_vx += math.cos(angle_from_bullet) * force
                dodge_vy += math.sin(angle_from_bullet) * force

        target = None
        target_dist = float("inf")

        for player in living_players:
            distance = math.hypot(self.x - player.x, self.y - player.y)
            if distance < 700 and distance < target_dist:
                target = player
                target_dist = distance

        if target is None:
            for shape in shapes:
                if not shape.alive:
                    continue
                distance = math.hypot(self.x - shape.x, self.y - shape.y)
                if distance < 400 and distance < target_dist:
                    target = shape
                    target_dist = distance

        self.shoot_intent = False
        target_angle = self.angle

        if is_dodging:
            dodge_angle = math.atan2(dodge_vy, dodge_vx)
            self.move_vx = math.cos(dodge_angle) * self.speed
            self.move_vy = math.sin(dodge_angle) * self.speed
            target_angle = math.atan2(target.y - self.y, target.x - self.x) if target else dodge_angle
            self.shoot_intent = target is not None
        elif target is not None:
            target_angle = math.atan2(target.y - self.y, target.x - self.x)
            if target_dist > 250:
                self.move_vx = math.cos(target_angle) * self.speed
                self.move_vy = math.sin(target_angle) * self.speed
            else:
                orbit_angle = target_angle + (math.pi / 2)
                self.move_vx = math.cos(orbit_angle) * (self.speed * 0.8)
                self.move_vy = math.sin(orbit_angle) * (self.speed * 0.8)
            self.shoot_intent = True
        else:
            self.state_timer -= 1
            if self.state_timer <= 0:
                self.wander_angle = random.uniform(0, math.pi * 2)
                self.state_timer = random.randint(60, 180)
            target_angle = self.wander_angle
            self.move_vx = math.cos(target_angle) * (self.speed * 0.5)
            self.move_vy = math.sin(target_angle) * (self.speed * 0.5)

        self.angle = target_angle

        self.push_vx += (self.push_target_vx - self.push_vx) * 0.24
        self.push_vy += (self.push_target_vy - self.push_vy) * 0.24
        self.push_target_vx *= 0.62
        self.push_target_vy *= 0.62

        self.x += self.move_vx + self.push_vx
        self.y += self.move_vy + self.push_vy

        self.push_vx *= 0.88
        self.push_vy *= 0.88

        for wall in walls:
            resolve_circle_rect_collision(self, wall.rect)

        self.x = max(self.radius, min(WORLD_WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(WORLD_HEIGHT - self.radius, self.y))

        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

        self.frames_since_last_hit += 1
        if self.hp < self.max_hp and self.frames_since_last_hit > 300:
            self.hp += self.max_hp * 0.005
            self.hp = min(self.hp, self.max_hp)

        if self.damage_flash_timer > 0:
            self.damage_flash_timer -= 1

    def wants_to_fire_ai(self):
        return self.alive and self.shoot_intent and self.shoot_cooldown <= 0


class Bullet(GameObject):
    def __init__(self, owner, shot):
        radius = max(8.0, 12.0 * shot["size_multiplier"])
        super().__init__(shot["x"], shot["y"], radius, owner.color)
        self.owner = owner
        self.owner_id = owner.entity_id
        self.border_color = owner.border_color
        self.speed = (8 + (owner.stats[3] * 1.5)) * shot["speed_multiplier"]
        self.damage = (7 + (owner.stats[5] * 3)) * shot["damage_multiplier"]
        self.hp = float(max(1, (2 + (owner.stats[4] * 1.5)) + shot["penetration_bonus"]))
        self.max_hp = self.hp
        self.dx = math.cos(shot["angle"]) * self.speed
        self.dy = math.sin(shot["angle"]) * self.speed
        self.life_timer = max(20, int((90 + (owner.stats[3] * 5)) * shot["life_multiplier"]))

    def get_bullet_collision_damage(self):
        return max(1.0, self.damage * 0.25)

    def spend_hp(self, amount):
        self.hp -= amount
        if self.hp <= 0:
            self.hp = 0
            self.alive = False

    def hit_target(self, resistance=1.0):
        resistance = max(1.0, float(resistance))
        damage_ratio = min(1.0, self.hp / resistance)
        dealt_damage = self.damage * damage_ratio
        self.spend_hp(resistance)
        return dealt_damage

    def update(self, walls):
        self.x += self.dx
        self.y += self.dy
        self.life_timer -= 1

        if self.life_timer <= 0 or not (0 < self.x < WORLD_WIDTH) or not (0 < self.y < WORLD_HEIGHT):
            self.alive = False
            return

        for wall in walls:
            if circle_rect_collide(self.x, self.y, self.radius, wall.rect):
                self.alive = False
                return

    def to_state(self):
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "radius": round(self.radius, 2),
            "color": list(self.color),
            "border_color": list(self.border_color),
        }


class Shape(GameObject):
    def __init__(self, x, y, sides, stats):
        super().__init__(x, y, stats["radius"], stats["color"])
        self.border_color = stats["border"]
        self.sides = sides
        self.hp = stats["hp"]
        self.max_hp = stats["hp"]
        self.xp_value = stats["xp"]
        self.angle = random.uniform(0, math.pi * 2)
        self.rot_speed = random.uniform(-0.02, 0.02)
        self.dx = random.uniform(-0.5, 0.5)
        self.dy = random.uniform(-0.5, 0.5)

    def take_damage(self, amount):
        self.hp -= amount

    def update(self, walls):
        self.angle += self.rot_speed
        self.x += self.dx
        self.y += self.dy

        if self.x <= self.radius or self.x >= WORLD_WIDTH - self.radius:
            self.dx *= -1
            self.x = max(self.radius, min(WORLD_WIDTH - self.radius, self.x))
        if self.y <= self.radius or self.y >= WORLD_HEIGHT - self.radius:
            self.dy *= -1
            self.y = max(self.radius, min(WORLD_HEIGHT - self.radius, self.y))

        for wall in walls:
            resolve_circle_rect_collision(self, wall.rect)

    def to_state(self):
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "radius": self.radius,
            "sides": self.sides,
            "angle": round(self.angle, 4),
            "color": list(self.color),
            "border_color": list(self.border_color),
            "hp": round(self.hp, 2),
            "max_hp": round(self.max_hp, 2),
        }


def create_shape():
    x = random.randint(50, WORLD_WIDTH - 50)
    y = random.randint(50, WORLD_HEIGHT - 50)
    rand = random.random()

    if rand < 0.6:
        return Shape(x, y, 4, SHAPE_STATS["square"])
    if rand < 0.9:
        return Shape(x, y, 3, SHAPE_STATS["triangle"])
    return Shape(x, y, 5, SHAPE_STATS["pentagon"])


def killer_name_for_shape(shape):
    return {3: "Triangle", 4: "Square", 5: "Pentagon"}.get(shape.sides, "Polygon")


__all__ = [
    "Bullet",
    "Enemy",
    "GameObject",
    "Player",
    "Shape",
    "Wall",
    "build_barrel_polygon",
    "circle_rect_collide",
    "create_shape",
    "default_input_state",
    "get_tank_definition",
    "get_tank_display_name",
    "killer_name_for_shape",
    "resolve_circle_collision",
    "resolve_circle_rect_collision",
]
