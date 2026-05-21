import math
import random
from pathlib import Path

import numpy as np
import pygame

from entities import (
    Bullet,
    Enemy,
    Player,
    Wall,
    circle_rect_collide,
    create_shape,
    default_input_state,
    killer_name_for_shape,
    resolve_circle_collision,
)
from farming_ai import (
    FARMING_MODEL_NAME,
    apply_farming_upgrades,
    build_farming_observation,
    decode_farming_action,
    maybe_replan_explore_goal,
)
from combat_ai import (
    COMBAT_MODEL_NAME,
    apply_combat_upgrades,
    build_combat_observation,
    decode_combat_action,
    find_combat_target,
)
from settings import DEFAULT_BOT_COUNT, FPS, WORLD_HEIGHT, WORLD_WIDTH

WALL_BLOCK_SIZE = 100
SHAPE_TARGET_COUNT = 100
FARMING_POLICY_MODEL_PATH = Path(__file__).resolve().parent / f"{FARMING_MODEL_NAME}.zip"
COMBAT_POLICY_MODEL_PATH = Path(__file__).resolve().parent / f"{COMBAT_MODEL_NAME}.zip"


class WallGrid:
    def __init__(self, walls, cell_size=200):
        self.cell_size = cell_size
        self.cells = {}
        for wall in walls:
            # 벽이 차지하는 모든 격자 칸을 계산하여 등록
            left = wall.rect.left // cell_size
            right = wall.rect.right // cell_size
            top = wall.rect.top // cell_size
            bottom = wall.rect.bottom // cell_size

            for cx in range(left, right + 1):
                for cy in range(top, bottom + 1):
                    if (cx, cy) not in self.cells:
                        self.cells[(cx, cy)] = set()
                    self.cells[(cx, cy)].add(wall)

    def get_nearby(self, obj):
        # 객체의 위치를 기반으로 주변(겹치는) 벽만 가져옴
        left = int((obj.x - obj.radius) // self.cell_size)
        right = int((obj.x + obj.radius) // self.cell_size)
        top = int((obj.y - obj.radius) // self.cell_size)
        bottom = int((obj.y + obj.radius) // self.cell_size)

        nearby_walls = set()
        for cx in range(left, right + 1):
            for cy in range(top, bottom + 1):
                if (cx, cy) in self.cells:
                    nearby_walls.update(self.cells[(cx, cy)])
        return list(nearby_walls)

def generate_walls():
    walls = []
    safe_zone = pygame.Rect(WORLD_WIDTH // 2 - 600, WORLD_HEIGHT // 2 - 600, 1200, 1200)
    attempts = 0

    while len(walls) < 60 and attempts < 2000:
        attempts += 1

        if random.choice([True, False]):
            w_blocks, h_blocks = random.randint(3, 8), random.randint(1, 2)
        else:
            w_blocks, h_blocks = random.randint(1, 2), random.randint(3, 8)

        width = w_blocks * WALL_BLOCK_SIZE
        height = h_blocks * WALL_BLOCK_SIZE
        x = random.randint(0, (WORLD_WIDTH - width) // WALL_BLOCK_SIZE) * WALL_BLOCK_SIZE
        y = random.randint(0, (WORLD_HEIGHT - height) // WALL_BLOCK_SIZE) * WALL_BLOCK_SIZE

        new_rect = pygame.Rect(x, y, width, height)
        if new_rect.colliderect(safe_zone):
            continue

        overlap = False
        for wall in walls:
            logical_rect = pygame.Rect(wall.rect.x, wall.rect.y, wall.rect.width - 6, wall.rect.height - 6)
            if new_rect.colliderect(logical_rect):
                overlap = True
                break

        if not overlap:
            walls.append(Wall(new_rect.x, new_rect.y, new_rect.width + 6, new_rect.height + 6))

    return walls


def get_safe_spawn(walls, radius=30):
    while True:
        spawn_x = random.randint(200, WORLD_WIDTH - 200)
        spawn_y = random.randint(200, WORLD_HEIGHT - 200)
        spawn_rect = pygame.Rect(spawn_x - radius - 10, spawn_y - radius - 10, radius * 2 + 20, radius * 2 + 20)

        if not any(spawn_rect.colliderect(wall.rect) for wall in walls):
            return spawn_x, spawn_y


def create_shape_avoiding_walls(walls, max_attempts=500):
    shape = None

    for _ in range(max_attempts):
        shape = create_shape()
        if not any(circle_rect_collide(shape.x, shape.y, shape.radius, wall.rect) for wall in walls):
            return shape

    if shape is None:
        shape = create_shape()

    spawn_x, spawn_y = get_safe_spawn(walls, radius=shape.radius)
    shape.x = float(spawn_x)
    shape.y = float(spawn_y)
    return shape


class MazeWorld:
    def __init__(
        self,
        bot_target=DEFAULT_BOT_COUNT,
        use_trained_bots=True,
        bot_model_path=None,
        combat_model_path=None,
        use_maze=True,
        shape_target_count=SHAPE_TARGET_COUNT,
    ):
        self.tick_count = 0
        self.bot_target = bot_target
        self.shape_target_count = shape_target_count
        self.players = {}
        self.player_inputs = {}
        self.bots = {}
        self.use_maze = use_maze
        self.walls = generate_walls() if use_maze else []
        self.wall_grid = WallGrid(self.walls)
        self.shapes = [create_shape_avoiding_walls(self.walls) for _ in range(self.shape_target_count)]
        self.bullets = []
        self.last_events = []
        self.farming_policy = self._load_policy(
            use_trained_bots,
            Path(bot_model_path) if bot_model_path else FARMING_POLICY_MODEL_PATH,
            "farming",
        )
        self.combat_policy = self._load_policy(
            use_trained_bots,
            Path(combat_model_path) if combat_model_path else COMBAT_POLICY_MODEL_PATH,
            "combat",
        )
        self.bot_policy = self.farming_policy

        self._next_entity_number = 1
        self._next_player_number = 1

    def _load_policy(self, use_trained_bots, model_path, label):
        if not use_trained_bots or self.bot_target <= 0:
            return None

        if not model_path.exists():
            return None

        try:
            from stable_baselines3 import PPO

            policy = PPO.load(str(model_path))
            print(f"Loaded trained {label} bot model: {model_path.name}")
            return policy
        except Exception as exc:
            print(f"Could not load trained {label} bot model ({model_path.name}): {exc}")
            return None

    def _new_entity_id(self, prefix):
        entity_id = f"{prefix}-{self._next_entity_number}"
        self._next_entity_number += 1
        return entity_id

    def _get_safe_tank_spawn(self, radius=30, max_attempts=1000):
        for _ in range(max_attempts):
            spawn_x, spawn_y = get_safe_spawn(self.walls, radius=radius)
            blocked = False

            for shape in self.shapes:
                if shape.alive and math.hypot(shape.x - spawn_x, shape.y - spawn_y) < shape.radius + radius + 90:
                    blocked = True
                    break

            if blocked:
                continue

            for tank in self._active_tanks():
                if math.hypot(tank.x - spawn_x, tank.y - spawn_y) < tank.radius + radius + 120:
                    blocked = True
                    break

            if not blocked:
                return spawn_x, spawn_y

        return get_safe_spawn(self.walls, radius=radius)

    def _object_id(self, obj):
        return getattr(obj, "entity_id", str(id(obj)))

    def _object_kind(self, obj):
        if hasattr(obj, "is_bot"):
            return "tank"
        if hasattr(obj, "sides"):
            return "shape"
        if isinstance(obj, Bullet):
            return "bullet"
        return type(obj).__name__.lower()

    def _record_damage(self, attacker, target, amount, source):
        if attacker is None or amount <= 0:
            return

        attacker_id = getattr(attacker, "entity_id", None)
        if attacker_id is None:
            return

        self.last_events.append(
            {
                "type": "damage",
                "attacker_id": attacker_id,
                "target_id": self._object_id(target),
                "target_kind": self._object_kind(target),
                "amount": float(amount),
                "source": source,
            }
        )

    def _record_kill(self, killer, victim, xp_award, source):
        if killer is None:
            return

        killer_id = getattr(killer, "entity_id", None)
        if killer_id is None:
            return

        self.last_events.append(
            {
                "type": "kill",
                "attacker_id": killer_id,
                "target_id": self._object_id(victim),
                "target_kind": self._object_kind(victim),
                "xp_award": int(xp_award),
                "source": source,
            }
        )

    def _deal_damage(self, attacker, target, amount, source):
        before_hp = float(getattr(target, "hp", 0.0))
        target.take_damage(amount)
        after_hp = max(0.0, float(getattr(target, "hp", 0.0)))
        actual_damage = max(0.0, before_hp - after_hp)
        self._record_damage(attacker, target, actual_damage, source)
        return actual_damage

    def add_player(self, requested_name):
        name = (requested_name or "").strip()[:20] or f"Player {self._next_player_number}"
        self._next_player_number += 1

        spawn_x, spawn_y = self._get_safe_tank_spawn()
        player = Player(spawn_x, spawn_y, name=name, entity_id=self._new_entity_id("player"))
        player.spawn_tick = self.tick_count

        self.players[player.entity_id] = player
        self.player_inputs[player.entity_id] = default_input_state()
        return player

    def remove_player(self, player_id):
        self.players.pop(player_id, None)
        self.player_inputs.pop(player_id, None)

    def update_input(self, player_id, payload):
        if player_id not in self.player_inputs:
            return

        current = default_input_state()
        current.update(self.player_inputs[player_id])
        current.update(
            {
                "up": bool(payload.get("up")),
                "down": bool(payload.get("down")),
                "left": bool(payload.get("left")),
                "right": bool(payload.get("right")),
                "fire": bool(payload.get("fire")),
                "aim_angle": float(payload.get("aim_angle", 0.0)),
                "auto_fire": bool(payload.get("auto_fire")),
                "auto_spin": bool(payload.get("auto_spin")),
            }
        )
        self.player_inputs[player_id] = current

    def upgrade_player_stat(self, player_id, stat_index, bulk=False):
        player = self.players.get(player_id)
        if not player:
            return

        if bulk:
            while player.stat_points > 0 and player.stats[stat_index] < 7:
                player.upgrade_stat(stat_index)
        else:
            player.upgrade_stat(stat_index)

    def evolve_player(self, player_id, tank_type):
        player = self.players.get(player_id)
        if not player:
            return False
        return player.evolve_tank(tank_type)

    def respawn_player(self, player_id):
        player = self.players.get(player_id)
        if not player or player.alive:
            return

        spawn_x, spawn_y = self._get_safe_tank_spawn(radius=player.radius)
        player.respawn(spawn_x, spawn_y, self.tick_count)

    def _spawn_bullet(self, owner):
        shots = owner.build_shots()
        if not shots:
            return

        for shot in shots:
            self.bullets.append(Bullet(owner, shot))
            owner.apply_recoil(shot["angle"], shot["recoil_multiplier"] / len(shots))
        owner.shoot_cooldown = owner.get_reload_ticks()

    def _ensure_bots(self):
        while len(self.bots) < self.bot_target:
            spawn_x, spawn_y = self._get_safe_tank_spawn()
            bot = Enemy(spawn_x, spawn_y, entity_id=self._new_entity_id("bot"))
            bot.spawn_tick = self.tick_count
            self.bots[bot.entity_id] = bot

    def _update_trained_bots(self):
        alive_bots = [bot for bot in self.bots.values() if bot.alive]
        if not alive_bots:
            return

        updated_bot_ids = set()
        combat_bots = []
        farming_bots = []

        for bot in alive_bots:
            combat_target, _, _ = find_combat_target(self, bot)
            if self.combat_policy is not None and combat_target is not None:
                combat_bots.append(bot)
            elif self.farming_policy is not None:
                farming_bots.append(bot)

        if combat_bots:
            observations = np.array(
                [build_combat_observation(self, bot)[0] for bot in combat_bots],
                dtype=np.float32,
            )
            try:
                actions, _ = self.combat_policy.predict(observations, deterministic=True)
            except Exception as exc:
                print(f"Trained combat bot model failed during prediction: {exc}")
                self.combat_policy = None
                combat_bots = []
                actions = []

            for bot, action in zip(combat_bots, actions):
                apply_combat_upgrades(bot)
                input_state = decode_combat_action(self, bot, action)
                bot.shoot_intent = input_state["fire"]
                nearby_walls = self.wall_grid.get_nearby(bot)
                bot.update(input_state, nearby_walls)
                updated_bot_ids.add(bot.entity_id)

        if farming_bots:
            observations = np.array(
                [build_farming_observation(self, bot)[0] for bot in farming_bots],
                dtype=np.float32,
            )
            try:
                actions, _ = self.farming_policy.predict(observations, deterministic=True)
            except Exception as exc:
                print(f"Trained farming bot model failed during prediction: {exc}")
                self.farming_policy = None
                self.bot_policy = None
                return

            for bot, action in zip(farming_bots, actions):
                apply_farming_upgrades(bot)
                before_x, before_y = bot.x, bot.y
                _, pre_info = build_farming_observation(self, bot)
                input_state = decode_farming_action(self, bot, action)
                bot.shoot_intent = input_state["fire"]
                nearby_walls = self.wall_grid.get_nearby(bot)
                bot.update(input_state, nearby_walls)
                moved_dist = math.hypot(bot.x - before_x, bot.y - before_y)
                maybe_replan_explore_goal(
                    self,
                    bot,
                    moved_dist,
                    target_visible=pre_info.get("target_visible", False),
                )
                updated_bot_ids.add(bot.entity_id)

        return updated_bot_ids

    def _update_rule_bots(self, living_players, skip_bot_ids=None):
        skip_bot_ids = skip_bot_ids or set()
        for bot in self.bots.values():
            if bot.entity_id in skip_bot_ids:
                continue
            bot.update_ai(self.walls, living_players, self.shapes, self.bullets)

    def _update_bot_low_level(self, bot, input_state):
        bot.shoot_intent = input_state["fire"]
        nearby_walls = self.wall_grid.get_nearby(bot)
        bot.update(input_state, nearby_walls)

    def _update_bot_with_farming_policy(self, bot, action):
        apply_farming_upgrades(bot)
        input_state = decode_farming_action(self, bot, action)
        self._update_bot_low_level(bot, input_state)

    def _update_bot_with_combat_policy(self, bot, action):
        apply_combat_upgrades(bot)
        input_state = decode_combat_action(self, bot, action)
        self._update_bot_low_level(bot, input_state)

    def _kill_tank(self, tank, killer_name):
        if tank.is_bot:
            tank.alive = False
            tank.hp = 0
            return

        time_alive = max(0, (self.tick_count - tank.spawn_tick) // FPS)
        tank.mark_dead(killer_name, time_alive)

    def _active_tanks(self):
        return [player for player in self.players.values() if player.alive] + [
            bot for bot in self.bots.values() if bot.alive
        ]

    def _handle_all_collisions(self):
        active_tanks = self._active_tanks()
        CELL_SIZE = 200  # 맵을 나눌 격자의 크기
        spatial_grid = {}

        # 1. 살아있는 도형과 탱크를 그리드(격자)에 할당
        def add_to_grid(obj, obj_type):
            min_x = int((obj.x - obj.radius) // CELL_SIZE)
            max_x = int((obj.x + obj.radius) // CELL_SIZE)
            min_y = int((obj.y - obj.radius) // CELL_SIZE)
            max_y = int((obj.y + obj.radius) // CELL_SIZE)

            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    cell = (x, y)
                    if cell not in spatial_grid:
                        spatial_grid[cell] = []
                    spatial_grid[cell].append((obj, obj_type))

        for shape in self.shapes:
            if shape.alive:
                add_to_grid(shape, "shape")
        for tank in active_tanks:
            if tank.alive:
                add_to_grid(tank, "tank")
        for bullet in self.bullets:
            if bullet.alive:
                add_to_grid(bullet, "bullet")

        # 2. 물리적 몸체 충돌 처리 (탱크 및 도형 간)
        checked_pairs = set()
        for cell_objs in spatial_grid.values():
            n = len(cell_objs)
            for i in range(n):
                obj_a, type_a = cell_objs[i]
                if not obj_a.alive: continue

                for j in range(i + 1, n):
                    obj_b, type_b = cell_objs[j]
                    if not obj_b.alive: continue

                    # 중복 검사 방지
                    pair_id = frozenset([id(obj_a), id(obj_b)])
                    if pair_id in checked_pairs: continue
                    checked_pairs.add(pair_id)

                    if type_a == "bullet" or type_b == "bullet":
                        continue

                    if not obj_a.collides_with(obj_b): continue

                    # 도형 vs 도형
                    if type_a == "shape" and type_b == "shape":
                        resolve_circle_collision(obj_a, obj_b)

                    # 탱크 vs 탱크
                    elif type_a == "tank" and type_b == "tank":
                        resolve_circle_collision(obj_a, obj_b)
                        self._deal_damage(obj_b, obj_a, 3 + (obj_b.stats[2] * 2), "body")
                        self._deal_damage(obj_a, obj_b, 3 + (obj_a.stats[2] * 2), "body")

                        if obj_a.hp <= 0:
                            xp_award = 1500 if not obj_a.is_bot else 1000
                            self._kill_tank(obj_a, obj_b.name)
                            obj_b.add_xp(xp_award)
                            self._record_kill(obj_b, obj_a, xp_award, "body")
                        if obj_b.hp <= 0:
                            xp_award = 1500 if not obj_b.is_bot else 1000
                            self._kill_tank(obj_b, obj_a.name)
                            obj_a.add_xp(xp_award)
                            self._record_kill(obj_a, obj_b, xp_award, "body")

                    # 탱크 vs 도형
                    else:
                        tank = obj_a if type_a == "tank" else obj_b
                        shape = obj_b if type_a == "tank" else obj_a

                        resolve_circle_collision(tank, shape)
                        self._deal_damage(None, tank, 2, "body_collision")
                        self._deal_damage(tank, shape, 2 + (tank.stats[2] * 3), "body")

                        if shape.hp <= 0:
                            xp_award = shape.xp_value
                            shape.alive = False
                            tank.add_xp(xp_award)
                            self._record_kill(tank, shape, xp_award, "body")
                        if tank.hp <= 0:
                            self._kill_tank(tank, killer_name_for_shape(shape))
                            break

        # 3. 총알 충돌 처리 (동일한 그리드 재활용)
        processed_bullet_pairs = set()
        for bullet in self.bullets:
            if not bullet.alive: continue

            min_x = int((bullet.x - bullet.radius) // CELL_SIZE)
            max_x = int((bullet.x + bullet.radius) // CELL_SIZE)
            min_y = int((bullet.y - bullet.radius) // CELL_SIZE)
            max_y = int((bullet.y + bullet.radius) // CELL_SIZE)

            bullet_checked = set()
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    if not bullet.alive: break

                    cell = (x, y)
                    if cell not in spatial_grid: continue

                    for obj, type_tag in spatial_grid[cell]:
                        if not bullet.alive: break
                        if not obj.alive: continue

                        if id(obj) in bullet_checked: continue
                        bullet_checked.add(id(obj))

                        if type_tag == "shape":
                            if bullet.collides_with(obj):
                                damage = bullet.hit_target()
                                self._deal_damage(bullet.owner, obj, damage, "bullet")
                                if obj.hp <= 0:
                                    xp_award = obj.xp_value
                                    obj.alive = False
                                    bullet.owner.add_xp(xp_award)
                                    self._record_kill(bullet.owner, obj, xp_award, "bullet")

                        elif type_tag == "tank":
                            if obj is bullet.owner: continue
                            if bullet.collides_with(obj):
                                resistance = 1.0 + (obj.stats[2] * 0.35)
                                damage = bullet.hit_target(resistance)
                                self._deal_damage(bullet.owner, obj, damage, "bullet")

                                if obj.hp <= 0:
                                    self._kill_tank(obj, bullet.owner.name)
                                    if obj.is_bot:
                                        xp_award = 1000
                                    elif obj.entity_id in self.players:
                                        xp_award = 1500
                                    else:
                                        xp_award = 0

                                    if xp_award > 0:
                                        bullet.owner.add_xp(xp_award)
                                        self._record_kill(bullet.owner, obj, xp_award, "bullet")

                        elif type_tag == "bullet":
                            if obj is bullet or obj.owner_id == bullet.owner_id:
                                continue

                            pair_id = frozenset((id(bullet), id(obj)))
                            if pair_id in processed_bullet_pairs:
                                continue

                            processed_bullet_pairs.add(pair_id)
                            if bullet.collides_with(obj):
                                bullet.spend_hp(obj.get_bullet_collision_damage())
                                obj.spend_hp(bullet.get_bullet_collision_damage())

    def tick(self):
        self.tick_count += 1
        self.last_events = []
        self._ensure_bots()

        for player_id, player in self.players.items():
            # 주변 벽만 가져와서 충돌 검사
            nearby_walls = self.wall_grid.get_nearby(player)
            player.update(self.player_inputs.get(player_id), nearby_walls)
            if player.wants_to_fire(self.player_inputs.get(player_id, {})):
                self._spawn_bullet(player)

        for shape in self.shapes:
            if shape.alive:
                nearby_walls = self.wall_grid.get_nearby(shape)
                shape.update(nearby_walls)

        for bullet in self.bullets:
            if bullet.alive:
                nearby_walls = self.wall_grid.get_nearby(bullet)
                bullet.update(nearby_walls)

        living_players = [player for player in self.players.values() if player.alive]
        updated_bot_ids = set()
        if self.farming_policy is not None or self.combat_policy is not None:
            updated_bot_ids = self._update_trained_bots() or set()

        self._update_rule_bots(living_players, skip_bot_ids=updated_bot_ids)

        for bot in self.bots.values():
            if bot.wants_to_fire_ai():
                self._spawn_bullet(bot)

        self._handle_all_collisions()

        self.shapes = [shape for shape in self.shapes if shape.alive]
        self.bullets = [bullet for bullet in self.bullets if bullet.alive]
        self.bots = {bot_id: bot for bot_id, bot in self.bots.items() if bot.alive}

        while len(self.shapes) < self.shape_target_count:
            self.shapes.append(create_shape_avoiding_walls(self.walls))

    def build_welcome(self, player_id):
        return {
            "type": "welcome",
            "player_id": player_id,
            "world_width": WORLD_WIDTH,
            "world_height": WORLD_HEIGHT,
            "walls": [wall.to_state() for wall in self.walls],
        }

    def build_snapshot_for_player(self, player_id):
        player = self.players.get(player_id)
        if not player:
            return None

        # 화면 해상도(1920x1080)를 넉넉하게 커버하는 시야 반경 설정
        VIEW_DISTANCE = 1200

        # 시야 내에 있는지 확인하는 헬퍼 함수
        def in_view(obj):
            return math.hypot(obj.x - player.x, obj.y - player.y) < VIEW_DISTANCE

        tanks = [p.to_state() for p in self.players.values() if in_view(p)] + \
                [bot.to_state() for bot in self.bots.values() if bot.alive and in_view(bot)]

        # 리더보드용 데이터 (KeyError 방지를 위해 to_state() 전체 정보 유지, 단 상위 10명만)
        all_tanks = list(self.players.values()) + [b for b in self.bots.values() if b.alive]
        top_10_tanks = sorted(all_tanks, key=lambda x: x.xp, reverse=True)[:10]
        leaderboard = [t.to_state() for t in top_10_tanks]

        return {
            "type": "state",
            "tick": self.tick_count,
            "tanks": tanks,
            "bullets": [bullet.to_state() for bullet in self.bullets if in_view(bullet)],
            "shapes": [shape.to_state() for shape in self.shapes if in_view(shape)],
            "leaderboard": leaderboard,
        }
