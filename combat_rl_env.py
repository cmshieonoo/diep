import math
import random

import numpy as np

from combat_ai import (
    COMBAT_DANGER_DISTANCE,
    COMBAT_IDEAL_MAX_DISTANCE,
    COMBAT_IDEAL_MIN_DISTANCE,
    COMBAT_OBSERVATION_SIZE,
    apply_combat_upgrades,
    build_combat_observation,
    clamp,
    decode_combat_action,
    desired_combat_vector,
    movement_vector,
    normalize_move_action,
    scripted_combat_input,
)
from entities import circle_rect_collide
from settings import FPS, LEVEL_XP, WORLD_HEIGHT, WORLD_WIDTH
from world import MazeWorld


OBSERVATION_SIZE = COMBAT_OBSERVATION_SIZE
MAX_EPISODE_TICKS = FPS * 35


class CombatDiepEnv:
    def __init__(self, max_episode_ticks=MAX_EPISODE_TICKS, curriculum=True):
        self.world = None
        self.ai_player_id = None
        self.opponent_id = None
        self.max_episode_ticks = max_episode_ticks
        self.curriculum = curriculum
        self.reset_count = 0
        self.episode_ticks = 0
        self.strafe_sign = 1.0
        self.current_info = None
        self.last_reward_breakdown = {}

        self.last_agent_hp = 0.0
        self.last_opponent_hp = 0.0
        self.last_agent_x = 0.0
        self.last_agent_y = 0.0
        self.last_agent_xp = 0

    def reset(self):
        self.reset_count += 1
        self.episode_ticks = 0
        self.strafe_sign = random.choice([-1.0, 1.0])

        use_maze = self._use_maze_for_next_episode()
        self.world = MazeWorld(
            bot_target=0,
            use_trained_bots=False,
            use_maze=use_maze,
            shape_target_count=0,
        )

        agent = self.world.add_player("Combat_Agent")
        opponent = self.world.add_player("Sparring_Target")
        self.ai_player_id = agent.entity_id
        self.opponent_id = opponent.entity_id

        self._place_duelists(agent, opponent)
        self._prepare_tank(agent)
        self._prepare_tank(opponent)

        self.last_agent_hp = agent.hp
        self.last_opponent_hp = opponent.hp
        self.last_agent_x = agent.x
        self.last_agent_y = agent.y
        self.last_agent_xp = agent.xp
        self.last_reward_breakdown = {}

        return self._get_obs()

    def _use_maze_for_next_episode(self):
        if not self.curriculum:
            return True
        if self.reset_count <= 8:
            return False
        maze_probability = clamp((self.reset_count - 8) / 32.0, 0.10, 1.0)
        return random.random() < maze_probability

    def _prepare_tank(self, tank):
        level = random.randint(12, 22)
        tank._set_progression_for_level(level, xp=LEVEL_XP[level - 1])
        tank.update_base_stats()
        tank.hp = tank.max_hp
        apply_combat_upgrades(tank)

    def _position_is_safe(self, x, y, radius):
        if x < radius or x > WORLD_WIDTH - radius or y < radius or y > WORLD_HEIGHT - radius:
            return False
        return not any(circle_rect_collide(x, y, radius + 15, wall.rect) for wall in self.world.walls)

    def _place_duelists(self, agent, opponent):
        for _ in range(500):
            center_x, center_y = self.world._get_safe_tank_spawn(radius=agent.radius)
            angle = random.uniform(-math.pi, math.pi)
            distance = random.uniform(540.0, 820.0)
            opponent_x = center_x + math.cos(angle) * distance
            opponent_y = center_y + math.sin(angle) * distance

            if not self._position_is_safe(opponent_x, opponent_y, opponent.radius):
                continue

            agent.x = float(center_x)
            agent.y = float(center_y)
            opponent.x = float(opponent_x)
            opponent.y = float(opponent_y)
            agent.angle = angle
            opponent.angle = angle + math.pi
            return

        agent.x = WORLD_WIDTH / 2 - 350
        agent.y = WORLD_HEIGHT / 2
        opponent.x = WORLD_WIDTH / 2 + 350
        opponent.y = WORLD_HEIGHT / 2
        agent.angle = 0.0
        opponent.angle = math.pi

    def step(self, action):
        agent = self.world.players.get(self.ai_player_id)
        opponent = self.world.players.get(self.opponent_id)
        if not agent or not agent.alive:
            return np.zeros(OBSERVATION_SIZE, dtype=np.float32), -1.0, True, {"truncated": False}
        if not opponent or not opponent.alive:
            return np.zeros(OBSERVATION_SIZE, dtype=np.float32), 1.0, True, {"truncated": False}

        action_index = normalize_move_action(action)
        pre_obs, pre_info = build_combat_observation(self.world, agent, candidates=[opponent])
        agent_input = decode_combat_action(self.world, agent, action_index, pre_info)
        opponent_input = scripted_combat_input(
            self.world,
            opponent,
            agent,
            self.episode_ticks,
            strafe_sign=-self.strafe_sign,
        )

        self.episode_ticks += 1
        self.world.update_input(self.ai_player_id, agent_input)
        self.world.update_input(self.opponent_id, opponent_input)
        self.world.tick()

        agent = self.world.players.get(self.ai_player_id)
        opponent = self.world.players.get(self.opponent_id)
        if agent and agent.alive:
            apply_combat_upgrades(agent)
        if opponent and opponent.alive:
            apply_combat_upgrades(opponent)

        obs = self._get_obs()
        reward, breakdown = self._get_reward(action_index, pre_info, agent_input)
        self.last_reward_breakdown = breakdown

        done = self._check_done()
        truncated = not done and self.episode_ticks >= self.max_episode_ticks
        info = {
            "truncated": truncated,
            "reward_breakdown": breakdown,
            "episode_ticks": self.episode_ticks,
            "maze": bool(self.world.use_maze),
        }
        if agent:
            info["agent_hp"] = round(agent.hp, 2)
            info["agent_xp"] = int(agent.xp)
        if opponent:
            info["opponent_hp"] = round(opponent.hp, 2)

        self._advance_reward_memory(agent, opponent)
        return obs, reward, done, info

    def _get_obs(self):
        agent = self.world.players.get(self.ai_player_id)
        opponent = self.world.players.get(self.opponent_id)
        if not agent or not agent.alive or not opponent or not opponent.alive:
            self.current_info = None
            return np.zeros(OBSERVATION_SIZE, dtype=np.float32)

        obs, info = build_combat_observation(self.world, agent, candidates=[opponent])
        self.current_info = info
        return obs

    def _get_reward(self, action_index, pre_info, agent_input):
        agent = self.world.players.get(self.ai_player_id)
        opponent = self.world.players.get(self.opponent_id)
        breakdown = {}
        reward = 0.0

        def add(name, value):
            nonlocal reward
            if abs(value) < 1e-9:
                return
            reward += value
            breakdown[name] = round(float(value), 4)

        add("time_cost", -0.002)

        if not agent:
            add("missing_agent", -1.0)
            return -1.0, breakdown

        events = [event for event in self.world.last_events if event.get("attacker_id") == self.ai_player_id]
        tank_damage = sum(
            event["amount"]
            for event in events
            if event["type"] == "damage" and event.get("target_kind") == "tank"
        )
        tank_kills = sum(
            1
            for event in events
            if event["type"] == "kill" and event.get("target_kind") == "tank"
        )

        agent_damage_taken = max(0.0, self.last_agent_hp - agent.hp)
        xp_gained = max(0, agent.xp - self.last_agent_xp)

        add("tank_damage", min(0.60, tank_damage * 0.030))
        add("tank_kill", tank_kills * 1.0)
        add("xp_gain", min(0.20, xp_gained * 0.0008))
        add("damage_taken", -min(0.65, agent_damage_taken * 0.030))

        if opponent is None or not opponent.alive:
            add("win", 1.0)
            return float(clamp(reward, -1.0, 1.0)), breakdown

        if not agent.alive:
            add("death", -1.0)
            return float(clamp(reward, -1.0, 1.0)), breakdown

        current_info = self.current_info or {}
        pre_dist = pre_info.get("target_dist", float("inf"))
        current_dist = current_info.get("target_dist", float("inf"))
        same_target = pre_info.get("target_id") == current_info.get("target_id")

        if same_target and math.isfinite(pre_dist) and math.isfinite(current_dist):
            progress = clamp(pre_dist - current_dist, -70.0, 70.0)
            if pre_dist > COMBAT_IDEAL_MAX_DISTANCE:
                add("chase_progress", progress * 0.0012)

        if pre_info.get("target_visible") and COMBAT_IDEAL_MIN_DISTANCE <= pre_dist <= COMBAT_IDEAL_MAX_DISTANCE:
            add("good_duel_range", 0.004)

        if math.isfinite(pre_dist) and pre_dist < COMBAT_DANGER_DISTANCE:
            add("ramming_risk", -0.20 * ((COMBAT_DANGER_DISTANCE - pre_dist) / COMBAT_DANGER_DISTANCE))

        if agent_input["fire"] and pre_info.get("target_visible"):
            add("valid_fire_window", 0.002)

        move_x, move_y = movement_vector(action_index)
        desired_x, desired_y = desired_combat_vector(pre_info, strafe_sign=self.strafe_sign)
        move_mag = math.hypot(move_x, move_y)
        desired_mag = math.hypot(desired_x, desired_y)
        if desired_mag > 0.01:
            if move_mag > 0.01:
                alignment = ((move_x * desired_x) + (move_y * desired_y)) / (move_mag * desired_mag)
                add("positioning", alignment * 0.006)
            else:
                add("idle_positioning", -0.006)

        threat_before = pre_info.get("bullet_threat_score", 0.0)
        threat_after = current_info.get("bullet_threat_score", 0.0)
        if threat_before > 0.05:
            add("threat_reduced", clamp(threat_before - threat_after, -0.2, 0.2) * 0.08)

        moved_dist = math.hypot(agent.x - self.last_agent_x, agent.y - self.last_agent_y)
        if move_mag > 0.01 and moved_dist < 0.35:
            add("stuck", -0.05)

        hp_ratio = agent.hp / agent.max_hp
        if hp_ratio < 0.30:
            add("low_hp_risk", -(0.30 - hp_ratio) * 0.12)

        return float(clamp(reward, -1.0, 1.0)), breakdown

    def _advance_reward_memory(self, agent, opponent):
        if agent:
            self.last_agent_hp = agent.hp
            self.last_agent_x = agent.x
            self.last_agent_y = agent.y
            self.last_agent_xp = agent.xp
        if opponent:
            self.last_opponent_hp = opponent.hp

    def _check_done(self):
        agent = self.world.players.get(self.ai_player_id)
        opponent = self.world.players.get(self.opponent_id)
        return agent is None or opponent is None or not agent.alive or not opponent.alive
