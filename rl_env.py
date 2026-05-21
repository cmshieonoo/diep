import math
import random

import numpy as np

from farming_ai import (
    DANGER_DISTANCE,
    EXPLORE_GOAL_REACHED_DISTANCE,
    FARMING_OBSERVATION_SIZE,
    IDEAL_MAX_DISTANCE,
    IDEAL_MIN_DISTANCE,
    apply_farming_upgrades,
    build_farming_observation,
    clamp,
    decode_farming_action,
    desired_navigation_vector,
    get_explore_goal,
    maybe_replan_explore_goal,
    movement_vector,
    normalize_move_action,
)
from settings import FPS
from world import MazeWorld


OBSERVATION_SIZE = FARMING_OBSERVATION_SIZE
MAX_EPISODE_TICKS = FPS * 35


class DiepEnv:
    def __init__(self, max_episode_ticks=MAX_EPISODE_TICKS, curriculum=True):
        self.world = None
        self.ai_player_id = None
        self.max_episode_ticks = max_episode_ticks
        self.curriculum = curriculum
        self.reset_count = 0
        self.episode_ticks = 0
        self.current_info = None
        self.last_reward_breakdown = {}
        self.stuck_ticks = 0
        self.visited_cells = set()

        self.last_xp = 0
        self.last_hp = 0.0
        self.last_level = 1
        self.last_x = 0.0
        self.last_y = 0.0

    def reset(self):
        self.reset_count += 1
        self.episode_ticks = 0

        use_maze = self._use_maze_for_next_episode()
        self.world = MazeWorld(bot_target=0, use_trained_bots=False, use_maze=use_maze)
        ai_player = self.world.add_player("AI_Agent")
        self.ai_player_id = ai_player.entity_id

        self.last_xp = ai_player.xp
        self.last_hp = ai_player.hp
        self.last_level = ai_player.level
        self.last_x = ai_player.x
        self.last_y = ai_player.y
        self.last_reward_breakdown = {}
        self.stuck_ticks = 0
        self.visited_cells = {self._cell_for(ai_player)}
        get_explore_goal(self.world, ai_player, force=True)

        obs = self._get_obs()
        return obs

    def _use_maze_for_next_episode(self):
        if not self.curriculum:
            return True

        if self.reset_count <= 4:
            return False

        maze_probability = clamp((self.reset_count - 4) / 24.0, 0.15, 1.0)
        return random.random() < maze_probability

    def step(self, action):
        player = self.world.players.get(self.ai_player_id)
        if not player or not player.alive:
            return np.zeros(OBSERVATION_SIZE, dtype=np.float32), -1.0, True, {"truncated": False}

        action_index = normalize_move_action(action)
        pre_obs, pre_info = build_farming_observation(self.world, player)
        input_payload = decode_farming_action(self.world, player, action_index, pre_info)

        self.episode_ticks += 1
        self.world.update_input(self.ai_player_id, input_payload)
        self.world.tick()

        player = self.world.players.get(self.ai_player_id)
        if player and player.alive:
            apply_farming_upgrades(player)
            moved_dist = math.hypot(player.x - self.last_x, player.y - self.last_y)
            maybe_replan_explore_goal(
                self.world,
                player,
                moved_dist,
                target_visible=pre_info.get("target_visible", False),
            )

        obs = self._get_obs()
        reward, breakdown = self._get_reward(input_payload.get("_action_index", action_index), pre_info, input_payload)
        self.last_reward_breakdown = breakdown

        done = self._check_done()
        truncated = not done and self.episode_ticks >= self.max_episode_ticks
        info = {
            "truncated": truncated,
            "reward_breakdown": breakdown,
            "episode_ticks": self.episode_ticks,
            "maze": bool(self.world.use_maze),
        }
        if player:
            info["xp"] = int(player.xp)
            info["level"] = int(player.level)
            info["stuck_ticks"] = int(self.stuck_ticks)

        self._advance_reward_memory(player)
        return obs, reward, done, info

    def _get_obs(self):
        player = self.world.players.get(self.ai_player_id)
        if not player or not player.alive:
            self.current_info = None
            return np.zeros(OBSERVATION_SIZE, dtype=np.float32)

        obs, info = build_farming_observation(self.world, player)
        self.current_info = info
        return obs

    def _cell_for(self, player, cell_size=260):
        return int(player.x // cell_size), int(player.y // cell_size)

    def _get_reward(self, action_index, pre_info, input_payload):
        player = self.world.players.get(self.ai_player_id)
        breakdown = {}
        reward = 0.0

        def add(name, value):
            nonlocal reward
            if abs(value) < 1e-9:
                return
            reward += value
            breakdown[name] = round(float(value), 4)

        add("time_cost", -0.0015)

        if not player:
            add("missing_player", -1.0)
            return -1.0, breakdown

        events = [event for event in self.world.last_events if event.get("attacker_id") == self.ai_player_id]
        shape_damage = sum(
            event["amount"]
            for event in events
            if event["type"] == "damage" and event.get("target_kind") == "shape"
        )
        shape_kills = sum(
            1
            for event in events
            if event["type"] == "kill" and event.get("target_kind") == "shape"
        )
        tank_damage = sum(
            event["amount"]
            for event in events
            if event["type"] == "damage" and event.get("target_kind") == "tank"
        )

        xp_gained = max(0, player.xp - self.last_xp)
        level_gained = max(0, player.level - self.last_level)
        hp_lost = max(0.0, self.last_hp - player.hp)

        add("shape_damage", min(0.45, shape_damage * 0.03))
        add("shape_kill", shape_kills * 0.55)
        add("xp_gain", min(0.20, xp_gained * 0.012))
        add("level_gain", level_gained * 0.18)
        add("tank_damage", min(0.25, tank_damage * 0.02))
        add("hp_loss", -min(0.45, hp_lost * 0.12))

        if not player.alive:
            add("death", -1.0)
            return float(clamp(reward, -1.0, 1.0)), breakdown

        current_info = self.current_info or {}
        pre_target = pre_info.get("target")
        pre_dist = pre_info.get("target_dist", float("inf"))
        current_dist = current_info.get("target_dist", float("inf"))
        same_target = pre_info.get("target_id") == current_info.get("target_id")
        moved_dist = math.hypot(player.x - self.last_x, player.y - self.last_y)
        move_x, move_y = movement_vector(action_index)
        tried_to_move = bool(move_x or move_y)
        pre_target_visible = bool(pre_info.get("target_visible"))

        if pre_target is not None and math.isfinite(pre_dist) and pre_target_visible:
            if same_target and math.isfinite(current_dist):
                progress = clamp(pre_dist - current_dist, -80.0, 80.0)
                if pre_dist > IDEAL_MIN_DISTANCE:
                    add("target_progress", progress * 0.0028)

            if IDEAL_MIN_DISTANCE <= pre_dist <= IDEAL_MAX_DISTANCE:
                add("good_range", 0.012)

            if pre_dist < DANGER_DISTANCE:
                add("too_close", -0.18 * ((DANGER_DISTANCE - pre_dist) / DANGER_DISTANCE))

            desired_x, desired_y = desired_navigation_vector(player, pre_info)
            desired_mag = math.hypot(desired_x, desired_y)
            move_mag = math.hypot(move_x, move_y)
            if desired_mag > 0.01:
                if move_mag > 0.01:
                    alignment = ((move_x * desired_x) + (move_y * desired_y)) / (move_mag * desired_mag)
                    add("move_alignment", alignment * 0.025)
                else:
                    add("idle_when_far", -0.015)

            if input_payload["fire"] and pre_info.get("target_visible"):
                add("valid_fire_window", 0.006)
        else:
            pre_goal_dist = pre_info.get("goal_dist", float("inf"))
            current_goal_dist = current_info.get("goal_dist", float("inf"))
            if math.isfinite(pre_goal_dist) and math.isfinite(current_goal_dist):
                goal_progress = clamp(pre_goal_dist - current_goal_dist, -80.0, 80.0)
                add("explore_progress", goal_progress * 0.0035)

            desired_x, desired_y = desired_navigation_vector(player, pre_info)
            desired_mag = math.hypot(desired_x, desired_y)
            move_mag = math.hypot(move_x, move_y)
            if desired_mag > 0.01 and move_mag > 0.01:
                alignment = ((move_x * desired_x) + (move_y * desired_y)) / (move_mag * desired_mag)
                add("explore_alignment", alignment * 0.018)
            elif tried_to_move:
                add("searching", 0.006)

            if current_goal_dist < EXPLORE_GOAL_REACHED_DISTANCE:
                add("explore_goal_reached", 0.08)

            cell = self._cell_for(player)
            if cell not in self.visited_cells:
                self.visited_cells.add(cell)
                add("new_area", 0.018)

        min_wall_probe = pre_info.get("min_wall_probe", 1.0)
        if min_wall_probe < 0.16:
            add("wall_pressure", -(0.16 - min_wall_probe) * 0.35)

        if input_payload.get("_action_blocked"):
            add("blocked_action", -0.025)
        if input_payload.get("_action_repaired"):
            add("repaired_action", -0.008)

        if tried_to_move and moved_dist < 0.35:
            self.stuck_ticks += 1
            add("stuck", -min(0.16, 0.035 + self.stuck_ticks * 0.003))
        else:
            self.stuck_ticks = max(0, self.stuck_ticks - 2)
            if moved_dist > 1.2 and not pre_target_visible:
                add("free_motion", 0.006)

        if not tried_to_move and not input_payload["fire"]:
            add("idle", -0.030 if not pre_target_visible else -0.012)

        hp_ratio = player.hp / player.max_hp
        if hp_ratio < 0.35:
            add("low_hp_risk", -(0.35 - hp_ratio) * 0.08)

        return float(clamp(reward, -1.0, 1.0)), breakdown

    def _advance_reward_memory(self, player):
        if not player:
            return

        self.last_xp = player.xp
        self.last_hp = player.hp
        self.last_level = player.level
        self.last_x = player.x
        self.last_y = player.y

    def _check_done(self):
        player = self.world.players.get(self.ai_player_id)
        return player is None or not player.alive or self.stuck_ticks >= 90
