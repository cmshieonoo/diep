import argparse
import json
import math
import os
import random
import socket
import threading
import time
from pathlib import Path

import pygame

from entities import build_barrel_polygon, get_tank_definition
from server import GameServer
from settings import (
    BG_COLOR,
    BLACK,
    BASE_TANK_RADIUS,
    BLUE,
    DEFAULT_BIND,
    DEFAULT_BOT_COUNT,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DARK_BLUE,
    FPS,
    GRID_NORMAL,
    GREY,
    HEIGHT,
    HP_GREEN,
    LEVEL_XP,
    MINIMAP_BORDER,
    MINIMAP_WALL,
    SCORE_COLOR,
    STAT_COLORS,
    STAT_NAMES,
    UI_BG,
    WALL_BASE,
    WALL_BORDER,
    WHITE,
    WIDTH,
    WORLD_HEIGHT,
    WORLD_WIDTH,
    YELLOW,
)
from tank_defs import OFFICIAL_CLASS_TREE

screen = None
clock = None
font_score = None
font_level = None
font_stat = None
font_point = None
font_scoreboard_title = None
font_summary_title = None
font_summary_text = None
font_name = None
font_connect = None
SCRIPT_DIR = Path(__file__).resolve().parent


def encode_message(payload):
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


class NetworkClient:
    def __init__(self, host, port, name):
        self.host = host
        self.port = port
        self.name = name

        self.sock = None
        self.running = False
        self.receiver = None
        self.send_lock = threading.Lock()
        self.state_lock = threading.Lock()

        self.player_id = None
        self.walls = []
        self.snapshot = {
            "tick": 0,
            "tanks": [],
            "bullets": [],
            "shapes": [],
            "leaderboard": [],
        }
        self.error = ""

    def connect(self):
        try:
            self.sock = socket.create_connection((self.host, self.port))
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.running = True
            self.send({"type": "join", "name": self.name})
            self.receiver = threading.Thread(target=self.recv_loop, daemon=True)
            self.receiver.start()
            return True
        except OSError:
            return False

    def send(self, payload):
        if not self.running or self.sock is None:
            return

        try:
            with self.send_lock:
                self.sock.sendall(encode_message(payload))
        except OSError:
            self.error = "Disconnected from server."
            self.close()

    def recv_loop(self):
        buffer = ""
        self.sock.settimeout(0.25)

        try:
            while self.running:
                try:
                    chunk = self.sock.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    self.error = "Disconnected from server."
                    break

                if not chunk:
                    self.error = "Server connection closed."
                    break

                buffer += chunk.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line:
                        continue

                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    self.handle_message(payload)
        finally:
            self.close()

    def handle_message(self, payload):
        with self.state_lock:
            if payload.get("type") == "welcome":
                self.player_id = payload.get("player_id")
                self.walls = payload.get("walls", [])
            elif payload.get("type") == "state":
                self.snapshot = payload

    def get_state(self):
        with self.state_lock:
            return self.player_id, self.walls, self.snapshot, self.error

    def close(self):
        if not self.running:
            return

        self.running = False
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None


def get_font(size):
    font_path = SCRIPT_DIR / "Ubuntu-Bold.ttf"
    if font_path.exists():
        return pygame.font.Font(font_path, size)
    return pygame.font.SysFont("arial", size, bold=True)


def init_client_display():
    global screen
    global clock
    global font_score
    global font_level
    global font_stat
    global font_point
    global font_scoreboard_title
    global font_summary_title
    global font_summary_text
    global font_name
    global font_connect

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("diep.io - Maze Multiplayer")
    clock = pygame.time.Clock()

    font_score = get_font(18)
    font_level = get_font(22)
    font_stat = get_font(16)
    font_point = get_font(28)
    font_scoreboard_title = get_font(26)
    font_summary_title = get_font(32)
    font_summary_text = get_font(22)
    font_name = get_font(14)
    font_connect = get_font(24)


def draw_text_outlined(surface, text, font, text_color, center_x, center_y):
    text_surf = font.render(text, True, text_color)
    outline_surf = font.render(text, True, (0, 0, 0))
    rect = text_surf.get_rect(center=(center_x, center_y - 2))

    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        surface.blit(outline_surf, (rect.x + dx, rect.y + dy))
    surface.blit(text_surf, rect)


def draw_notification(surface, text):
    if not text:
        return

    text_surf = font_point.render(text, True, WHITE)
    padding_x, padding_y = 24, 12
    rect_w = text_surf.get_width() + padding_x * 2
    rect_h = text_surf.get_height() + padding_y * 2
    rect_x = WIDTH // 2 - rect_w // 2
    rect_y = 40

    pygame.draw.rect(surface, (135, 140, 215), (rect_x, rect_y, rect_w, rect_h))
    surface.blit(text_surf, (rect_x + padding_x, rect_y + padding_y))


def format_score(score):
    if score >= 1000:
        return f"{score / 1000:.1f}k"
    return str(int(score))


def draw_wall(surface, wall, cam_x, cam_y):
    rect = pygame.Rect(int(wall["x"] - cam_x), int(wall["y"] - cam_y), wall["w"], wall["h"])
    pygame.draw.rect(surface, WALL_BASE, rect)
    pygame.draw.rect(surface, WALL_BORDER, rect, 6)


def draw_shape(surface, shape, cam_x, cam_y):
    dx = shape["x"] - cam_x
    dy = shape["y"] - cam_y
    radius = shape["radius"]
    sides = shape["sides"]
    angle = shape["angle"]
    color = tuple(shape["color"])
    border_color = tuple(shape["border_color"])

    points = [
        (
            dx + math.cos(angle + (i * 2 * math.pi / sides)) * radius,
            dy + math.sin(angle + (i * 2 * math.pi / sides)) * radius,
        )
        for i in range(sides)
    ]
    pygame.draw.polygon(surface, color, points)
    pygame.draw.polygon(surface, border_color, points, 4)

    if shape["hp"] < shape["max_hp"]:
        ratio = max(0, shape["hp"] / shape["max_hp"])
        bar_w, bar_h = int(radius * 1.5), 8
        bar_x, bar_y = dx - bar_w // 2, dy + radius + 12
        pygame.draw.rect(surface, UI_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        pygame.draw.rect(surface, HP_GREEN, (bar_x, bar_y, int(bar_w * ratio), bar_h), border_radius=3)
        pygame.draw.rect(surface, (0, 0, 0), (bar_x, bar_y, bar_w, bar_h), 2, border_radius=3)


def draw_bullet(surface, bullet, cam_x, cam_y):
    dx = int(bullet["x"] - cam_x)
    dy = int(bullet["y"] - cam_y)
    radius = max(1, int(bullet["radius"]))
    pygame.draw.circle(surface, tuple(bullet["color"]), (dx, dy), radius)
    pygame.draw.circle(surface, tuple(bullet["border_color"]), (dx, dy), radius, 3)


def draw_tank_body(surface, tank_type, center_x, center_y, radius, angle, body_color, border_color):
    tank_def = get_tank_definition(tank_type)
    barrel_scale = radius / BASE_TANK_RADIUS

    for barrel in tank_def["barrels"]:
        points = build_barrel_polygon(center_x, center_y, angle, barrel, scale=barrel_scale)
        pygame.draw.polygon(surface, GREY, points)
        pygame.draw.polygon(surface, BLACK, points, 4)

    if tank_type in {"Smasher", "Auto Smasher", "Landmine", "Spike"}:
        sides = 6 if tank_type != "Spike" else 12
        outer_radius = int(radius) + (8 if tank_type == "Spike" else 0)
        points = [
            (
                center_x + math.cos(angle + (i * 2 * math.pi / sides)) * outer_radius,
                center_y + math.sin(angle + (i * 2 * math.pi / sides)) * outer_radius,
            )
            for i in range(sides)
        ]
        pygame.draw.polygon(surface, body_color, points)
        pygame.draw.polygon(surface, border_color, points, 4)
    else:
        pygame.draw.circle(surface, body_color, (int(center_x), int(center_y)), int(radius))
        pygame.draw.circle(surface, border_color, (int(center_x), int(center_y)), int(radius), 4)


def draw_tank_preview(surface, tank_type, rect, body_color=BLUE, border_color=DARK_BLUE):
    preview_radius = min(rect.width * 0.17, rect.height * 0.22)
    center_x = rect.centerx
    center_y = rect.y + rect.height * 0.42
    draw_tank_body(surface, tank_type, center_x, center_y, preview_radius, -0.35, body_color, border_color)


def draw_tank(surface, tank, cam_x, cam_y):
    if not tank["alive"]:
        return

    dx = int(tank["x"] - cam_x)
    dy = int(tank["y"] - cam_y)
    radius = tank["radius"]
    angle = tank["angle"]
    body_color = tuple(tank["color"])
    border_color = tuple(tank["border_color"])
    draw_tank_body(surface, tank["tank_type"], dx, dy, radius, angle, body_color, border_color)

    draw_text_outlined(surface, tank["name"], font_name, WHITE, dx, dy - int(radius) - 18)

    if tank["hp"] < tank["max_hp"]:
        ratio = max(0, tank["hp"] / tank["max_hp"])
        bar_w, bar_h = 60, 8
        bar_x, bar_y = dx - bar_w // 2, dy + int(radius) + 12
        pygame.draw.rect(surface, UI_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        pygame.draw.rect(surface, HP_GREEN, (bar_x, bar_y, int(bar_w * ratio), bar_h), border_radius=3)
        pygame.draw.rect(surface, (0, 0, 0), (bar_x, bar_y, bar_w, bar_h), 2, border_radius=3)


def draw_hud(surface, player, leaderboard):
    if not player:
        return

    level_bar_w, level_bar_h = 500, 30
    level_x = WIDTH // 2 - level_bar_w // 2
    level_y = HEIGHT - 80

    prev_xp = LEVEL_XP[player["level"] - 1] if player["level"] > 1 else 0
    next_xp = LEVEL_XP[player["level"]] if player["level"] < 45 else player["xp"]
    xp_ratio = 1.0 if player["level"] == 45 else max(0.0, min(1.0, (player["xp"] - prev_xp) / max(1, next_xp - prev_xp)))

    pygame.draw.rect(surface, UI_BG, (level_x, level_y, level_bar_w, level_bar_h), border_radius=11)
    if xp_ratio > 0:
        pygame.draw.rect(surface, YELLOW, (level_x, level_y, int(level_bar_w * xp_ratio), level_bar_h), border_radius=11)
    pygame.draw.rect(surface, BLACK, (level_x, level_y, level_bar_w, level_bar_h), 4, border_radius=11)
    draw_text_outlined(
        surface,
        f"Lvl {player['level']} {player['tank_display_name']}",
        font_level,
        WHITE,
        level_x + level_bar_w // 2,
        level_y + level_bar_h // 2,
    )

    score_bar_w, score_bar_h = 400, 25
    score_x = WIDTH // 2 - score_bar_w // 2
    score_y = level_y - score_bar_h - 8
    top_score = max((entry["xp"] for entry in leaderboard), default=max(player["xp"], 1))
    score_ratio = max(0.0, min(1.0, player["xp"] / max(1, top_score)))

    pygame.draw.rect(surface, UI_BG, (score_x, score_y, score_bar_w, score_bar_h), border_radius=8)
    pygame.draw.rect(surface, SCORE_COLOR, (score_x, score_y, int(score_bar_w * score_ratio), score_bar_h), border_radius=8)
    pygame.draw.rect(surface, BLACK, (score_x, score_y, score_bar_w, score_bar_h), 4, border_radius=8)
    draw_text_outlined(surface, f"Score: {int(player['xp'])}", font_score, WHITE, score_x + score_bar_w // 2, score_y + score_bar_h // 2)


def draw_leaderboard(surface, player_id, leaderboard):
    board_w = 220
    board_x = WIDTH - board_w - 20
    board_y = 20

    draw_text_outlined(surface, "Scoreboard", font_scoreboard_title, WHITE, board_x + board_w // 2, board_y + 10)

    y_offset = board_y + 40
    for tank in leaderboard[:10]:
        bar_h = 24
        bg_color = (152, 238, 169) if tank["id"] == player_id else (170, 190, 180)
        rect = pygame.Rect(board_x, y_offset, board_w, bar_h)
        pygame.draw.rect(surface, bg_color, rect, border_radius=12)
        pygame.draw.rect(surface, BLACK, rect, 3, border_radius=12)

        text = f"{tank['name']} - {format_score(tank['xp'])}"
        draw_text_outlined(surface, text, font_stat, WHITE, board_x + board_w // 2, y_offset + bar_h // 2)

        circle_x = board_x + board_w - 14
        circle_y = y_offset + bar_h // 2
        pygame.draw.circle(surface, tuple(tank["color"]), (circle_x, circle_y), 7)
        pygame.draw.circle(surface, BLACK, (circle_x, circle_y), 7, 2)

        y_offset += bar_h + 4


def draw_minimap(surface, player, walls, tanks):
    if not player:
        return

    map_size = 200
    map_x, map_y = WIDTH - 220, HEIGHT - 220
    pygame.draw.rect(surface, (204, 204, 204), (map_x, map_y, map_size, map_size))

    scale_x = map_size / WORLD_WIDTH
    scale_y = map_size / WORLD_HEIGHT

    for wall in walls:
        wall_x = map_x + int(wall["x"] * scale_x)
        wall_y = map_y + int(wall["y"] * scale_y)
        wall_w = int(wall["w"] * scale_x)
        wall_h = int(wall["h"] * scale_y)
        pygame.draw.rect(surface, MINIMAP_WALL, (wall_x, wall_y, wall_w, wall_h))

    for tank in tanks:
        if not tank["alive"] or tank["id"] == player["id"]:
            continue
        dot_x = map_x + int(tank["x"] * scale_x)
        dot_y = map_y + int(tank["y"] * scale_y)
        pygame.draw.circle(surface, tuple(tank["color"]), (dot_x, dot_y), 3)

    px = map_x + int(player["x"] * scale_x)
    py = map_y + int(player["y"] * scale_y)
    tri_size = 8
    pt1 = (px + math.cos(player["angle"]) * tri_size, py + math.sin(player["angle"]) * tri_size)
    pt2 = (px + math.cos(player["angle"] + 2.45) * tri_size, py + math.sin(player["angle"] + 2.45) * tri_size)
    pt3 = (px + math.cos(player["angle"] - 2.45) * tri_size, py + math.sin(player["angle"] - 2.45) * tri_size)
    pygame.draw.polygon(surface, BLACK, [pt1, pt2, pt3])
    pygame.draw.rect(surface, MINIMAP_BORDER, (map_x, map_y, map_size, map_size), 8)


def draw_upgrade_ui(surface, player):
    rects = []
    if not player or player["stat_points"] <= 0:
        return rects

    start_x = 20
    start_y = HEIGHT - 40 - (8 * 30)
    draw_text_outlined(surface, f"x{player['stat_points']}", font_point, WHITE, start_x + 230, start_y - 10)

    for index in range(8):
        y_pos = start_y + (index * 30)
        color = STAT_COLORS[index]
        level = player["stats"][index]
        bar_w, bar_h, plus_w = 260, 26, 40
        gauge_w = bar_w - plus_w

        bar_rect = pygame.Rect(start_x, y_pos, bar_w, bar_h)
        rects.append((index, bar_rect))

        pygame.draw.rect(surface, UI_BG, bar_rect, border_radius=12)
        if level > 0:
            fill_w = int((gauge_w / 7) * level)
            pygame.draw.rect(surface, color, (start_x, y_pos, fill_w, bar_h), border_top_left_radius=12, border_bottom_left_radius=12)
        if level < 7:
            plus_rect = pygame.Rect(start_x + gauge_w, y_pos, plus_w, bar_h)
            pygame.draw.rect(surface, color, plus_rect, border_top_right_radius=12, border_bottom_right_radius=12)

        pygame.draw.rect(surface, BLACK, bar_rect, 4, border_radius=12)
        pygame.draw.line(surface, BLACK, (start_x + gauge_w, y_pos), (start_x + gauge_w, y_pos + bar_h - 1), 4)

        draw_text_outlined(surface, STAT_NAMES[index], font_stat, WHITE, start_x + gauge_w // 2, y_pos + bar_h // 2)
        key_txt = font_stat.render(f"[{index + 1}]", True, WHITE)
        key_rect = key_txt.get_rect(midright=(start_x + gauge_w - 8, y_pos + bar_h // 2 - 2))
        surface.blit(key_txt, key_rect)

        if level < 7:
            draw_text_outlined(surface, "+", font_point, (0, 0, 0), start_x + gauge_w + plus_w // 2, y_pos + bar_h // 2)

    return rects


def draw_evolution_ui(surface, player):
    rects = []
    if not player or not player["available_tank_upgrades"] or not player["alive"]:
        return rects

    options = player["available_tank_upgrades"]
    panel_x = 18
    panel_y = 18
    card_w = 172
    card_h = 108
    gap = 10
    cols = min(2, len(options))
    rows = (len(options) + cols - 1) // cols
    panel_w = cols * card_w + (cols - 1) * gap + 10
    panel_h = rows * card_h + max(0, rows - 1) * gap + 10
    panel_rect = pygame.Rect(panel_x - 5, panel_y - 5, panel_w, panel_h)
    pygame.draw.rect(surface, (42, 42, 42), panel_rect, border_radius=12)
    pygame.draw.rect(surface, BLACK, panel_rect, 3, border_radius=12)

    option_colors = [
        (109, 188, 255),
        (145, 230, 122),
        (255, 118, 119),
        (255, 231, 105),
    ]

    for index, option in enumerate(options):
        col = index % cols
        row = index // cols
        rect = pygame.Rect(panel_x + col * (card_w + gap), panel_y + row * (card_h + gap), card_w, card_h)
        rects.append((option, rect))

        fill = option_colors[index % len(option_colors)]
        pygame.draw.rect(surface, fill, rect, border_radius=10)
        pygame.draw.rect(surface, BLACK, rect, 4, border_radius=14)
        draw_tank_preview(surface, option, rect)
        draw_text_outlined(surface, option, font_stat, WHITE, rect.centerx, rect.bottom - 30)
        draw_text_outlined(surface, f"[{index + 1}]", font_stat, WHITE, rect.centerx, rect.bottom - 12)

    return rects


def draw_death_overlay(surface, player):
    if not player or player["alive"]:
        return

    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((50, 50, 50, 100))
    surface.blit(overlay, (0, 0))

    draw_text_outlined(surface, "You were killed by", font_summary_text, WHITE, WIDTH // 2, HEIGHT // 2 - 120)
    draw_text_outlined(surface, player["killer_name"] or "Unknown", font_summary_title, WHITE, WIDTH // 2, HEIGHT // 2 - 80)
    draw_text_outlined(surface, f"Score: {int(player['xp'])}", font_summary_text, WHITE, WIDTH // 2, HEIGHT // 2)
    draw_text_outlined(surface, f"Level: {player['level']}", font_summary_text, WHITE, WIDTH // 2, HEIGHT // 2 + 30)
    draw_text_outlined(surface, f"Time: {player['time_alive']}s", font_summary_text, WHITE, WIDTH // 2, HEIGHT // 2 + 60)

    if (pygame.time.get_ticks() // 500) % 2 == 0:
        draw_text_outlined(surface, "Press [ENTER] to Respawn", font_summary_text, (0, 200, 255), WIDTH // 2, HEIGHT // 2 + 130)


def draw_connection_overlay(surface, text):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((40, 40, 40, 180))
    surface.blit(overlay, (0, 0))
    draw_text_outlined(surface, text, font_connect, WHITE, WIDTH // 2, HEIGHT // 2)


def build_class_tree_parent_map():
    parents = {}
    for parent, entry in OFFICIAL_CLASS_TREE.items():
        for child in entry["children"]:
            parents.setdefault(child, []).append(parent)
    return parents


CLASS_TREE_PARENTS = build_class_tree_parent_map()


def draw_class_tree_overlay(surface, player):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((35, 35, 35, 210))
    surface.blit(overlay, (0, 0))

    tier_columns = {
        1: ["Tank"],
        2: ["Twin", "Sniper", "Machine Gun", "Flank Guard"],
        3: [
            "Triple Shot", "Quad Tank", "Twin Flank",
            "Overseer", "Assassin", "Hunter", "Trapper",
            "Destroyer", "Gunner", "Sprayer",
            "Tri-Angle", "Auto 3", "Smasher",
        ],
        4: [
            "Penta Shot", "Spread Shot", "Triplet", "Octo Tank", "Auto 5", "Triple Twin",
            "Battleship", "Overlord", "Manager", "Necromancer", "Overtrapper", "Factory",
            "Ranger", "Stalker", "Predator", "Streamliner", "Mega Trapper", "Tri-Trapper",
            "Gunner Trapper", "Auto Trapper", "Hybrid", "Annihilator", "Skimmer", "Rocketeer",
            "Auto Gunner", "Booster", "Fighter", "Landmine", "Auto Smasher", "Spike",
        ],
    }

    positions = {}
    x_positions = {1: 140, 2: 370, 3: 700, 4: 1235}
    card_sizes = {
        1: (138, 38),
        2: (150, 40),
        3: (164, 38),
        4: (156, 28),
    }

    for tier, names in tier_columns.items():
        card_w, card_h = card_sizes[tier]
        top_margin = 160 if tier == 1 else 70
        if tier == 4:
            gap = 6
        elif tier == 3:
            gap = 8
        else:
            gap = 14

        total_h = len(names) * card_h + max(0, len(names) - 1) * gap
        start_y = max(top_margin, (HEIGHT - total_h) // 2)

        for index, name in enumerate(names):
            rect = pygame.Rect(x_positions[tier], start_y + index * (card_h + gap), card_w, card_h)
            positions[name] = rect

    current_tank = player["tank_display_name"] if player else "Tank"
    active_path = {current_tank}
    frontier = [current_tank]
    while frontier:
        current = frontier.pop()
        for parent in CLASS_TREE_PARENTS.get(current, []):
            if parent not in active_path:
                active_path.add(parent)
                frontier.append(parent)
    if current_tank == "Basic":
        active_path.add("Tank")

    for parent, entry in OFFICIAL_CLASS_TREE.items():
        parent_rect = positions.get(parent)
        if not parent_rect:
            continue
        for child in entry["children"]:
            child_rect = positions.get(child)
            if not child_rect:
                continue
            color = (250, 250, 250) if parent in active_path and child in active_path else (120, 120, 120)
            pygame.draw.line(surface, color, parent_rect.midright, child_rect.midleft, 2)

    for name, rect in positions.items():
        tier = OFFICIAL_CLASS_TREE.get(name, {}).get("tier", 4)
        if name in active_path or (name == "Tank" and current_tank == "Basic"):
            fill = (102, 180, 255)
        elif tier == 1:
            fill = (90, 90, 90)
        elif tier == 2:
            fill = (124, 171, 214)
        elif tier == 3:
            fill = (110, 150, 194)
        else:
            fill = (92, 122, 164)

        pygame.draw.rect(surface, fill, rect, border_radius=10)
        pygame.draw.rect(surface, BLACK, rect, 3, border_radius=10)
        font = font_stat if tier < 4 else font_name
        draw_text_outlined(surface, name, font, WHITE, rect.centerx, rect.centery)

    title = "Class Tree (hold Y)"
    subtitle = f"Current: {current_tank}" if player else "Current: Tank"
    draw_text_outlined(surface, title, font_level, WHITE, WIDTH // 2, 28)
    draw_text_outlined(surface, subtitle, font_stat, WHITE, WIDTH // 2, 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Maze mode with human players and AI bots.")
    parser.add_argument("--join", help="Join a remote host instead of launching a local host.")
    parser.add_argument("--server", action="store_true", help="Run only the dedicated server.")
    parser.add_argument("--bind", default=DEFAULT_BIND, help="Address for the local server to bind to.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port for the game server.")
    parser.add_argument("--bots", type=int, default=DEFAULT_BOT_COUNT, help="How many AI bots the server should maintain.")
    parser.add_argument("--name", help="Player name shown in the scoreboard.")
    return parser.parse_args()


def default_name():
    base = os.getenv("USERNAME") or "Player"
    return f"{base[:10]}-{random.randint(10, 99)}"


def run_client(host, port, name):
    init_client_display()
    client = NetworkClient(host, port, name)
    client.connect()

    auto_fire = False
    auto_spin = False
    notify_text = ""
    notify_timer = 0
    upgrade_rects = []
    evolution_rects = []
    running = True

    while running:
        clock.tick(FPS)
        screen.fill(BG_COLOR)

        if notify_timer > 0:
            notify_timer -= 1

        player_id, walls, snapshot, error_text = client.get_state()
        tanks = snapshot.get("tanks", [])
        leaderboard = snapshot.get("leaderboard", [])
        local_player = next((tank for tank in tanks if tank["id"] == player_id), None)
        available_evolutions = local_player["available_tank_upgrades"] if local_player else []

        target_x = local_player["x"] if local_player else WORLD_WIDTH / 2
        target_y = local_player["y"] if local_player else WORLD_HEIGHT / 2
        cam_x = max(0, min(target_x - WIDTH // 2, WORLD_WIDTH - WIDTH))
        cam_y = max(0, min(target_y - HEIGHT // 2, WORLD_HEIGHT - HEIGHT))

        for x in range(int(cam_x // 35) * 35, int((cam_x + WIDTH) // 35 + 1) * 35, 35):
            screen_x = x - cam_x
            pygame.draw.line(screen, GRID_NORMAL, (screen_x, 0), (screen_x, HEIGHT), 1)

        for y in range(int(cam_y // 35) * 35, int((cam_y + HEIGHT) // 35 + 1) * 35, 35):
            screen_y = y - cam_y
            pygame.draw.line(screen, GRID_NORMAL, (0, screen_y), (WIDTH, screen_y), 1)

        pygame.draw.rect(screen, (140, 140, 140), (-cam_x, -cam_y, WORLD_WIDTH, WORLD_HEIGHT), 25)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                if pygame.K_1 <= event.key <= pygame.K_4 and available_evolutions:
                    option_index = event.key - pygame.K_1
                    if option_index < len(available_evolutions):
                        client.send({"type": "evolve", "tank_type": available_evolutions[option_index]})
                        notify_text = f"Evolution: {available_evolutions[option_index]}"
                        notify_timer = 120
                elif pygame.K_1 <= event.key <= pygame.K_8:
                    stat_idx = event.key - pygame.K_1
                    keys_pressed = pygame.key.get_pressed()
                    client.send({"type": "upgrade", "stat": stat_idx, "bulk": keys_pressed[pygame.K_m] or keys_pressed[pygame.K_u]})

                if event.key == pygame.K_e:
                    auto_fire = not auto_fire
                    notify_text = f"Auto Fire: {'ON' if auto_fire else 'OFF'}"
                    notify_timer = 120

                if event.key == pygame.K_c:
                    auto_spin = not auto_spin
                    notify_text = f"Auto Spin: {'ON' if auto_spin else 'OFF'}"
                    notify_timer = 120

                if event.key == pygame.K_RETURN and local_player and not local_player["alive"]:
                    client.send({"type": "respawn"})

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked_evolution = False
                if local_player and available_evolutions:
                    for tank_type, rect in evolution_rects:
                        if rect.collidepoint(event.pos):
                            client.send({"type": "evolve", "tank_type": tank_type})
                            notify_text = f"Evolution: {tank_type}"
                            notify_timer = 120
                            clicked_evolution = True
                            break

                if not clicked_evolution and local_player and local_player["stat_points"] > 0:
                    for stat_idx, rect in upgrade_rects:
                        if rect.collidepoint(event.pos):
                            client.send({"type": "upgrade", "stat": stat_idx, "bulk": False})
                            break

        if local_player and not local_player["alive"]:
            fire_pressed = False
            aim_angle = local_player["angle"]
        else:
            fire_pressed = pygame.mouse.get_pressed()[0]
            mouse_x, mouse_y = pygame.mouse.get_pos()
            aim_angle = local_player["angle"] if local_player else 0.0
            if local_player:
                aim_angle = math.atan2((mouse_y + cam_y) - local_player["y"], (mouse_x + cam_x) - local_player["x"])

        keys = pygame.key.get_pressed()
        client.send(
            {
                "type": "input",
                "up": bool(keys[pygame.K_w]),
                "down": bool(keys[pygame.K_s]),
                "left": bool(keys[pygame.K_a]),
                "right": bool(keys[pygame.K_d]),
                "fire": fire_pressed,
                "aim_angle": aim_angle,
                "auto_fire": auto_fire,
                "auto_spin": auto_spin,
            }
        )

        for wall in walls:
            draw_wall(screen, wall, cam_x, cam_y)
        for shape in snapshot.get("shapes", []):
            draw_shape(screen, shape, cam_x, cam_y)
        for bullet in snapshot.get("bullets", []):
            draw_bullet(screen, bullet, cam_x, cam_y)
        for tank in tanks:
            draw_tank(screen, tank, cam_x, cam_y)

        draw_hud(screen, local_player, leaderboard)
        draw_leaderboard(screen, player_id, leaderboard)
        draw_minimap(screen, local_player, walls, tanks)
        upgrade_rects = draw_upgrade_ui(screen, local_player)
        evolution_rects = draw_evolution_ui(screen, local_player)
        draw_death_overlay(screen, local_player)

        if notify_timer > 0:
            draw_notification(screen, notify_text)

        if player_id is None:
            draw_connection_overlay(screen, "Connecting to server...")
        elif error_text:
            draw_connection_overlay(screen, error_text)

        if pygame.key.get_pressed()[pygame.K_y]:
            draw_class_tree_overlay(screen, local_player)

        pygame.display.flip()

        if error_text:
            time.sleep(1.0)
            running = False

    client.close()


def main():
    args = parse_args()
    name = args.name or default_name()

    if args.server:
        server = GameServer(args.bind, args.port, bot_count=args.bots)
        print(f"Server listening on {args.bind}:{args.port}")
        server.start(background=False)
        return

    embedded_server = None
    connect_host = args.join

    if not args.join:
        embedded_server = GameServer(args.bind, args.port, bot_count=args.bots)
        embedded_server.start(background=True)
        connect_host = DEFAULT_HOST
        print(f"Local host started on {args.bind}:{args.port}")
        time.sleep(0.3)

    try:
        run_client(connect_host, args.port, name)
    finally:
        if embedded_server is not None:
            embedded_server.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
