import os

import numpy as np
import pygame
from stable_baselines3 import PPO

from train_ai import GymDiepEnv, MODEL_PATH


def run_spectator():
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    pygame.init()
    screen = pygame.display.set_mode((1000, 800))
    pygame.display.set_caption("AI Spectator")
    clock = pygame.time.Clock()

    print("Loading the environment and trained model.")
    env = GymDiepEnv()

    if not MODEL_PATH.exists():
        print(f"Error: {MODEL_PATH.name} was not found. Train the model first.")
        return

    model = PPO.load(str(MODEL_PATH))

    obs, _ = env.reset()
    world = env.env.world
    running = True

    print("Starting spectator mode.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        screen.fill((204, 204, 204))

        player = world.players.get(env.env.ai_player_id)
        if player:
            cam_x, cam_y = player.x - 500, player.y - 400

            for shape in world.shapes:
                if shape.alive:
                    pygame.draw.circle(
                        screen,
                        (255, 232, 105),
                        (int(shape.x - cam_x), int(shape.y - cam_y)),
                        int(shape.radius),
                    )

            for bullet in world.bullets:
                if bullet.alive:
                    pygame.draw.circle(
                        screen,
                        (0, 178, 225),
                        (int(bullet.x - cam_x), int(bullet.y - cam_y)),
                        int(bullet.radius),
                    )

            px, py = int(player.x - cam_x), int(player.y - cam_y)
            pygame.draw.circle(screen, (0, 178, 225), (px, py), int(player.radius))

            angle = player.angle if hasattr(player, "angle") else 0.0
            end_x = px + np.cos(angle) * 30
            end_y = py + np.sin(angle) * 30
            pygame.draw.line(screen, (85, 85, 85), (px, py), (end_x, end_y), 5)

        pygame.display.flip()
        clock.tick(60)

        if done or truncated:
            obs, _ = env.reset()
            world = env.env.world

    pygame.quit()


if __name__ == "__main__":
    run_spectator()
