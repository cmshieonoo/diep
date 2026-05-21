import time
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame

pygame.init()

from world import MazeWorld


def run_headless_benchmark():
    print("1. 게임 월드(맵, 벽)를 생성하는 중...")
    world = MazeWorld(bot_target=50)
    print("2. 게임 월드 생성 완료! 시뮬레이션을 시작합니다.")

    target_ticks = 60000
    start_time = time.perf_counter()

    for tick in range(1, target_ticks + 1):
        world.tick()

        # 답답하지 않게 1,000틱마다 계속 진행 상황을 보여줍니다.
        if tick % 1000 == 0:
            print(f"... {tick} 틱 연산 통과!")

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    tps = target_ticks / elapsed
    speed_multiplier = tps / 60

    print(f"\n[성공] 소요 시간: {elapsed:.2f} 초 (약 {speed_multiplier:.1f} 배속)")


if __name__ == "__main__":
    run_headless_benchmark()