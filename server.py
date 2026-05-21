import asyncio
import json
import uuid
from settings import DEFAULT_BOT_COUNT, FPS, STATE_BROADCAST_FPS, WORLD_WIDTH, WORLD_HEIGHT
from world import MazeWorld


class GameServer:
    def __init__(self, bind_address, port, bot_count=DEFAULT_BOT_COUNT):
        self.bind_address = bind_address
        self.port = port
        self.bot_count = bot_count

        self.running = False
        self.clients = {}  # {websocket_connection: player_id} 구조로 클라이언트 매핑
        self.world = MazeWorld()
        self.server_task = None

        # 세계 인프라 및 기본 봇 생성
        self.world.initialize_maze()
        for i in range(self.bot_count):
            self.world.spawn_bot(f"Bot-{random_id()[:4]}")

    def start(self, background=False):
        import websockets
        self.running = True

        # 이벤트 루프 가져오기
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # 서버 비동기 태스크 생성
        if background:
            self.server_task = loop.create_task(self.run_server_async())
        else:
            print(f"Starting Dedicated WebSocket Server on {self.bind_address}:{self.port}")
            loop.run_until_complete(self.run_server_async())

    def stop(self):
        self.running = False
        if self.server_task:
            self.server_task.cancel()

    async def run_server_async(self):
        import websockets
        # 외부 수신을 감당할 웹소켓 서버 바인딩
        async with websockets.serve(self.handler, self.bind_address, self.port):
            # 물리 엔진 연동 및 상태 브로드캐스트 전송 루프 동시 가동
            await asyncio.gather(self.physics_loop(), self.broadcast_loop())

    async def handler(self, websocket, path):
        player_id = None
        try:
            async for message in websocket:
                if not self.running:
                    break

                payload = json.loads(message)
                msg_type = payload.get("type")

                if msg_type == "join":
                    # 유저 추가 및 고유 세션 할당
                    player_name = payload.get("name", "Unknown")[:12]
                    player_id = str(uuid.uuid4())
                    self.clients[websocket] = player_id
                    self.world.spawn_player(player_id, player_name)

                    # 방 구조(벽 정보) 및 초기 환영 패킷 전송
                    walls_data = [{"x": w.x, "y": w.y, "w": w.w, "h": w.h} for w in self.world.walls]
                    welcome = {"type": "welcome", "player_id": player_id, "walls": walls_data}
                    await websocket.send(json.dumps(welcome))

                elif msg_type == "input" and player_id:
                    # 유저 조작 패킷 업데이트
                    self.world.update_player_input(player_id, payload)

                elif msg_type == "upgrade" and player_id:
                    # 능력치 강화 시스템 반영
                    self.world.upgrade_player_stat(player_id, payload.get("stat"), payload.get("bulk", False))

                elif msg_type == "evolve" and player_id:
                    # 전직 트리 반영
                    self.world.evolve_player_tank(player_id, payload.get("tank_type"))

                elif msg_type == "respawn" and player_id:
                    # 부활 처리
                    self.world.respawn_player(player_id)

        except Exception as e:
            print(f"Session disconnected with exception: {e}")
        finally:
            # 커넥션 파기 시 메모리 해제
            if websocket in self.clients:
                p_id = self.clients[websocket]
                self.world.remove_player(p_id)
                del self.clients[websocket]

    async def physics_loop(self):
        """서버 내부 게임 로직 주기적 갱신 루프 (FPS 기반)"""
        dt = 1.0 / FPS
        while self.running:
            start_time = asyncio.get_event_loop().time()

            # 세계 상태 업데이트 (물리, 탄막, 충돌 등)
            self.world.update(dt)

            # 오차 보정 계산을 포함한 정밀 딜레이
            elapsed = asyncio.get_event_loop().time() - start_time
            await asyncio.sleep(max(0.0, dt - elapsed))

    async def broadcast_loop(self):
        """모든 클라이언트에게 실시간 월드 데이터 배송 루프"""
        dt = 1.0 / STATE_BROADCAST_FPS
        while self.running:
            start_time = asyncio.get_event_loop().time()

            if self.clients:
                # MazeWorld로부터 현재 프레임 직렬화 상태 가져오기
                state_data = self.world.get_serialized_state()
                state_data["type"] = "state"
                message = json.dumps(state_data, separators=(",", ":"))

                # 접속 중인 모든 웹소켓에 동시 브로드캐스트 발송
                tasks = [ws.send(message) for ws in self.clients.keys()]
                await asyncio.gather(*tasks, return_exceptions=True)

            elapsed = asyncio.get_event_loop().time() - start_time
            await asyncio.sleep(max(0.0, dt - elapsed))


def random_id():
    return str(uuid.uuid4())