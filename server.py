import json
import socket
import threading
import time

from settings import DEFAULT_BOT_COUNT, FPS, STATE_BROADCAST_FPS
from world import MazeWorld


def encode_message(payload):
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


class ClientConnection:
    def __init__(self, server, sock, address):
        self.server = server
        self.sock = sock
        self.address = address
        self.player_id = None
        self.alive = True
        self.send_lock = threading.Lock()
        self.thread = threading.Thread(target=self.recv_loop, daemon=True)

    def start(self):
        self.thread.start()

    def send(self, payload):
        if not self.alive:
            return

        try:
            with self.send_lock:
                self.sock.sendall(encode_message(payload))
        except OSError:
            self.close()

    def recv_loop(self):
        buffer = ""
        self.sock.settimeout(0.25)

        try:
            while self.alive and self.server.running:
                try:
                    chunk = self.sock.recv(65536)
                except socket.timeout:
                    continue

                if not chunk:
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
        message_type = payload.get("type")

        if message_type == "join" and self.player_id is None:
            with self.server.world_lock:
                player = self.server.world.add_player(payload.get("name", "Player"))
                self.player_id = player.entity_id
                welcome = self.server.world.build_welcome(self.player_id)
                snapshot = self.server.world.build_snapshot_for_player(self.player_id)

            self.send(welcome)
            self.send(snapshot)
            return

        if self.player_id is None:
            return

        if message_type == "input":
            with self.server.world_lock:
                self.server.world.update_input(self.player_id, payload)
            return

        if message_type == "upgrade":
            stat_index = int(payload.get("stat", -1))
            if 0 <= stat_index < 8:
                with self.server.world_lock:
                    self.server.world.upgrade_player_stat(self.player_id, stat_index, bool(payload.get("bulk")))
            return

        if message_type == "evolve":
            tank_type = payload.get("tank_type")
            if isinstance(tank_type, str):
                with self.server.world_lock:
                    self.server.world.evolve_player(self.player_id, tank_type)
            return

        if message_type == "respawn":
            with self.server.world_lock:
                self.server.world.respawn_player(self.player_id)

    def close(self):
        if not self.alive:
            return

        self.alive = False
        try:
            self.sock.close()
        except OSError:
            pass

        self.server.remove_client(self)


class GameServer:
    def __init__(self, host, port, bot_count=DEFAULT_BOT_COUNT):
        self.host = host
        self.port = port
        self.bot_count = bot_count

        self.world = MazeWorld(bot_target=bot_count)
        self.world_lock = threading.Lock()

        self.server_socket = None
        self.clients = []
        self.clients_lock = threading.Lock()

        self.running = False
        self.accept_thread = None
        self.loop_thread = None

    def start(self, background=False):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen()
        self.server_socket.settimeout(0.25)

        self.running = True
        self.accept_thread = threading.Thread(target=self.accept_loop, daemon=True)
        self.accept_thread.start()

        if background:
            self.loop_thread = threading.Thread(target=self.run_loop, daemon=True)
            self.loop_thread.start()
            return

        self.run_loop()

    def accept_loop(self):
        while self.running:
            try:
                client_sock, address = self.server_socket.accept()
                client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                connection = ClientConnection(self, client_sock, address)
                with self.clients_lock:
                    self.clients.append(connection)
                connection.start()

            except socket.timeout:
                continue
            except OSError:
                break

    def remove_client(self, connection):
        with self.clients_lock:
            if connection in self.clients:
                self.clients.remove(connection)

        if connection.player_id is not None:
            with self.world_lock:
                self.world.remove_player(connection.player_id)

    def broadcast(self, payload):
        with self.clients_lock:
            clients = list(self.clients)

        for client in clients:
            client.send(payload)

    def _collect_snapshot_messages(self):
        with self.clients_lock:
            clients = [client for client in self.clients if client.player_id is not None]

        with self.world_lock:
            messages = []
            for client in clients:
                snapshot = self.world.build_snapshot_for_player(client.player_id)
                if snapshot is not None:
                    messages.append((client, snapshot))

        return messages

    def run_loop(self):
        tick_length = 1.0 / FPS
        broadcast_every = max(1, FPS // STATE_BROADCAST_FPS)

        try:
            while self.running:
                started_at = time.perf_counter()

                with self.world_lock:
                    self.world.tick()
                    should_broadcast = self.world.tick_count % broadcast_every == 0

                if should_broadcast:
                    for client, snapshot in self._collect_snapshot_messages():
                        client.send(snapshot)

                target_time = started_at + tick_length

                sleep_duration = target_time - time.perf_counter() - 0.002
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

                while time.perf_counter() < target_time:
                    pass

        finally:
            self.stop()

    def stop(self):
        if not self.running:
            return

        self.running = False

        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except OSError:
                pass
            self.server_socket = None

        with self.clients_lock:
            clients = list(self.clients)

        for client in clients:
            client.close()
