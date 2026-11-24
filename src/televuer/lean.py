"""Server side of TeleVuer's lean WebXR client (TeleVuer(client="lean")).

The page in lean_client/ replaces vuer's React client on the headset, and this small aiohttp
server replaces the vuer server. Nothing else changes: same port, same certificates, same
shared memory, same pose conventions as the vuer client.

  GET /        the page
  WS  /pose    headset -> PC, one binary message per XR frame (head + both controllers or hands)
  WS  /video   PC -> headset, one binary message per camera frame (one JPEG per eye)

Pose message, little-endian (written by lean_client/main.js):
   0  f64      client time, ms
   8  u32      seq of the video frame drawn in this XR frame (0: none)
  12  u8       kind: 0 controllers, 1 hands
  13  u8       flags: bit0 head valid, bit1 left valid, bit2 right valid
  16  f32[16]  head matrix, column-major
  80  left block, then right block
      controllers (84 bytes each): f32[16] pose, f32 trigger, f32 squeeze, f32 thumbstick x, y, u32 buttons
      hands (1608 bytes each):     f32[25 * 16] joint matrices, f32 pinch distance, f32 squeeze distance

Video message, little-endian:
   0  u32      seq
   4  u8       n (number of eyes), 3 bytes padding
   8  u32[n]   JPEG sizes, followed by the n JPEGs
"""
import asyncio
import hashlib
import json
import os
import ssl
import struct
import time
from pathlib import Path

import numpy as np
from aiohttp import web, WSMsgType

CLIENT_DIR = Path(__file__).resolve().parent / "lean_client"
ASSETS = {"main.js": "application/javascript", "video_worker.js": "application/javascript"}

_POSE_HEADER = struct.Struct("<dIBBH")
_HEAD_OFFSET = 16
_BLOCKS_OFFSET = 80
_CTRL_BLOCK = 84
_HAND_BLOCK = 1608
_BUTTONS = ("trigger", "squeeze", "touchpad", "thumbstick", "aButton", "bButton")  # bit order = gamepad.buttons order
_ROT_IDX = [0, 1, 2, 4, 5, 6, 8, 9, 10]
# same thresholds as vuer's Hands component (metres)
_PINCH_DIST = 0.01
_SQUEEZE_DIST = 0.07


class LeanServer:
    def __init__(self, tv, cert_file, key_file, port=8012):
        self.tv = tv
        self.cert_file, self.key_file = cert_file, key_file
        self.port = port
        self.open_ws = set()
        self._np = {}
        self._video_seq = 0
        self._sent = {}  # video seq -> send time, for the send -> shown latency
        self._shown = 0

    # ---------------------------------------------------------------- http
    def _version(self):
        h = hashlib.sha1()
        for name in ("index.html", *ASSETS):
            h.update((CLIENT_DIR / name).read_bytes())
        return h.hexdigest()[:12]

    def _config(self):
        tv = self.tv
        ego = tv.display_mode == "ego"
        return {
            "type": "config",
            "version": self._version(),
            "mode": tv.display_mode,
            "hand": bool(tv.use_hand_tracking),
            "video": tv.display_mode != "pass-through",
            # image plane, same placement as the vuer client: height (m) at distance (m) in front of the head
            "aspect": tv.aspect_ratio,
            "height": 0.75 if ego else 1.0,
            "distance": 2.0 if ego else 1.0,
            "stats": tv._stats.interval if tv._stats is not None else 0,
        }

    @staticmethod
    def _respond(request, body, content_type):
        # no-cache: the headset revalidates on every load, so it never runs an outdated client
        etag = '"%s"' % hashlib.sha1(body).hexdigest()[:16]
        headers = {"Cache-Control": "no-cache", "ETag": etag}
        if request.headers.get("If-None-Match") == etag:
            return web.Response(status=304, headers=headers)
        return web.Response(body=body, content_type=content_type, headers=headers)

    async def _index(self, request):
        html = (CLIENT_DIR / "index.html").read_text(encoding="utf-8").replace("__VERSION__", self._version())
        return self._respond(request, html.encode("utf-8"), "text/html")

    async def _asset(self, request):
        name = request.path.lstrip("/")
        return self._respond(request, (CLIENT_DIR / name).read_bytes(), ASSETS[name])

    # ---------------------------------------------------------------- poses
    def _shared(self, name):
        """numpy view on one of TeleVuer's shared arrays (take its lock before writing)."""
        view = self._np.get(name)
        if view is None:
            view = self._np[name] = np.frombuffer(getattr(self.tv, name).get_obj(), dtype=np.float64)
        return view

    def _write(self, name, values):
        with getattr(self.tv, name).get_lock():
            self._shared(name)[:] = values

    def _set(self, name, value):
        shared = getattr(self.tv, name)
        with shared.get_lock():
            shared.value = value

    def _on_pose(self, data):
        if len(data) < _BLOCKS_OFFSET:
            return
        ts_ms, video_seq, kind, flags, _ = _POSE_HEADER.unpack_from(data)
        tv, stats = self.tv, self.tv._stats
        if flags & 1:
            self._write("head_pose_shared", np.frombuffer(data, "<f4", 16, _HEAD_OFFSET))
            if stats is not None:
                stats.tick("HEAD", data[_HEAD_OFFSET:_BLOCKS_OFFSET], ts_ms / 1000.0)
        if stats is not None and video_seq != self._shown:
            self._shown = video_seq
            sent = self._sent.pop(video_seq, None)
            if sent is not None:
                stats.video_shown(time.monotonic() - sent)
        # like the vuer client, hands / controllers are applied only when both sides are tracked
        if flags & 6 != 6:
            return
        if kind == 1 and tv.use_hand_tracking and len(data) >= _BLOCKS_OFFSET + 2 * _HAND_BLOCK:
            for k, side in enumerate(("left", "right")):
                off = _BLOCKS_OFFSET + k * _HAND_BLOCK
                joints = np.frombuffer(data, "<f4", 25 * 16, off).reshape(25, 16)
                pinch, squeeze = struct.unpack_from("<2f", data, off + 25 * 16 * 4)
                self._write(f"{side}_arm_pose_shared", joints[0])  # wrist
                self._write(f"{side}_hand_position_shared", joints[:, 12:15].ravel())
                self._write(f"{side}_hand_orientation_shared", joints[:, _ROT_IDX].ravel())
                self._set(f"{side}_hand_pinch_shared", pinch < _PINCH_DIST)
                self._set(f"{side}_hand_pinchValue_shared", pinch)
                self._set(f"{side}_hand_squeeze_shared", squeeze < _SQUEEZE_DIST)
                self._set(f"{side}_hand_squeezeValue_shared", squeeze)
            name = "HAND"
        elif kind == 0 and not tv.use_hand_tracking and len(data) >= _BLOCKS_OFFSET + 2 * _CTRL_BLOCK:
            for k, side in enumerate(("left", "right")):
                off = _BLOCKS_OFFSET + k * _CTRL_BLOCK
                self._write(f"{side}_arm_pose_shared", np.frombuffer(data, "<f4", 16, off))
                trigger, squeeze, stick_x, stick_y, buttons = struct.unpack_from("<4fI", data, off + 64)
                self._set(f"{side}_ctrl_triggerValue_shared", trigger)
                self._set(f"{side}_ctrl_squeezeValue_shared", squeeze)
                self._write(f"{side}_ctrl_thumbstickValue_shared", (stick_x, stick_y))
                for bit, button in enumerate(_BUTTONS):
                    if button != "touchpad":
                        self._set(f"{side}_ctrl_{button}_shared", bool(buttons >> bit & 1))
            name = "CONTROLLER"
        else:
            return
        self._set("motion_data_time_shared", time.monotonic())
        self._set("motion_data_ready_shared", True)
        if stats is not None:
            stats.tick(name, data[_BLOCKS_OFFSET:], ts_ms / 1000.0)

    async def _pose_ws(self, request):
        ws = web.WebSocketResponse(max_msg_size=0, compress=False)
        await ws.prepare(request)
        self.open_ws.add(ws)
        stats = self.tv._stats
        if stats is not None:
            stats.session_started()
        try:
            await ws.send_str(json.dumps(self._config()))
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    try:
                        self._on_pose(msg.data)
                    except Exception as e:
                        print(f"[TeleVuer] bad pose message ({len(msg.data)} bytes): {e}")
                elif msg.type == WSMsgType.TEXT:
                    self._on_text(msg.data)
        finally:
            self.open_ws.discard(ws)
        return ws

    def _on_text(self, text):
        try:
            msg = json.loads(text)
        except ValueError:
            return
        if msg.get("type") == "stats" and self.tv._stats is not None:
            self.tv._stats.client = msg
        elif msg.get("type") == "log":
            print(f"[TeleVuer] headset: {msg.get('message')}", flush=True)

    # ---------------------------------------------------------------- video
    async def _video_ws(self, request):
        if not hasattr(self.tv, "jpeg_shm"):
            # pass-through: no images. A page left over from an image mode may still ask until it gets the new config.
            return web.Response(status=404)
        ws = web.WebSocketResponse(max_msg_size=0, compress=False)  # JPEG does not deflate
        await ws.prepare(request)
        self.open_ws.add(ws)
        sender = asyncio.ensure_future(self._send_video(ws, request.transport))
        try:
            async for _ in ws:
                pass
        finally:
            sender.cancel()
            self.open_ws.discard(ws)
        return ws

    async def _send_video(self, ws, transport):
        stats = self.tv._stats
        try:
            async for jpegs in self.tv._jpeg_frames():
                if transport is None or transport.is_closing():
                    return
                if transport.get_write_buffer_size() > 0:
                    # the link is behind: drop this frame instead of queueing it, so the picture stays current
                    if stats is not None:
                        stats.video_dropped += 1
                    continue
                self._video_seq = seq = self._video_seq % 0xFFFFFFFF + 1
                header = struct.pack(f"<IB3x{len(jpegs)}I", seq, len(jpegs), *map(len, jpegs))
                self._sent[seq] = time.monotonic()
                while len(self._sent) > 64:
                    del self._sent[next(iter(self._sent))]
                await ws.send_bytes(b"".join((header, *jpegs)))
        except (ConnectionError, asyncio.CancelledError):
            pass
        except Exception as e:
            print(f"[TeleVuer] video stream stopped: {e}")

    # ---------------------------------------------------------------- run
    async def _serve(self):
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/pose", self._pose_ws)
        app.router.add_get("/video", self._video_ws)
        for name in ASSETS:
            app.router.add_get("/" + name, self._asset)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        ssl_context = None
        if self.cert_file and self.key_file and os.path.exists(self.cert_file) and os.path.exists(self.key_file):
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(certfile=self.cert_file, keyfile=self.key_file)
        else:
            print("[TeleVuer] no SSL certificate found, serving plain http: works through `adb reverse` "
                  "(http://localhost), not over Wi-Fi.")
        await web.TCPSite(runner, "0.0.0.0", self.port, ssl_context=ssl_context).start()
        scheme = "https" if ssl_context else "http"
        print(f"[TeleVuer] lean client ready: {scheme}://localhost:{self.port}/ (USB: adb reverse tcp:{self.port} tcp:{self.port})",
              flush=True)
        await asyncio.Event().wait()

    def run(self):
        try:
            asyncio.run(self._serve())
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"[TeleVuer] lean server error: {e}")
