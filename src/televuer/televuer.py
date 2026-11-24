from vuer import Vuer
from vuer.schemas import ImageBackground, Hands, MotionControllers, WebRTCVideoPlane, WebRTCStereoVideoPlane
from multiprocessing import Value, Array, Process, shared_memory
import numpy as np
import asyncio
import threading
import cv2
import os
import re
import time
import hashlib
from pathlib import Path
from typing import Literal
from aiohttp import web
from .lean import LeanServer


# vuer's ImagePlane (rendered by ImageBackground) builds PlaneGeometry(1, 1, img.width, img.height):
# one quad per pixel, ~1.8 M triangles per eye for a 1280x720 image. That caps the headset at ~22 fps
# whatever the video rate. The grid is only needed for depth displacement, so the plain branch
# (no depth map) is served with a single quad.
_IMAGE_PLANE_GRID = re.compile(r'args:\[1,1,([\w$]+)==null\?void 0:\1\.width,\1==null\?void 0:\1\.height\]')


def _patched_vuer_client_files(client_root):
    """Return {url path: patched JS bytes} for the vuer client files that contain the image plane grid."""
    client_root = Path(client_root)
    patched = {}
    for f in sorted(client_root.glob("assets/**/*.js")):
        try:
            src = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "planeGeometry" not in src:
            continue
        parts, last = [], 0
        for m in _IMAGE_PLANE_GRID.finditer(src):
            if "displacementMap:" in src[m.end():m.end() + 400]:
                continue  # depth branch: keeps the per-pixel grid for displacement
            parts += [src[last:m.start()], "args:[1,1,1,1]"]
            last = m.end()
        if parts:
            parts.append(src[last:])
            patched["/" + f.relative_to(client_root).as_posix()] = "".join(parts).encode("utf-8")
    return patched


class _RateStats:
    """Pose / video rates measured in the vuer process (enable with TELEVUER_STATS=<seconds>).

    'unique' counts messages whose pose differs from the previous one, i.e. real new samples.
    'gap' is the largest time between two messages; 'client gap' is the same using the headset's
    own send timestamps, so a large gap there means the headset itself did not send in time.
    'N over 30 ms' counts the arrival gaps longer than LATE.
    """
    LATE = 0.030

    def __init__(self, interval):
        self.interval = interval if interval > 0 else 2.0
        self.streams = {}
        self.video_frames = 0
        self.video_dropped = 0
        self.shown_n, self.shown_sum, self.shown_max = 0, 0.0, 0.0
        self.client = None  # the lean client's own numbers for the last interval
        self.ws_pool = {}  # open websockets, set by TeleVuer
        self._task = None

    def session_started(self):
        # one report loop per process, however many websocket sessions are open
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self.report_loop())

    def tick(self, name, pose, client_ts):
        now = time.monotonic()
        cts = client_ts.timestamp() if hasattr(client_ts, "timestamp") else float(client_ts)
        st = self.streams.setdefault(name, {"n": 0, "unique": 0, "gap": 0.0, "cgap": 0.0, "late": 0, "last": None, "last_t": None, "last_c": None})
        st["n"] += 1
        if pose != st["last"]:
            st["unique"] += 1
        if st["last_t"] is not None:
            st["gap"] = max(st["gap"], now - st["last_t"])
            st["cgap"] = max(st["cgap"], cts - st["last_c"])
            if now - st["last_t"] > self.LATE:
                st["late"] += 1
        st["last"], st["last_t"], st["last_c"] = pose, now, cts

    def video_shown(self, latency):
        """A video frame was drawn on the headset `latency` seconds after it was sent (lean client)."""
        self.shown_n += 1
        self.shown_sum += latency
        self.shown_max = max(self.shown_max, latency)

    async def report_loop(self):
        t0 = time.monotonic()
        while True:
            await asyncio.sleep(self.interval)
            dt = time.monotonic() - t0
            t0 += dt
            parts = []
            for name, st in self.streams.items():
                frozen = " FROZEN" if st["n"] and not st["unique"] else ""
                late = f", {st['late']} over {self.LATE * 1000:.0f} ms" if st["late"] else ""
                parts.append(f"{name} {st['n'] / dt:5.1f} Hz (unique {st['unique'] / dt:5.1f}{frozen}) "
                             f"gap max {st['gap'] * 1000:4.0f} ms (client {st['cgap'] * 1000:4.0f} ms{late})")
                st.update(n=0, unique=0, gap=0.0, cgap=0.0, late=0)
            video = f"video {self.video_frames / dt:4.1f} fps"
            if self.shown_n:
                video += f" (sent->shown avg {self.shown_sum / self.shown_n * 1000:3.0f} max {self.shown_max * 1000:3.0f} ms)"
            if self.video_dropped:
                video += f" dropped {self.video_dropped}"
            parts.append(f"{video} | open websockets {len(self.ws_pool)}")
            self.video_frames = self.video_dropped = self.shown_n = 0
            self.shown_sum = self.shown_max = 0.0
            print("[TeleVuer stats] " + " | ".join(parts), flush=True)
            c, self.client = self.client, None
            if c:
                try:
                    print(f"[TeleVuer stats]   headset: XR {c['xr_fps']:5.1f} fps (target {c['target_fps']:g}) gap max {c['xr_gap_ms']:4.0f} ms, "
                          f"frame work avg {c['frame_ms']:.2f} max {c['frame_max_ms']:.1f} ms, poses skipped {c['pose_skipped']} | "
                          f"video in {c['video_recv_fps']:4.1f} shown {c['video_shown_fps']:4.1f} fps, dropped {c['video_dropped']}, "
                          f"decode {c['decode_ms']:.1f} ms, upload avg {c['upload_ms']:.1f} max {c['upload_max_ms']:.1f} ms", flush=True)
                except (KeyError, TypeError, ValueError):
                    pass


class TeleVuer:
    def __init__(self, use_hand_tracking: bool, binocular: bool=True, img_shape: tuple=None, display_fps: float=30.0,
                       display_mode: Literal["immersive", "pass-through", "ego"]="immersive", zmq: bool=False, webrtc: bool=False, webrtc_url: str=None, 
                       cert_file: str=None, key_file: str=None, control_fps: float=90.0, jpeg_quality: int=80,
                       client: Literal["vuer", "lean"]=None):
        """
        TeleVuer class for OpenXR-based XR teleoperate applications.
        This class handles the communication with the Vuer server and manages image and pose data.

        :param use_hand_tracking: bool, whether to use hand tracking or controller tracking.
        :param binocular: bool, whether the application is binocular (stereoscopic) or monocular.
        :param img_shape: tuple, shape of the head image (height, width).
        :param display_fps: float, target frames per second for display updates (default: 30.0).
        
        :param display_mode: str, controls the VR viewing mode. Options are "immersive", "pass-through", and "ego".
        :param zmq: bool, whether to use zmq for image transmission.
        :param webrtc: bool, whether to use webrtc for real-time communication.
        :param webrtc_url: str, URL for the webrtc offer. must be provided if webrtc is True.
        :param cert_file: str, path to the SSL certificate file.
        :param key_file: str, path to the SSL key file.
        :param control_fps: float, rate at which the XR client sends hand/controller poses (default: 90.0).
                            vuer's client default is 30. Poses are sampled once per XR frame, so the effective
                            rate is capped by the headset refresh rate (72/90 Hz) and drops if the headset browser is busy.
        :param jpeg_quality: int, JPEG quality of the zmq image stream (default: 80).
        :param client: str, which web client the headset runs. "vuer": vuer's own client. "lean": TeleVuer's
                       small WebXR page (see lean.py), which sends one pose message per XR frame and decodes the
                       images off the main thread, so the pose rate stays at the headset refresh rate while images
                       are streamed. Same URL, port and pose conventions. It has no webrtc video: with webrtc=True
                       the vuer client is used. Default: env var TELEVUER_CLIENT, else "vuer".

        Set the env var TELEVUER_STATS=<seconds> to print pose / video rates from the server process.
        TELEVUER_PORT changes the port (default 8012).

        Note:

        - display_mode controls what the VR headset displays:
            * "immersive": fully immersive mode; VR shows the robot's first-person view (zmq or webrtc must be enabled).
            * "pass-through": VR shows the real world through the VR headset cameras; no image from zmq or webrtc is displayed (even if enabled).
            * "ego": a small window in the center shows the robot's first-person view, while the surrounding area shows the real world.
        
        - Only one image mode is active at a time.
        - Image transmission to VR occurs only if display_mode is "immersive" or "ego" and the corresponding zmq or webrtc option is enabled.
        - If zmq and webrtc simultaneously enabled, webrtc will be prioritized.

        --------------              -------------------           --------------       -----------------                     -------
         display_mode       |        display behavior         |    image to VR     |      image source        |               Notes
        --------------              -------------------           --------------       -----------------                     ------- 
           immersive        |   fully immersive view (robot)  |     Yes (full)     |     zmq or webrtc        |   if both enabled, webrtc prioritized
        --------------              -------------------           --------------       -----------------                     -------
         pass-through       |       Real world view (VR)      |         No         |          N/A             |  even if image source enabled, don't display
        --------------              -------------------           --------------       -----------------                     -------
              ego           |      ego view (robot + VR)      |    Yes (small)     |     zmq or webrtc        |   if both enabled, webrtc prioritized
        --------------              -------------------           --------------       -----------------                     -------

        """
        self.use_hand_tracking = use_hand_tracking
        self.binocular = binocular
        if img_shape is None:
            raise ValueError("[TeleVuer] img_shape must be provided.")
        self.img_shape = (img_shape[0], img_shape[1], 3)
        self.display_fps = display_fps
        self.control_fps = control_fps
        self.jpeg_quality = int(jpeg_quality)
        self.img_height = self.img_shape[0]
        if self.binocular:
            self.img_width  = self.img_shape[1] // 2
        else:
            self.img_width  = self.img_shape[1]
        self.aspect_ratio = self.img_width / self.img_height

        # SSL certificate path resolution
        env_cert = os.getenv("XR_TELEOP_CERT")
        env_key = os.getenv("XR_TELEOP_KEY")
        if cert_file is None or key_file is None:
            # 1.Try environment variables
            if env_cert and env_key:
                cert_file = cert_file or env_cert
                key_file = key_file or env_key
            else:
                # 2.Try ~/.config/xr_teleoperate/
                user_conf_dir = Path.home() / ".config" / "xr_teleoperate"
                cert_path_user = user_conf_dir / "cert.pem"
                key_path_user = user_conf_dir / "key.pem"

                if cert_path_user.exists() and key_path_user.exists():
                    cert_file = cert_file or str(cert_path_user)
                    key_file = key_file or str(key_path_user)
                else:
                    # 3.Fallback to package root (current logic)
                    current_module_dir = Path(__file__).resolve().parent.parent.parent
                    cert_file = cert_file or str(current_module_dir / "cert.pem")
                    key_file = key_file or str(current_module_dir / "key.pem")

        if display_mode not in ("immersive", "ego", "pass-through"):
            raise ValueError(f"[TeleVuer] Unknown display_mode: {display_mode}")
        self.display_mode = display_mode
        self.zmq = zmq
        self.webrtc = webrtc
        self.webrtc_url = webrtc_url
        port = int(os.getenv("TELEVUER_PORT", "8012"))
        self._stats = _RateStats(float(os.getenv("TELEVUER_STATS", "0") or 0)) if os.getenv("TELEVUER_STATS") else None

        client = (client or os.getenv("TELEVUER_CLIENT") or "vuer").lower()
        if client not in ("vuer", "lean"):
            raise ValueError(f"[TeleVuer] Unknown client: {client}")
        if client == "lean" and self.webrtc and self.display_mode != "pass-through":
            print("[TeleVuer] The lean client has no webrtc video, using the vuer client.")
            client = "vuer"
        self.client = client

        if self.client == "lean":
            self.vuer = None
            if self.display_mode != "pass-through":
                if not self.zmq:
                    raise ValueError(f"[TeleVuer] {self.display_mode} mode requires zmq=True or webrtc=True.")
                self._setup_jpeg_pipe()
            self._lean = LeanServer(self, cert_file, key_file, port=port)
            if self._stats is not None:
                self._stats.ws_pool = self._lean.open_ws
            server_main = self._lean.run
        else:
            self._setup_vuer(cert_file, key_file, port)
            server_main = self._vuer_run

        self.head_pose_shared = Array('d', 16, lock=True)
        self.left_arm_pose_shared = Array('d', 16, lock=True)
        self.right_arm_pose_shared = Array('d', 16, lock=True)
        self.motion_data_ready_shared = Value('b', False, lock=True)
        self.motion_data_time_shared = Value('d', 0.0, lock=True)  # time.monotonic() of the last hand / controller data
        if self.use_hand_tracking:
            self.left_hand_position_shared = Array('d', 75, lock=True)
            self.right_hand_position_shared = Array('d', 75, lock=True)
            self.left_hand_orientation_shared = Array('d', 25 * 9, lock=True)
            self.right_hand_orientation_shared = Array('d', 25 * 9, lock=True)

            self.left_hand_pinch_shared = Value('b', False, lock=True)
            self.left_hand_pinchValue_shared = Value('d', 0.0, lock=True)
            self.left_hand_squeeze_shared = Value('b', False, lock=True)
            self.left_hand_squeezeValue_shared = Value('d', 0.0, lock=True)

            self.right_hand_pinch_shared = Value('b', False, lock=True)
            self.right_hand_pinchValue_shared = Value('d', 0.0, lock=True)
            self.right_hand_squeeze_shared = Value('b', False, lock=True)
            self.right_hand_squeezeValue_shared = Value('d', 0.0, lock=True)
        else:
            self.left_ctrl_trigger_shared = Value('b', False, lock=True)
            self.left_ctrl_triggerValue_shared = Value('d', 0.0, lock=True)
            self.left_ctrl_squeeze_shared = Value('b', False, lock=True)
            self.left_ctrl_squeezeValue_shared = Value('d', 0.0, lock=True)
            self.left_ctrl_thumbstick_shared = Value('b', False, lock=True)
            self.left_ctrl_thumbstickValue_shared = Array('d', 2, lock=True)
            self.left_ctrl_aButton_shared = Value('b', False, lock=True)
            self.left_ctrl_bButton_shared = Value('b', False, lock=True)

            self.right_ctrl_trigger_shared = Value('b', False, lock=True)
            self.right_ctrl_triggerValue_shared = Value('d', 0.0, lock=True)
            self.right_ctrl_squeeze_shared = Value('b', False, lock=True)
            self.right_ctrl_squeezeValue_shared = Value('d', 0.0, lock=True)
            self.right_ctrl_thumbstick_shared = Value('b', False, lock=True)
            self.right_ctrl_thumbstickValue_shared = Array('d', 2, lock=True)
            self.right_ctrl_aButton_shared = Value('b', False, lock=True)
            self.right_ctrl_bButton_shared = Value('b', False, lock=True)

        self.process = Process(target=server_main)
        self.process.daemon = True
        self.process.start()
    
    def _setup_vuer(self, cert_file, key_file, port):
        self.vuer = Vuer(host='0.0.0.0', port=port, cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)
        if not os.getenv("TELEVUER_NO_CLIENT_PATCH"):
            self._serve_patched_client()
        self.vuer.add_handler("CAMERA_MOVE")(self.on_cam_move)
        if self.use_hand_tracking:
            self.vuer.add_handler("HAND_MOVE")(self.on_hand_move)
        else:
            self.vuer.add_handler("CONTROLLER_MOVE")(self.on_controller_move)

        if self.display_mode == "immersive":
            if self.webrtc:
                fn = self.main_image_binocular_webrtc if self.binocular else self.main_image_monocular_webrtc
            elif self.zmq:
                self._setup_jpeg_pipe()
                fn = self.main_image_binocular_zmq if self.binocular else self.main_image_monocular_zmq
            else:
                raise ValueError("[TeleVuer] immersive mode requires zmq=True or webrtc=True.")
        elif self.display_mode == "ego":
            if self.webrtc:
                fn = self.main_image_binocular_webrtc_ego if self.binocular else self.main_image_monocular_webrtc_ego
            elif self.zmq:
                self._setup_jpeg_pipe()
                fn = self.main_image_binocular_zmq_ego if self.binocular else self.main_image_monocular_zmq_ego
            else:
                raise ValueError("[TeleVuer] ego mode requires zmq=True or webrtc=True.")
        elif self.display_mode == "pass-through":
            fn = self.main_pass_through

        if self._stats is not None:
            self._stats.ws_pool = self.vuer.ws
        self.vuer.spawn(start=False)(self._with_stats(fn))

    def _vuer_run(self):
        try:
            self.vuer.run()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Vuer encountered an error: {e}")
        finally:
            if hasattr(self, "stop_writer_event"):
                self.stop_writer_event.set()

    def _serve_patched_client(self):
        """Serve the patched vuer client files (see _IMAGE_PLANE_GRID). Registered before vuer adds its
        /assets static route, so these paths win. no-cache makes headsets revalidate instead of reusing
        a heuristically cached unpatched copy on later visits."""
        files = _patched_vuer_client_files(self.vuer.client_root)
        if not files:
            print("[TeleVuer] Warning: vuer client image plane not found, not patched; large images will render slowly on the headset.")
            return
        for path, body in files.items():
            etag = '"tv-%s"' % hashlib.sha1(body).hexdigest()[:16]

            async def handler(request, body=body, etag=etag):
                headers = {"Cache-Control": "no-cache", "ETag": etag}
                if request.headers.get("If-None-Match") == etag:
                    return web.Response(status=304, headers=headers)
                return web.Response(body=body, content_type="application/javascript", headers=headers)

            self.vuer._add_route(path, handler)

    def _setup_jpeg_pipe(self):
        """JPEG frames are encoded here, in the caller's process, and handed to the vuer process
        through shared memory, so the vuer event loop (which also receives the poses) never encodes."""
        n_eyes = 2 if self.binocular else 1
        self.jpeg_slot_size = self.img_height * self.img_width * 3  # raw size: upper bound for one JPEG
        self.jpeg_shm = shared_memory.SharedMemory(create=True, size=self.jpeg_slot_size * n_eyes)
        self.jpeg_lens = Array('i', n_eyes, lock=False)
        self.jpeg_seq = Value('L', 0, lock=True)  # its lock guards jpeg_shm + jpeg_lens too
        self.latest_frame = None
        self.new_frame_event = threading.Event()
        self.stop_writer_event = threading.Event()
        self.writer_thread = threading.Thread(target=self._xr_render_loop, daemon=True)
        self.writer_thread.start()

    def _xr_render_loop(self):
        params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        slot = self.jpeg_slot_size
        while not self.stop_writer_event.is_set():
            if not self.new_frame_event.wait(timeout=0.1):
                continue
            self.new_frame_event.clear()
            frame = self.latest_frame
            if frame is None:
                continue
            try:
                # frame is BGR, which is what cv2.imencode expects
                eyes = (frame[:, :self.img_width], frame[:, self.img_width:]) if self.binocular else (frame,)
                encoded = [cv2.imencode(".jpg", eye, params)[1] for eye in eyes]
                if any(len(b) > slot for b in encoded):
                    continue
                with self.jpeg_seq.get_lock():
                    for k, b in enumerate(encoded):
                        self.jpeg_shm.buf[k * slot: k * slot + len(b)] = b
                        self.jpeg_lens[k] = len(b)
                    self.jpeg_seq.value += 1
            except Exception as e:
                print(f"[TeleVuer] Failed to encode frame of shape {getattr(frame, 'shape', None)}: {e}")

    async def _jpeg_frames(self):
        """Yield each new camera frame (one JPEG per eye) once, at most display_fps times per second."""
        slot = self.jpeg_slot_size
        period = 1.0 / self.display_fps
        last_seq, next_send = 0, 0.0
        while True:
            delay = next_send - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            with self.jpeg_seq.get_lock():
                seq = self.jpeg_seq.value
                if seq != last_seq:
                    jpegs = [bytes(self.jpeg_shm.buf[k * slot: k * slot + self.jpeg_lens[k]]) for k in range(len(self.jpeg_lens))]
            if seq == last_seq:
                await asyncio.sleep(0.002)
                continue
            last_seq = seq
            # average send interval >= period; after an idle stretch the next frame may go out at once
            next_send = max(next_send + period, time.monotonic() - 0.5 * period)
            if self._stats is not None:
                self._stats.video_frames += 1
            yield jpegs

    def _with_stats(self, fn):
        if self._stats is None:
            return fn
        async def run(session):
            self._stats.session_started()
            await fn(session)
        return run

    def render_to_xr(self, image):
        if self.webrtc or self.display_mode == "pass-through":
            print("[TeleVuer] Warning: render_to_xr is ignored when webrtc is enabled or pass_through is True.")
            return
        self.latest_frame = image
        self.new_frame_event.set()

    def close(self):
        self.process.terminate()
        self.process.join(timeout=0.5)
        if hasattr(self, "writer_thread"):
            self.stop_writer_event.set()
            self.new_frame_event.set()
            self.writer_thread.join(timeout=0.5)
            try:
                self.jpeg_shm.close()
                self.jpeg_shm.unlink()
            except:
                pass

    async def on_cam_move(self, event, session, fps=60):
        try:
            with self.head_pose_shared.get_lock():
                self.head_pose_shared[:] = event.value["camera"]["matrix"]
            if self._stats is not None:
                self._stats.tick("CAMERA_MOVE", event.value["camera"]["matrix"], event.ts)
        except:
            pass

    async def on_controller_move(self, event, session, fps=60):
        """https://docs.vuer.ai/en/latest/examples/20_motion_controllers.html"""
        try:
            # ControllerData
            with self.left_arm_pose_shared.get_lock():
                self.left_arm_pose_shared[:] = event.value["left"]
            with self.right_arm_pose_shared.get_lock():
                self.right_arm_pose_shared[:] = event.value["right"]
            # ControllerState
            left_controller = event.value["leftState"]
            right_controller = event.value["rightState"]

            def extract_controllers(controllerState, prefix):
                # trigger
                with getattr(self, f"{prefix}_ctrl_trigger_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_trigger_shared").value = bool(controllerState.get("trigger", False))
                with getattr(self, f"{prefix}_ctrl_triggerValue_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_triggerValue_shared").value = float(controllerState.get("triggerValue", 0.0))
                # squeeze
                with getattr(self, f"{prefix}_ctrl_squeeze_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_squeeze_shared").value = bool(controllerState.get("squeeze", False))
                with getattr(self, f"{prefix}_ctrl_squeezeValue_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_squeezeValue_shared").value = float(controllerState.get("squeezeValue", 0.0))
                # thumbstick
                with getattr(self, f"{prefix}_ctrl_thumbstick_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_thumbstick_shared").value = bool(controllerState.get("thumbstick", False))
                with getattr(self, f"{prefix}_ctrl_thumbstickValue_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_thumbstickValue_shared")[:] = controllerState.get("thumbstickValue", [0.0, 0.0])
                # buttons
                with getattr(self, f"{prefix}_ctrl_aButton_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_aButton_shared").value = bool(controllerState.get("aButton", False))
                with getattr(self, f"{prefix}_ctrl_bButton_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_bButton_shared").value = bool(controllerState.get("bButton", False))

            extract_controllers(left_controller, "left")
            extract_controllers(right_controller, "right")
            with self.motion_data_time_shared.get_lock():
                self.motion_data_time_shared.value = time.monotonic()
            with self.motion_data_ready_shared.get_lock():
                self.motion_data_ready_shared.value = True
            if self._stats is not None:
                self._stats.tick("CONTROLLER_MOVE", (event.value["left"], event.value["right"]), event.ts)
        except:
            pass

    async def on_hand_move(self, event, session, fps=60):
        """https://docs.vuer.ai/en/latest/examples/19_hand_tracking.html"""
        try:
            # HandsData
            left_hand_data = event.value["left"]
            right_hand_data = event.value["right"]
            left_hand = event.value["leftState"]
            right_hand = event.value["rightState"]
            # HandState
            def extract_hand_poses(hand_data, arm_pose_shared, hand_position_shared, hand_orientation_shared):
                with arm_pose_shared.get_lock():
                    arm_pose_shared[:] = hand_data[0:16]

                with hand_position_shared.get_lock():
                    for i in range(25):
                        base = i * 16
                        hand_position_shared[i * 3: i * 3 + 3] = [hand_data[base + 12], hand_data[base + 13], hand_data[base + 14]]

                with hand_orientation_shared.get_lock():
                    for i in range(25):
                        base = i * 16
                        hand_orientation_shared[i * 9: i * 9 + 9] = [
                            hand_data[base + 0], hand_data[base + 1], hand_data[base + 2],
                            hand_data[base + 4], hand_data[base + 5], hand_data[base + 6],
                            hand_data[base + 8], hand_data[base + 9], hand_data[base + 10],
                        ]

            def extract_hands(handState, prefix):
                # pinch
                with getattr(self, f"{prefix}_hand_pinch_shared").get_lock():
                    getattr(self, f"{prefix}_hand_pinch_shared").value = bool(handState.get("pinch", False))
                with getattr(self, f"{prefix}_hand_pinchValue_shared").get_lock():
                    getattr(self, f"{prefix}_hand_pinchValue_shared").value = float(handState.get("pinchValue", 0.0))
                # squeeze
                with getattr(self, f"{prefix}_hand_squeeze_shared").get_lock():
                    getattr(self, f"{prefix}_hand_squeeze_shared").value = bool(handState.get("squeeze", False))
                with getattr(self, f"{prefix}_hand_squeezeValue_shared").get_lock():
                    getattr(self, f"{prefix}_hand_squeezeValue_shared").value = float(handState.get("squeezeValue", 0.0))

            extract_hand_poses(left_hand_data, self.left_arm_pose_shared, self.left_hand_position_shared, self.left_hand_orientation_shared)
            extract_hand_poses(right_hand_data, self.right_arm_pose_shared, self.right_hand_position_shared, self.right_hand_orientation_shared)
            extract_hands(left_hand, "left")
            extract_hands(right_hand, "right")
            with self.motion_data_time_shared.get_lock():
                self.motion_data_time_shared.value = time.monotonic()
            with self.motion_data_ready_shared.get_lock():
                self.motion_data_ready_shared.value = True
            if self._stats is not None:
                self._stats.tick("HAND_MOVE", (left_hand_data, right_hand_data), event.ts)

        except:
            pass
    
    ## immersive MODE
    async def main_image_binocular_zmq(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    fps=self.control_fps,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True,
                    fps=self.control_fps,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )
        async for jpegs in self._jpeg_frames():
            session.upsert(
                [
                    ImageBackground(
                        src=jpegs[0],
                        aspect=self.aspect_ratio,
                        height=1,
                        distanceToCamera=1,
                        # The underlying rendering engine supported a layer binary bitmask for both objects and the camera. 
                        # Below we set the two image planes, left and right, to layers=1 and layers=2. 
                        # Note that these two masks are associated with left eye’s camera and the right eye’s camera.
                        layers=1,
                        key="background-left",
                        interpolate=True,
                    ),
                    ImageBackground(
                        src=jpegs[1],
                        aspect=self.aspect_ratio,
                        height=1,
                        distanceToCamera=1,
                        layers=2,
                        key="background-right",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )

    async def main_image_monocular_zmq(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    fps=self.control_fps,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True,
                    fps=self.control_fps,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        async for jpegs in self._jpeg_frames():
            session.upsert(
                [
                    ImageBackground(
                        src=jpegs[0],
                        aspect=self.aspect_ratio,
                        height=1,
                        distanceToCamera=1,
                        key="background-mono",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )

    async def main_image_binocular_webrtc(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    fps=self.control_fps,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True,
                    fps=self.control_fps,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                WebRTCStereoVideoPlane(
                    src=self.webrtc_url,
                    iceServer=None,
                    iceServers=[], 
                    key="video-quad",
                    aspect=self.aspect_ratio,
                    height = 7,
                    layout="stereo-left-right"
                ),
                to="bgChildren",
            )
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_monocular_webrtc(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    fps=self.control_fps,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True,
                    fps=self.control_fps,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                WebRTCVideoPlane(
                    src=self.webrtc_url,
                    iceServer=None,
                    iceServers=[],
                    key="video-quad",
                    aspect=self.aspect_ratio,
                    height = 7,
                ),
                to="bgChildren",
            )
            await asyncio.sleep(1.0 / self.display_fps)

    ## ego MODE
    async def main_image_binocular_zmq_ego(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    fps=self.control_fps,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True,
                    fps=self.control_fps,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )
        async for jpegs in self._jpeg_frames():
            session.upsert(
                [
                    ImageBackground(
                        src=jpegs[0],
                        aspect=self.aspect_ratio,
                        height=0.75,
                        distanceToCamera=2,
                        # The underlying rendering engine supported a layer binary bitmask for both objects and the camera. 
                        # Below we set the two image planes, left and right, to layers=1 and layers=2. 
                        # Note that these two masks are associated with left eye’s camera and the right eye’s camera.
                        layers=1,
                        key="background-left",
                        interpolate=True,
                    ),
                    ImageBackground(
                        src=jpegs[1],
                        aspect=self.aspect_ratio,
                        height=0.75,
                        distanceToCamera=2,
                        layers=2,
                        key="background-right",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )

    async def main_image_monocular_zmq_ego(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    fps=self.control_fps,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True,
                    fps=self.control_fps,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        async for jpegs in self._jpeg_frames():
            session.upsert(
                [
                    ImageBackground(
                        src=jpegs[0],
                        aspect=self.aspect_ratio,
                        height=0.75,
                        distanceToCamera=2,
                        key="background-mono",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )

    async def main_image_binocular_webrtc_ego(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    fps=self.control_fps,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True,
                    fps=self.control_fps,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                WebRTCStereoVideoPlane(
                    src=self.webrtc_url,
                    iceServer=None,
                    iceServers=[], 
                    key="video-quad",
                    aspect=self.aspect_ratio,
                    height=3,
                    layout="stereo-left-right"
                ),
                to="bgChildren",
            )
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_monocular_webrtc_ego(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    fps=self.control_fps,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True,
                    fps=self.control_fps,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                WebRTCVideoPlane(
                    src=self.webrtc_url,
                    iceServer=None,
                    iceServers=[],
                    key="video-quad",
                    aspect=self.aspect_ratio,
                    height=3,
                ),
                to="bgChildren",
            )
            await asyncio.sleep(1.0 / self.display_fps)

    ## pass-through MODE
    async def main_pass_through(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    fps=self.control_fps,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True,
                    fps=self.control_fps,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while session.CURRENT_WS_ID in self.vuer.ws:
            await asyncio.sleep(1.0 / self.display_fps)

    # ==================== common data ====================
    @property
    def head_pose(self):
        """np.ndarray, shape (4, 4), head SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.head_pose_shared.get_lock():
            return np.array(self.head_pose_shared[:]).reshape(4, 4, order="F")

    @property
    def left_arm_pose(self):
        """np.ndarray, shape (4, 4), left arm SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.left_arm_pose_shared.get_lock():
            return np.array(self.left_arm_pose_shared[:]).reshape(4, 4, order="F")

    @property
    def right_arm_pose(self):
        """np.ndarray, shape (4, 4), right arm SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.right_arm_pose_shared.get_lock():
            return np.array(self.right_arm_pose_shared[:]).reshape(4, 4, order="F")

    # ==================== Hand Tracking Data ====================
    @property
    def left_hand_positions(self):
        """np.ndarray, shape (25, 3), left hand 25 landmarks' 3D positions."""
        with self.left_hand_position_shared.get_lock():
            return np.array(self.left_hand_position_shared[:]).reshape(25, 3)

    @property
    def right_hand_positions(self):
        """np.ndarray, shape (25, 3), right hand 25 landmarks' 3D positions."""
        with self.right_hand_position_shared.get_lock():
            return np.array(self.right_hand_position_shared[:]).reshape(25, 3)

    @property
    def left_hand_orientations(self):
        """np.ndarray, shape (25, 3, 3), left hand 25 landmarks' orientations (flattened 3x3 matrices, column-major)."""
        with self.left_hand_orientation_shared.get_lock():
            return np.array(self.left_hand_orientation_shared[:]).reshape(25, 9).reshape(25, 3, 3, order="F")

    @property
    def right_hand_orientations(self):
        """np.ndarray, shape (25, 3, 3), right hand 25 landmarks' orientations (flattened 3x3 matrices, column-major)."""
        with self.right_hand_orientation_shared.get_lock():
            return np.array(self.right_hand_orientation_shared[:]).reshape(25, 9).reshape(25, 3, 3, order="F")

    @property
    def left_hand_pinch(self):
        """bool, whether left hand is pinching."""
        with self.left_hand_pinch_shared.get_lock():
            return self.left_hand_pinch_shared.value

    @property
    def left_hand_pinchValue(self):
        """float, pinch strength of left hand."""
        with self.left_hand_pinchValue_shared.get_lock():
            return self.left_hand_pinchValue_shared.value

    @property
    def left_hand_squeeze(self):
        """bool, whether left hand is squeezing."""
        with self.left_hand_squeeze_shared.get_lock():
            return self.left_hand_squeeze_shared.value

    @property
    def left_hand_squeezeValue(self):
        """float, squeeze strength of left hand."""
        with self.left_hand_squeezeValue_shared.get_lock():
            return self.left_hand_squeezeValue_shared.value

    @property
    def right_hand_pinch(self):
        """bool, whether right hand is pinching."""
        with self.right_hand_pinch_shared.get_lock():
            return self.right_hand_pinch_shared.value

    @property
    def right_hand_pinchValue(self):
        """float, pinch strength of right hand."""
        with self.right_hand_pinchValue_shared.get_lock():
            return self.right_hand_pinchValue_shared.value

    @property
    def right_hand_squeeze(self):
        """bool, whether right hand is squeezing."""
        with self.right_hand_squeeze_shared.get_lock():
            return self.right_hand_squeeze_shared.value

    @property
    def right_hand_squeezeValue(self):
        """float, squeeze strength of right hand."""
        with self.right_hand_squeezeValue_shared.get_lock():
            return self.right_hand_squeezeValue_shared.value

    # ==================== Controller Data ====================
    @property
    def left_ctrl_trigger(self):
        """bool, left controller trigger pressed or not."""
        with self.left_ctrl_trigger_shared.get_lock():
            return self.left_ctrl_trigger_shared.value

    @property
    def left_ctrl_triggerValue(self):
        """float, left controller trigger analog value (0.0 ~ 1.0)."""
        with self.left_ctrl_triggerValue_shared.get_lock():
            return self.left_ctrl_triggerValue_shared.value

    @property
    def left_ctrl_squeeze(self):
        """bool, left controller squeeze pressed or not."""
        with self.left_ctrl_squeeze_shared.get_lock():
            return self.left_ctrl_squeeze_shared.value

    @property
    def left_ctrl_squeezeValue(self):
        """float, left controller squeeze analog value (0.0 ~ 1.0)."""
        with self.left_ctrl_squeezeValue_shared.get_lock():
            return self.left_ctrl_squeezeValue_shared.value

    @property
    def left_ctrl_thumbstick(self):
        """bool, whether left thumbstick is touched or clicked."""
        with self.left_ctrl_thumbstick_shared.get_lock():
            return self.left_ctrl_thumbstick_shared.value

    @property
    def left_ctrl_thumbstickValue(self):
        """np.ndarray, shape (2,), left thumbstick 2D axis values (x, y)."""
        with self.left_ctrl_thumbstickValue_shared.get_lock():
            return np.array(self.left_ctrl_thumbstickValue_shared[:])

    @property
    def left_ctrl_aButton(self):
        """bool, left controller 'A' button pressed."""
        with self.left_ctrl_aButton_shared.get_lock():
            return self.left_ctrl_aButton_shared.value

    @property
    def left_ctrl_bButton(self):
        """bool, left controller 'B' button pressed."""
        with self.left_ctrl_bButton_shared.get_lock():
            return self.left_ctrl_bButton_shared.value

    @property
    def right_ctrl_trigger(self):
        """bool, right controller trigger pressed or not."""
        with self.right_ctrl_trigger_shared.get_lock():
            return self.right_ctrl_trigger_shared.value

    @property
    def right_ctrl_triggerValue(self):
        """float, right controller trigger analog value (0.0 ~ 1.0)."""
        with self.right_ctrl_triggerValue_shared.get_lock():
            return self.right_ctrl_triggerValue_shared.value

    @property
    def right_ctrl_squeeze(self):
        """bool, right controller squeeze pressed or not."""
        with self.right_ctrl_squeeze_shared.get_lock():
            return self.right_ctrl_squeeze_shared.value

    @property
    def right_ctrl_squeezeValue(self):
        """float, right controller squeeze analog value (0.0 ~ 1.0)."""
        with self.right_ctrl_squeezeValue_shared.get_lock():
            return self.right_ctrl_squeezeValue_shared.value

    @property
    def right_ctrl_thumbstick(self):
        """bool, whether right thumbstick is touched or clicked."""
        with self.right_ctrl_thumbstick_shared.get_lock():
            return self.right_ctrl_thumbstick_shared.value

    @property
    def right_ctrl_thumbstickValue(self):
        """np.ndarray, shape (2,), right thumbstick 2D axis values (x, y)."""
        with self.right_ctrl_thumbstickValue_shared.get_lock():
            return np.array(self.right_ctrl_thumbstickValue_shared[:])

    @property
    def right_ctrl_aButton(self):
        """bool, right controller 'A' button pressed."""
        with self.right_ctrl_aButton_shared.get_lock():
            return self.right_ctrl_aButton_shared.value

    @property
    def right_ctrl_bButton(self):
        """bool, right controller 'B' button pressed."""
        with self.right_ctrl_bButton_shared.get_lock():
            return self.right_ctrl_bButton_shared.value

    @property
    def motion_data_ready(self):
        """bool, whether at least one hand or controller motion data event has been received."""
        with self.motion_data_ready_shared.get_lock():
            return self.motion_data_ready_shared.value

    @property
    def motion_data_age(self):
        """float, seconds since the last hand or controller motion data arrived (inf if none yet).
        The lean client only sends while the headset is rendering XR frames, so this grows as soon as
        the headset is taken off, the session ends or the link drops."""
        with self.motion_data_time_shared.get_lock():
            t = self.motion_data_time_shared.value
        return time.monotonic() - t if t > 0 else float("inf")
