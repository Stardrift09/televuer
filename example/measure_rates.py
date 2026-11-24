"""Measure the real pose rate the headset delivers, with no robot involved.

Runs TeleVuer with a synthetic stereo image (or none in pass-through mode) and prints,
every --stats seconds, from the vuer process:

    CONTROLLER_MOVE <msgs> Hz (unique <new poses> Hz) gap max <ms> (client <ms>) | CAMERA_MOVE ... | video <fps>

(with --client lean the streams are named HEAD and CONTROLLER / HAND, and a second line gives the
headset's own numbers: XR frame rate, time spent per frame, video decode and upload times)

'unique' is what counts: messages that carry a pose different from the previous one.
'gap' uses PC arrival time, 'client' uses the headset's own send time.
Keep the controllers moving while measuring, a still controller repeats the same pose.

Headset: https://localhost:8012/?ws=wss://localhost:8012  (after `adb reverse tcp:8012 tcp:8012`)
         with --client lean just https://localhost:8012/
"""
import argparse
import os
import sys
import threading
import time

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--mode", choices=["pass-through", "immersive", "ego"], default="pass-through")
ap.add_argument("--hand", action="store_true", help="hand tracking instead of controllers")
ap.add_argument("--client", choices=["vuer", "lean"], default=None, help="web client on the headset (default: TELEVUER_CLIENT or vuer)")
ap.add_argument("--mono", action="store_true", help="one image for both eyes instead of a stereo pair")
ap.add_argument("--control-fps", type=float, default=90.0)
ap.add_argument("--display-fps", type=float, default=45.0, help="max video frames per second sent to the headset")
ap.add_argument("--cam-fps", type=float, default=30.0, help="rate of the synthetic camera feeding render_to_xr")
ap.add_argument("--sweep-fps", type=float, nargs="+", default=None, metavar="FPS",
                help="cycle the synthetic camera through these rates, --sweep-seconds each (instead of --cam-fps)")
ap.add_argument("--sweep-seconds", type=float, default=20.0)
ap.add_argument("--scroll", action="store_true", help="slide the whole synthetic image sideways (worst case for JPEG size; uncomfortable to watch)")
ap.add_argument("--img-shape", type=int, nargs=2, default=[720, 2560], metavar=("H", "W"))
ap.add_argument("--jpeg-quality", type=int, default=80)
ap.add_argument("--stats", type=float, default=2.0, help="seconds between stats lines")
ap.add_argument("--cert", default=None)
ap.add_argument("--key", default=None)
args = ap.parse_args()

os.environ["TELEVUER_STATS"] = str(args.stats)  # read by TeleVuer.__init__
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import cv2
import numpy as np
from televuer.televuer import TeleVuer


def synthetic_frame(h, w):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([(xx * 255 // w), (yy * 255 // h), ((xx + yy) % 256)], axis=-1).astype(np.uint8)
    for _ in range(60):
        color = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(10, 120)), color, -1)
    noise = cv2.GaussianBlur(rng.integers(0, 40, (h, w, 3), dtype=np.uint8), (5, 5), 0)
    return cv2.add(img, noise)


def feed(tv, stop):
    h, w = args.img_shape
    eye_w = w if args.mono else w // 2
    base = synthetic_frame(h, eye_w)  # one view, shown to both eyes with a small disparity
    rates = args.sweep_fps or [args.cam_fps]
    k, i, t_switch = -1, 0, 0.0
    while not stop.is_set():
        now = time.perf_counter()
        if now >= t_switch:
            k = (k + 1) % len(rates)
            period, t_next, t_switch = 1.0 / rates[k], now, now + args.sweep_seconds
            if len(rates) > 1:
                print(f"[measure_rates] camera now {rates[k]:g} fps", flush=True)
        # the background stays still (a picture sliding across the whole view makes people dizzy);
        # only the counter and a small marker change, enough to make every frame a new JPEG
        left = np.roll(base, (i * 7) % eye_w, axis=1) if args.scroll else base.copy()
        cv2.putText(left, f"{i}  {rates[k]:g} fps", (eye_w // 2 - 260, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        x = eye_w // 2 - 150 + (i * 5) % 300
        cv2.rectangle(left, (x, h // 2 + 40), (x + 30, h // 2 + 70), (255, 255, 255), -1)
        cv2.putText(left, "TOP LEFT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        frame = left if args.mono else np.hstack([left, np.roll(left, -6, axis=1)])
        tv.render_to_xr(np.ascontiguousarray(frame))
        i += 1
        t_next += period
        time.sleep(max(0.0, t_next - time.perf_counter()))


def main():
    tv = TeleVuer(use_hand_tracking=args.hand, binocular=not args.mono, img_shape=tuple(args.img_shape),
                  display_fps=args.display_fps, display_mode=args.mode, zmq=args.mode != "pass-through",
                  cert_file=args.cert, key_file=args.key, control_fps=args.control_fps, jpeg_quality=args.jpeg_quality,
                  client=args.client)
    stop = threading.Event()
    if args.mode != "pass-through":
        threading.Thread(target=feed, args=(tv, stop), daemon=True).start()
    print(f"[measure_rates] client={tv.client} mode={args.mode} control_fps={args.control_fps:g} display_fps={args.display_fps:g} "
          f"cam_fps={args.cam_fps:g} img={args.img_shape[0]}x{args.img_shape[1]} jpeg_quality={args.jpeg_quality}")
    url = "https://localhost:8012/" if tv.client == "lean" else "https://localhost:8012/?ws=wss://localhost:8012"
    print(f"[measure_rates] open {url} on the headset, enter VR, keep the controllers moving. Ctrl+C to stop.")
    xyz = lambda m: "(%+.2f %+.2f %+.2f)" % tuple(m[:3, 3])
    try:
        while True:
            time.sleep(max(args.stats, 1.0))
            if not tv.motion_data_ready:
                continue
            # where things are, in metres (x right, y up from the floor, z towards the user), and the inputs
            line = f"head {xyz(tv.head_pose)} left {xyz(tv.left_arm_pose)} right {xyz(tv.right_arm_pose)} age {tv.motion_data_age:.2f} s"
            if args.hand:
                line += (f" | pinch L {int(tv.left_hand_pinch)} {tv.left_hand_pinchValue:.3f} R {int(tv.right_hand_pinch)} {tv.right_hand_pinchValue:.3f}"
                         f" squeeze L {int(tv.left_hand_squeeze)} {tv.left_hand_squeezeValue:.3f} R {int(tv.right_hand_squeeze)} {tv.right_hand_squeezeValue:.3f}")
            else:
                side = lambda s: (f"trig {getattr(tv, s + '_ctrl_triggerValue'):.2f} grip {getattr(tv, s + '_ctrl_squeezeValue'):.2f} "
                                  f"stick ({getattr(tv, s + '_ctrl_thumbstickValue')[0]:+.2f} {getattr(tv, s + '_ctrl_thumbstickValue')[1]:+.2f}) "
                                  f"A{int(getattr(tv, s + '_ctrl_aButton'))} B{int(getattr(tv, s + '_ctrl_bButton'))}")
                line += f" | L {side('left')} | R {side('right')}"
            print(f"[measure_rates] {line}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        tv.close()


if __name__ == "__main__":
    main()
