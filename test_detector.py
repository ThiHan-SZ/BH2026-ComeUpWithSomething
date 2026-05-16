"""
test_detector_gazebo.py
Standalone smoke test: subscribes to Gazebo RGB camera,
runs YOLOv10 on every frame, shows live display + saves detect_*.jpg

No drone arming or flight needed. Just requires Gazebo to be running.

Run:
    python test_detector_gazebo.py
"""
import time
import queue
import threading
import numpy as np
import cv2

from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
# Run `gz topic -l` to confirm your exact topic path
RGB_TOPIC   = "/world/roboverse/model/x500_vision_0/link/camera_link/sensor/IMX214/image"
MODEL_PATH  = "models/barrel_detector_best.pt"
CONF        = 0.6
SAVE_DIR    = "./detections_test"
IMG_W, IMG_H = 640, 480
# ─────────────────────────────────────────────────────────────────────────────

import os
os.makedirs(SAVE_DIR, exist_ok=True)

# Load model once
print(f"[TEST] Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("[TEST] Model loaded. Waiting for Gazebo frames...")

# Thread-safe frame queue (maxsize=1 keeps only latest)
frame_q = queue.Queue(maxsize=1)
stats = {"frames": 0, "detections": 0, "saved": 0}

def gz_callback(msg: Image):
    """Gazebo callback — convert RGB→BGR and push to queue."""
    try:
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        frame = raw.reshape((msg.height, msg.width, 3))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        try:
            frame_q.put_nowait(frame_bgr)
        except queue.Full:
            pass  # drop old frame, keep latest
    except Exception as e:
        print(f"[gz_callback] {e}")

# Subscribe
node = Node()
ok = node.subscribe(Image, RGB_TOPIC, gz_callback)
if not ok:
    print(f"[ERROR] Could not subscribe to {RGB_TOPIC}")
    print("  → Is Gazebo running? Try: gz topic -l")
    exit(1)
print(f"[TEST] Subscribed to {RGB_TOPIC}\n")

# ── Main detection loop ───────────────────────────────────────────────────────
try:
    while True:
        try:
            frame = frame_q.get(timeout=2.0)
        except queue.Empty:
            print("[TEST] No frame received in 2s — is the sim paused?")
            continue

        stats["frames"] += 1

        # Run YOLO
        results = model(frame, verbose=False, conf=CONF)

        detections = []
        annotated = frame.copy()

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            annotated = r.plot()   # draws all boxes + labels

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf  = float(box.conf[0])
                cls   = r.names[int(box.cls[0])]
                cx    = (x1 + x2) / 2
                cy    = (y1 + y2) / 2
                err_x = cx - IMG_W / 2    # + = right of centre
                err_y = cy - IMG_H / 2    # + = below centre

                detections.append((cls, conf, cx, cy, err_x, err_y))
                stats["detections"] += 1

        # ── Print results ──────────────────────────────────────────────────
        if detections:
            for cls, conf, cx, cy, ex, ey in detections:
                colour = "🔴" if cls == "red_barrel" else "🟡"
                print(f"  {colour} {cls:<15}  conf={conf:.2f}  "
                      f"px=({cx:.0f},{cy:.0f})  err=({ex:+.0f},{ey:+.0f})")

            # ── Save annotated image ───────────────────────────────────────
            fname = os.path.join(SAVE_DIR, f"detect_{stats['saved']:04d}.jpg")
            cv2.imwrite(fname, annotated)
            stats["saved"] += 1
            print(f"  💾 Saved {fname}")
        else:
            print(f"  [frame {stats['frames']:04d}] No barrels detected.")

        # ── Live display ────────────────────────────────────────────────────
        # Draw image centre crosshair
        cv2.line(annotated, (IMG_W//2 - 20, IMG_H//2), (IMG_W//2 + 20, IMG_H//2), (0,255,0), 1)
        cv2.line(annotated, (IMG_W//2, IMG_H//2 - 20), (IMG_W//2, IMG_H//2 + 20), (0,255,0), 1)

        cv2.imshow("Barrel Detector — Gazebo Test", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Manual save on 's' key
            fname = os.path.join(SAVE_DIR, f"manual_{stats['saved']:04d}.jpg")
            cv2.imwrite(fname, annotated)
            print(f"  📸 Manual save: {fname}")
            stats["saved"] += 1

except KeyboardInterrupt:
    pass

finally:
    cv2.destroyAllWindows()
    print(f"\n[TEST] Done — {stats['frames']} frames | "
          f"{stats['detections']} detections | {stats['saved']} images saved")