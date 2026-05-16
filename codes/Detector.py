import threading
import time
import queue
import numpy as np
import cv2

from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from ultralytics import YOLO

class Detector:
    def __init__(
        self,
        base_model_path: str = "yolov10n.pt",
        finetuned_weights: str = "models/barrel_detector_best.pt",
        device: str = "cpu",
        conf_threshold: float = 0.6,
    ):
        print("[Detector] Initializing YOLOv10 detector...")

        try:
            self.model = YOLO(finetuned_weights).to(device)
            print(f"[Detector] Loaded fine-tuned weights: {finetuned_weights}")
        except Exception as e:
            print(f"[Detector] Failed to load {finetuned_weights}: {e}")
            print(f"[Detector] Falling back to base model: {base_model_path}")
            self.model = YOLO(base_model_path).to(device)

        self.conf_threshold = conf_threshold
        self.device = device
        print("[Detector] Ready.")

    def run_inference(self, frame: np.ndarray) -> dict:
        if frame is None or frame.size == 0:
            return {"status": "empty", "detections": [], "annotated": None}

        try:
            results = self.model(frame, verbose=False, conf=self.conf_threshold)

            all_dets = []
            annotated = None

            for r in results:
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                # Let Ultralytics draw all boxes once per Result
                if annotated is None:
                    annotated = r.plot()  # BGR image with boxes & labels

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().item())
                    cls_id = int(box.cls[0].cpu().item())
                    class_name = r.names[cls_id]

                    w = x2 - x1
                    h = y2 - y1

                    all_dets.append({
                        "class": class_name,
                        "confidence": conf,
                        "box": [x1, y1, w, h],
                    })

            status = "success" if all_dets else "empty"
            return {"status": status, "detections": all_dets, "annotated": annotated}

        except Exception as e:
            print(f"[Detector] Inference error: {e}")
            return {"status": "error", "detections": [], "annotated": None}

class DepthReceiver:
    def __init__(self, topic, shared_state):
        self.node = Node()
        self.topic = topic
        self.shared_state = shared_state
        self.latest_depth = None
        self.lock = threading.Lock()

        self.node.subscribe(Image, topic, self.depth_callback)
        print(f"DepthReceiver subscribed to: {topic}")

    def depth_callback(self, msg):
        try:
            if not msg.data:
                return

            raw = bytes(msg.data)
            expected_float32 = msg.width * msg.height * 4
            expected_uint16 = msg.width * msg.height * 2

            if len(raw) == expected_float32:
                depth_raw = np.frombuffer(raw, dtype=np.float32)
                depth_image = depth_raw.reshape((msg.height, msg.width))

            elif len(raw) == expected_uint16:
                depth_raw = np.frombuffer(raw, dtype=np.uint16)
                depth_image = depth_raw.reshape((msg.height, msg.width)).astype(np.float32)

            else:
                print(
                    f"Unexpected depth buffer size: {len(raw)}, "
                    f"expected {expected_float32} or {expected_uint16}"
                )
                return

            valid_depth = depth_image[np.isfinite(depth_image)]

            if valid_depth.size == 0:
                depth_data = {
                    "timestamp": time.time(),
                    "min_depth": None,
                    "max_depth": None,
                    "mean_depth": None,
                    "shape": depth_image.shape,
                    "raw_data": depth_image,
                    "valid_pixels": 0,
                }
            else:
                depth_data = {
                    "timestamp": time.time(),
                    "min_depth": float(valid_depth.min()),
                    "max_depth": float(valid_depth.max()),
                    "mean_depth": float(valid_depth.mean()),
                    "shape": depth_image.shape,
                    "raw_data": depth_image,
                    "valid_pixels": int(valid_depth.size),
                }

            self.shared_state.update_depth(depth_data)
        except Exception as e:
            print(f"Depth callback error: {e}")


class RGBReceiver:
    def __init__(self, topic, shared_state, detection_queue):
        self.node = Node()
        self.topic = topic
        self.shared_state = shared_state
        self.detection_queue = detection_queue
        self.latest_frame = None
        self.lock = threading.Lock()

        self.node.subscribe(Image, topic, self.rgb_callback)
        print(f"RGBReceiver subscribed to: {topic}")

    def rgb_callback(self, msg):
        try:
            if not msg.data:
                return

            frame_raw = np.frombuffer(msg.data, dtype=np.uint8)
            frame_rgb = frame_raw.reshape((msg.height, msg.width, 3))
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            rgb_data = {
                "timestamp": time.time(),
                "image": frame_bgr,
                "shape": frame_bgr.shape,
                "frame_id": self.topic,
            }

            with self.lock:
                self.latest_frame = frame_bgr

            self.shared_state.update_vision(rgb_data)

            try:
                self.detection_queue.put_nowait(frame_bgr.copy())
            except queue.Full:
                pass

        except Exception as e:
            print(f"RGB callback error: {e}")

class DetectorWorker(threading.Thread):
    def __init__(self, detection_queue, shared_state, detector):
        super().__init__(daemon=True)
        self.detection_queue = detection_queue
        self.shared_state = shared_state
        self.detector = detector
        self.running = True

    def run(self):
        while self.running:
            try:
                frame = self.detection_queue.get(timeout=1.0)
                detection_output = self.detector.run_inference(frame)

                detections = detection_output.get("detections", [])
                annotated = detection_output.get("annotated", None)

                detection_data = {
                    "timestamp": time.time(),
                    "detections": detections,
                    "status": detection_output.get("status", "unknown"),
                    "frame_shape": frame.shape,
                    "num_detections": len(detections),
                }

                # --- 2a. Save detectX.jpg when we have at least one barrel ---
                if annotated is not None and detections:
                    # Optional: ensure box shows at least 50% of barrel by slightly enlarging
                    # the drawn boxes is already handled by YOLO’s default annotations,
                    # but you can re-draw if you want to pad by, say, 1.2x.

                    ts = int(time.time() * 1000)
                    filename = f"detect_{ts}.jpg"
                    cv2.imwrite(filename, annotated)
                    detection_data["saved_image"] = filename  # for logging/telemetry

                # --- 2b. Optional: live display window ---
                # (only during testing, not in final autonomous run)
                if annotated is not None:
                    cv2.imshow("Barrel Detections", annotated)
                    cv2.waitKey(1)

                self.shared_state.update_planner(detection_data)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Detector worker error: {e}")

    def stop(self):
        self.running = False
