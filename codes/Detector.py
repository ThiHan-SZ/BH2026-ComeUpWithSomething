import threading
import time
import queue
import numpy as np
import cv2
import os

from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from ultralytics import YOLO

# Global transport node instance to prevent garbage collection dropping sockets
GLOBAL_GZ_NODE = Node()

class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self.vision = {}
        self.depth = {}
        self.planner = {}

    def update_vision(self, data: dict):
        with self._lock:
            self.vision = data

    def update_depth(self, data: dict):
        with self._lock:
            self.depth = data

    def update_planner(self, data: dict):
        with self._lock:
            self.planner = data

    def get_latest_data(self) -> tuple:
        with self._lock:
            return self.vision.copy(), self.depth.copy(), self.planner.copy()


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

                if annotated is None:
                    annotated = r.plot()

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
        self.topic = topic
        self.shared_state = shared_state
        
        if GLOBAL_GZ_NODE.subscribe(Image, topic, self.depth_callback):
            print(f"DepthReceiver successfully subscribed to: {topic}")
        else:
            print(f"[ERROR] DepthReceiver failed to subscribe to: {topic}")

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
                return

            valid_depth = depth_image[np.isfinite(depth_image)]

            if valid_depth.size == 0:
                depth_data = {
                    "timestamp": time.time(), "min_depth": None, "max_depth": None,
                    "mean_depth": None, "shape": depth_image.shape, "raw_data": depth_image, "valid_pixels": 0,
                }
            else:
                depth_data = {
                    "timestamp": time.time(), "min_depth": float(valid_depth.min()),
                    "max_depth": float(valid_depth.max()), "mean_depth": float(valid_depth.mean()),
                    "shape": depth_image.shape, "raw_data": depth_image, "valid_pixels": int(valid_depth.size),
                }
            self.shared_state.update_depth(depth_data)
        except Exception as e:
            print(f"Depth callback error: {e}")


class RGBReceiver:
    def __init__(self, topic, shared_state, detection_queue):
        self.topic = topic
        self.shared_state = shared_state
        self.detection_queue = detection_queue
        
        if GLOBAL_GZ_NODE.subscribe(Image, topic, self.rgb_callback):
            print(f"RGBReceiver successfully subscribed to: {topic}")
        else:
            print(f"[ERROR] RGBReceiver failed to subscribe to: {topic}")

    def rgb_callback(self, msg):
        try:
            if not msg.data:
                return

            image_bytes = bytes(msg.data)
            expected_size = msg.height * msg.width * 3
            
            if len(image_bytes) >= expected_size:
                pixel_data = image_bytes[:expected_size]
                frame_raw = np.frombuffer(pixel_data, dtype=np.uint8)
                frame_rgb = frame_raw.reshape((msg.height, msg.width, 3))
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                frame_raw = np.frombuffer(image_bytes, dtype=np.uint8)
                frame_bgr = cv2.imdecode(frame_raw, cv2.IMREAD_COLOR)

            if frame_bgr is None or frame_bgr.size == 0:
                return

            rgb_data = {
                "timestamp": time.time(), "image": frame_bgr, "shape": frame_bgr.shape, "frame_id": self.topic,
            }

            self.shared_state.update_vision(rgb_data)

            try:
                self.detection_queue.put_nowait(frame_bgr.copy())
            except queue.Full:
                pass
        except Exception as e:
            print(f"[RGB Callback Error] {e}")


class DetectorWorker(threading.Thread):
    def __init__(self, detection_queue, ui_queue, shared_state, detector, save_dir="../detections"):
        super().__init__(daemon=True)
        self.detection_queue = detection_queue
        self.ui_queue = ui_queue
        self.shared_state = shared_state
        self.detector = detector
        self.save_dir = save_dir
        self.running = True

        os.makedirs(self.save_dir, exist_ok=True)
        print(f"[Worker] Snapshots destination folder initialized at: {os.path.abspath(self.save_dir)}")

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

                if annotated is not None and detections:
                    ts = int(time.time() * 1000)
                    filename = os.path.join(self.save_dir, f"detect_{ts}.jpg")
                    cv2.imwrite(filename, annotated)
                    detection_data["saved_image"] = filename

                final_render = annotated if annotated is not None else frame
                try:
                    self.ui_queue.put_nowait(final_render)
                except queue.Full:
                    pass

                self.shared_state.update_planner(detection_data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Detector worker error: {e}")

    def stop(self):
        self.running = False