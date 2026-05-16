import threading
import time
import queue
import numpy as np
import cv2

from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

class Detector:
    def __init__(self):
        print("Initializing Object Detector...")
        time.sleep(1)
        print("Detector ready.")

    def run_inference(self, frame):
        """
        Takes a BGR frame and returns a detection result.
        Replace this placeholder logic with actual model inference.
        """
        if frame is not None:
            return {
                "status": "success",
                "detections": [
                    {
                        "class": "target",
                        "confidence": 0.92,
                        "box": [100, 100, 50, 50],
                    }
                ],
            }
        return {"status": "empty", "detections": []}

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

                detection_data = {
                    "timestamp": time.time(),
                    "detections": detection_output.get("detections", []),
                    "status": detection_output.get("status", "unknown"),
                    "frame_shape": frame.shape,
                    "num_detections": len(detection_output.get("detections", [])),
                }

                self.shared_state.update_planner(detection_data)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Detector worker error: {e}")

    def stop(self):
        self.running = False
