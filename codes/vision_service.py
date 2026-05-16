import threading
import time
import numpy as np
import cv2

from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

# ==========================================
# 1. MOCK DETECTOR CLASS
# Replace this or import your real model here
# ==========================================
class Detector:
    def __init__(self):
        print("Initializing Object Detector...")
        # e.g., self.model = yolov8.load('yolov8n.pt')
        time.sleep(1) # Simulating model loading time
        print("Detector ready.")

    def run_inference(self, frame):
        """
        Takes a BGR frame and returns a detection result.
        Replace this placeholder logic with your actual model inference.
        """
        # Placeholder result format: (label, confidence, bbox [x, y, w, h])
        # For testing, we just simulate finding a target if the image is valid
        if frame is not None:
            return {"status": "success", "detections": [{"class": "target", "confidence": 0.92, "box": [100, 100, 50, 50]}]}
        return {"status": "empty", "detections": []}


# ==========================================
# 2. VISION SERVICE CLASS
# ==========================================
class VisionService:
    def __init__(self, topic):
        self.node = Node()
        self.detector = Detector()
        
        # Thread locks for shared state resources
        self.frame_lock = threading.Lock()
        self.result_lock = threading.Lock()
        
        # Shared States
        self.latest_frame = None
        self.latest_result = None

        # Subscribe to the Gazebo RGB Image topic
        self.node.subscribe(Image, topic, self.rgb_callback)
        print(f"Vision Service subscribed to RGB topic: {topic}")

    def rgb_callback(self, msg):
        """
        Handles incoming RGB images from Gazebo asynchronously.
        Transforms raw bytes into standard OpenCV BGR format.
        """
        # Gazebo usually sends RGB8 data. Extract flat byte buffer to numpy array
        frame_raw = np.frombuffer(msg.data, dtype=np.uint8)
        
        # Reshape to 3 channels (Height, Width, RGB Channels)
        frame_rgb = frame_raw.reshape((msg.height, msg.width, 3))
        
        # OpenCV utilizes BGR natively, convert it so colors look normal
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        with self.frame_lock:
            self.latest_frame = frame_bgr

    def submit_latest_frame(self):
        """
        Fetches the most recent frame safely, runs it through the 
        detector, and commits the output back to the shared state.
        """
        # Thread-safe retrieval of the latest captured image
        with self.frame_lock:
            if self.latest_frame is None:
                return  # No frame available yet
            frame_to_process = self.latest_frame.copy()

        # Run model inference (expensive processing step outside the lock!)
        detection_output = self.detector.run_inference(frame_to_process)

        # Thread-safe storage of the detection outcomes
        with self.result_lock:
            self.latest_result = detection_output

    def get_latest_result(self):
        """Allows external systems/loops to grab the latest inferences safely."""
        with self.result_lock:
            return self.latest_result if self.latest_result is not None else {"status": "no_data"}


# ==========================================
# 3. TERMINAL TESTING ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    # Change to match your actual Gazebo camera topic (`gz topic -l`)
    RGB_TOPIC = "/world/roboverse/model/x500_depth_0/link/camera_link/sensor/IMX214/image" 
    
    service = VisionService(RGB_TOPIC)
    print("Vision Service processing loop running. Press Ctrl+C to stop.")

    try:
        while True:
            # 1. Grab the latest frame and process it through the detector
            service.submit_latest_frame()
            
            # 2. View results saved in the shared state
            results = service.get_latest_result()
            print(f"[SHARED STATE RESULT]: {results}")
            
            # 3. Optional: If you want to see the visual stream to confirm it works
            with service.frame_lock:
                display_frame = service.latest_frame.copy() if service.latest_frame is not None else None
            
            if display_frame is not None:
                cv2.imshow("Vision Service Monitor", display_frame)
            
            # Run loop at roughly ~20-30Hz
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping Vision Service cleanly...")
    finally:
        cv2.destroyAllWindows()