"""
UseDetectorExample.py
Main vision application setup that coordinates the Gazebo transport receivers,
the thread-safe shared state, and the background YOLOv10 detection worker.

Run:
    python3 UseDetectorExample.py
"""

import queue
import time
import os
import cv2

# Import components directly from your Detector.py file
from Detector import Detector, SharedState, RGBReceiver, DepthReceiver, DetectorWorker

class VisionApp:
    def __init__(self):
        print("[VisionApp] Initializing systems...")
        
        # 1. Initialize the thread-safe communication layer
        self.shared_state = SharedState()
        self.detection_queue = queue.Queue(maxsize=1) 
        self.ui_queue = queue.Queue(maxsize=1) 

        # Track model file position from inside the codes directory
        finetuned = "models/barrel_detector_best.pt"
        if not os.path.exists(finetuned) and os.path.exists("../models/barrel_detector_best.pt"):
            finetuned = "../models/barrel_detector_best.pt"

        # 2. Instantiate the core detector with your fine-tuned weights
        self.detector = Detector(
            base_model_path="yolov10n.pt",
            finetuned_weights=finetuned,
            device="cpu",
            conf_threshold=0.6
        )
        
        self.worker = None

    def run(self):
        # FIX HERE: This variable is what gets passed to the worker thread. 
        # Setting it to "../detections" forces it out of the codes folder.
        target_save_directory = "../detections"
        
        # 1. Start the heavy inference processing thread pointing to your parallel folder layout
        self.worker = DetectorWorker(
            self.detection_queue, 
            self.ui_queue, 
            self.shared_state, 
            self.detector,
            save_dir=target_save_directory
        )
        self.worker.start()

        # 2. Define your Gazebo topics
        rgb_topic = "/world/roboverse/model/x500_vision_0/link/camera_link/sensor/IMX214/image"

        # 3. Initialize the receiver node
        print("[VisionApp] Connecting to Gazebo network streams...")
        rgb_receiver = RGBReceiver(rgb_topic, self.shared_state, self.detection_queue)
        
        print("\n[VisionApp] System running. OpenCV window processing on Main Thread.")
        print("-> Press 'q' on the image window to close.")

        # Create the UI Window context explicitly on the Main Thread
        cv2.namedWindow("Barrel Detections", cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                try:
                    frame_to_show = self.ui_queue.get_nowait()
                    cv2.imshow("Barrel Detections", frame_to_show)
                except queue.Empty:
                    time.sleep(0.001)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[VisionApp] Exiting user interface...")
                    break
        except KeyboardInterrupt:
            print("\n[VisionApp] Shutdown command received.")
        finally:
            self.shutdown()

    def shutdown(self):
        print("[VisionApp] Cleaning up resource threads...")
        if self.worker is not None:
            self.worker.stop()
        cv2.destroyAllWindows()
        print("[VisionApp] Exited cleanly.")

if __name__ == "__main__":
    # Hardcode VM networking overrides within the runtime environment 
    os.environ["GZ_IP"] = "127.0.0.1"
    os.environ["GZ_PARTITION"] = "default"
    
    # Run application
    VisionApp().run()