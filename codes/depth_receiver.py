from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
import numpy as np
import threading
import cv2
import time

class DepthReceiver:
    def __init__(self, topic):
        self.node = Node()
        self.depth = None
        self.lock = threading.Lock()

        # ✅ FIXED LINE
        self.node.subscribe(Image, topic, self.callback)

    def callback(self, msg):
        depth = np.frombuffer(msg.data, dtype=np.float32)
        depth = depth.reshape((msg.height, msg.width))

        with self.lock:
            self.depth = depth

    def get_frame(self):
        with self.lock:
            return None if self.depth is None else self.depth.copy()
        
#most awesome main function in depth receiver main function code
if __name__ == "__main__":
    # run `gz topic -l` in the terminal
    DEPTH_TOPIC = "/depth_camera" 

    print("Starting Depth Receiver Test...")
    receiver = DepthReceiver(DEPTH_TOPIC)

    try:
        while True:
            frame = receiver.get_frame()

            if frame is not None:
                # 2. Normalize the float32 depth map to 0-255 for visualization
                # We handle NaNs/Infs by cleaning up the frame first
                clean_frame = np.nan_to_num(frame, nan=0.0, posinf=10.0, neginf=0.0)
                
                normalized_depth = cv2.normalize(
                    clean_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )

                # 3. Flip the depth representation if you want Red = Near, Blue = Far
                visual_depth = 255 - normalized_depth

                # 4. Apply Jet Colormap
                color_depth = cv2.applyColorMap(visual_depth, cv2.COLORMAP_JET)

                # 5. Show the frame in an OpenCV Window
                cv2.imshow("Gazebo Depth Stream", color_depth)
            else:
                print("Waiting for depth frames...")

            # Break loop on 'q' key press
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

            # Sleep slightly to prevent maxing out the CPU core
            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\nShutting down cleanly...")
    finally:
        cv2.destroyAllWindows()