#!/usr/bin/env python3
import asyncio
import numpy as np
import time
import math
import os
import queue
import cv2

# --- Navigation Imports ---
from depth_receiver import DepthReceiver
from drone_control import Drone
from AvoidancePlanner import AvoidancePlanner
from get_position_with_task import SharedState as NavSharedState, position_monitor_task

# --- Vision Imports ---
from Detector import Detector, SharedState as VisionSharedState, RGBReceiver, DetectorWorker
from mavsdk.action import ActionError

class IntegratedDroneApp:
    def __init__(self, depth_topic="/depth_camera", loop_hz=10.0):
        print("[System] Initializing Integrated Navigation & Vision Systems...")
        
        # ===================================
        # 🛰️ NAVIGATION & DRONE PROPERTIES
        # ===================================
        self.loop_hz = loop_hz
        self.running = True
        self.target_yaw_deg = 0.0
        self.yaw_tolerance = 4.0          
        self.large_turn_threshold = 15.0  

        self.pose = {
            "north": 0.0, "east": 0.0, "down": -2.0,
            "yaw": 0.0, "yaw_deg": 0.0
        }

        self.is_recovering = False
        self.recovery_yaw_deg = 0.0

        # Camera intrinsics
        K = np.array([[433.0, 0.0, 320.0],
                      [0.0, 433.0, 240.0],
                      [0.0, 0.0, 1.0]])

        self.receiver = DepthReceiver(depth_topic)
        self.planner = AvoidancePlanner(K=K, width=640, height=480, safe_distance=4.0, critical_distance=1.5)
        self.drone = Drone()
        self.position_state = NavSharedState()    
        self.monitor_task = None
        
        self.last_command_time = 0.0
        self.command_interval = 0.1  

        # ===================================
        # 👁️ VISION & DETECTION PROPERTIES (Deferred Initialization)
        # ===================================
        self.vision_shared_state = VisionSharedState()
        self.detection_queue = queue.Queue(maxsize=1) 
        self.ui_queue = queue.Queue(maxsize=1) 

        # Track model file position from inside the codes directory
        self.finetuned_path = "models/barrel_detector_best.pt"
        if not os.path.exists(self.finetuned_path) and os.path.exists("../models/barrel_detector_best.pt"):
            self.finetuned_path = "../models/barrel_detector_best.pt"

        self.detector = None
        self.worker = None
        self.rgb_topic = "/world/roboverse/model/x500_vision_0/link/camera_link/sensor/IMX214/image"

    def _yaw_error(self, target, current):
        error = target - current
        while error > 180: error -= 360
        while error < -180: error += 360
        return error

    async def update_pose(self):
        self.pose["north"] = self.position_state.latest_position.north_m
        self.pose["east"]  = self.position_state.latest_position.east_m
        self.pose["down"]  = self.position_state.latest_position.down_m
        self.pose["yaw_deg"]   = self.position_state.latest_yaw
        self.pose["yaw"] = np.deg2rad(self.pose["yaw_deg"])

    async def run_vision_ui(self):
        """Displays OpenCV window without blocking execution loop."""
        cv2.namedWindow("Barrel Detections", cv2.WINDOW_AUTOSIZE)
        try:
            while self.running:
                try:
                    frame_to_show = self.ui_queue.get_nowait()
                    cv2.imshow("Barrel Detections", frame_to_show)
                except queue.Empty:
                    pass
                
                # Check for UI exit key 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[Vision] User exited UI window.")
                    self.stop()
                    break
                await asyncio.sleep(0.01) # Yield execution control back to loop
        finally:
            cv2.destroyAllWindows()

    async def run_navigation_loop(self):
        """Continuous obstacle avoidance navigation loop executed after takeoff."""
        try:
            while self.running:
                t_start = time.monotonic()
                await self.update_pose()

                depth_frame = self.receiver.get_frame()

                # Process point cloud array path translations
                north, east, down, info = self.planner.compute_position_ned(
                    depth_frame, self.pose, step_size=1.5
                )

                # Extract smooth target heading updates from the planner
                relative_furthest_deg = np.rad2deg(info['furthest_direction']['angle_rad'])
                global_target_yaw = self.pose["yaw_deg"] + relative_furthest_deg
                self.target_yaw_deg = (global_target_yaw + 180) % 360 - 180

                heading_error = self._yaw_error(self.target_yaw_deg, self.pose["yaw_deg"])

                current_time = time.monotonic()
                if (current_time - self.last_command_time) >= self.command_interval:
                    
                    # -------------------------------------------------
                    # 🔄 INERTIAL RECOVERY LOGIC (STRAIGHT BACKWARD)
                    # -------------------------------------------------
                    if info['blocked']:
                        if not self.is_recovering:
                            # Capture orientation right before stopping
                            self.recovery_yaw_deg = self.pose["yaw_deg"]
                            self.is_recovering = True
                            print(f"⚠️ DEAD END DETECTED! Latching baseline heading: {self.recovery_yaw_deg:.1f}°")

                        print(f"⚠️ RECOVERY ACTIVE: Backing out straight down corridor axis...")
                        
                        # Calculate static backward vector components relative to the latched corridor angle
                        recovery_yaw_rad = np.deg2rad(self.recovery_yaw_deg)
                        v_north = -0.3 * math.cos(recovery_yaw_rad)
                        v_east  = -0.3 * math.sin(recovery_yaw_rad)
                        
                        # Convert global vector to a body command to preserve flight axis
                        current_yaw_rad = self.pose["yaw"]
                        vx_body = v_north * math.cos(current_yaw_rad) + v_east * math.sin(current_yaw_rad)
                        vy_body = -v_north * math.sin(current_yaw_rad) + v_east * math.cos(current_yaw_rad)
                        
                        await self.drone.send_velocity(vx_body, vy_body, 0.0, self.target_yaw_deg)
                        
                        # Flush smoothing lag memories to prevent rapid snaps on recovery exit
                        self.planner.prev_north = None 
                        self.planner.prev_east = None
                    
                    else:
                        # Clear recovery flag once front clearance reopens
                        if self.is_recovering:
                            print("🎉 Path cleared! Resuming standard flight setpoints.")
                            self.is_recovering = False

                        if abs(heading_error) > self.large_turn_threshold:
                            # Large rotation required: Pivot on spot
                            print(f"🔄 Large Heading Shift Required ({heading_error:.1f}°). Holding position to pivot.")
                            await self.drone.send_position_setpoint(
                                north=self.pose["north"],
                                east=self.pose["east"],
                                down=self.pose["down"],
                                yaw_deg=self.target_yaw_deg
                            )
                        else:
                            # Standard tracking flight mode
                            await self.drone.send_position_setpoint(
                                north=north,
                                east=east,
                                down=down,
                                yaw_deg=self.target_yaw_deg
                            )
                    
                    self.last_command_time = current_time

                print(f"Pos: [{self.pose['north']:.2f}, {self.pose['east']:.2f}] -> Target: [{north:.2f}, {east:.2f}] | Yaw Target: {self.target_yaw_deg:.1f}° (Err: {heading_error:.1f}°)")

                elapsed = time.monotonic() - t_start
                sleep_time = (1.0 / self.loop_hz) - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            print("🛑 Navigation cancelled")
        finally:
            try:
                await self.drone.send_velocity(0, 0, 0, self.pose["yaw_deg"])
            except Exception:
                pass
            print("Drone hovering safely")

    def stop(self):
        self.running = False
        if self.worker is not None:
            self.worker.stop()

    async def main_lifecycle(self):
        # 1. RUN FLIGHT SETUP SCRIPT CONTEXT IN ISOLATION FIRST
        print("\n🔍 RUNNING SEQUENTIAL DRONE INITIALIZATION\n")
        await self.drone.connect()
        
        print("[System] Waiting for telemetry stream handshake...")
        await asyncio.sleep(3.0) 
        
        print("Starting position monitor.")
        self.monitor_task = asyncio.create_task(
            position_monitor_task(self.drone, self.position_state, asyncio.Event())
        )
        
        # 🛡️ RETRY LOOP FOR HEADING ESTIMATE SETTLING
        print("[Navigation] Attempting to Arm and Take Off...")
        armed_successfully = False
        attempt = 1
        
        while not armed_successfully and self.running:
            try:
                await self.drone.arm_and_takeoff()
                armed_successfully = True
                print("[Navigation] Takeoff sequence accepted successfully!")
            except ActionError as e:
                print(f"⚠️ [Attempt {attempt}] Arming denied by PX4: Heading estimate probably still invalid. Retrying in 3 seconds...")
                attempt += 1
                await asyncio.sleep(3.0)
        
        print("[System] Takeoff sequence complete. Holding 5 seconds to stabilize hover...")
        await asyncio.sleep(5.0)  

        # 2. DRONE IS SAFELY HOVERING. INITIALIZE HEAVY YOLO RESOURCES NOW.
        print("\n[System] Hover stable. Initializing YOLOv10 detector framework...")
        self.detector = Detector(
            base_model_path="yolov10n.pt",
            finetuned_weights=self.finetuned_path,
            device="cpu",
            conf_threshold=0.6
        )

        print("[System] Spinning up background YOLO inference worker thread...")
        target_save_directory = "../detections"
        self.worker = DetectorWorker(
            self.detection_queue, self.ui_queue, self.vision_shared_state, self.detector,
            save_dir=target_save_directory
        )
        self.worker.start()

        print("[Vision] Connecting to Gazebo network streams...")
        rgb_receiver = RGBReceiver(self.rgb_topic, self.vision_shared_state, self.detection_queue)
        
        # Pull down initialization frame coordinates
        await self.update_pose()
        self.target_yaw_deg = self.pose["yaw_deg"]

        # 3. START CONCURRENT MONITORING LOOPS
        print("[System] All systems nominal. Launching concurrent runtime loops.")
        await asyncio.gather(
            self.run_navigation_loop(),
            self.run_vision_ui()
        )

if __name__ == "__main__":
    # Hardcode VM networking overrides within the runtime environment 
    os.environ["GZ_IP"] = "127.0.0.1"
    os.environ["GZ_PARTITION"] = "default"
    
    app = IntegratedDroneApp()
    try:
        asyncio.run(app.main_lifecycle())
    except KeyboardInterrupt:
        print("\n⌨️ Shutdown command received.")
    finally:
        app.stop()
        print("[System] Exited cleanly.")