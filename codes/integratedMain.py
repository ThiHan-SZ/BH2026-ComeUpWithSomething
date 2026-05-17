#!/usr/bin/env python3
import asyncio
import numpy as np
import time
import math
import os
import queue
import cv2

# ===================================
# IMPORT ALL MODULE DEPENDENCIES
# ===================================
from depth_receiver import DepthReceiver
from drone_control import Drone
from AvoidancePlanner import AvoidancePlanner
from get_position_with_task import SharedState as DroneSharedState, position_monitor_task

# Vision dependencies from Detector.py
from Detector import Detector, SharedState as VisionSharedState, RGBReceiver, DetectorWorker
from mavsdk.action import ActionError


class IntegratedDroneApp:
    def __init__(self, depth_topic="/depth_camera", loop_hz=10.0):
        print("[System] Initializing Integrated Navigation & Vision Systems...")
        self.loop_hz = loop_hz
        self.running = True

        # -----------------------------------
        # 🎯 NAVIGATION SETTINGS & TRACKING
        # -----------------------------------
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

        # Navigation Components
        self.receiver = DepthReceiver(depth_topic)
        self.planner = AvoidancePlanner(K=K, width=640, height=480, safe_distance=4.0, critical_distance=1.5)
        self.drone = Drone()
        self.position_state = DroneSharedState()    
        self.monitor_task = None
        
        self.last_command_time = 0.0
        self.command_interval = 0.1  

        # -----------------------------------
        # 👁️ VISION & DETECTION SYSTEM SETUP
        # -----------------------------------
        self.vision_state = VisionSharedState()
        self.detection_queue = queue.Queue(maxsize=1) 
        self.ui_queue = queue.Queue(maxsize=1) 

        # Track model file position
        finetuned = "models/barrel_detector_best.pt"
        if not os.path.exists(finetuned) and os.path.exists("../models/barrel_detector_best.pt"):
            finetuned = "../models/barrel_detector_best.pt"

        self.detector = Detector(
            base_model_path="yolov10n.pt",
            finetuned_weights=finetuned,
            device="cpu",
            conf_threshold=0.6
        )
        
        self.vision_worker = None
        self.rgb_receiver = None
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

    async def run_vision_ui_loop(self):
        """Asynchronous wrapper for managing OpenCV UI without blocking navigation."""
        print("[Vision] Spawning explicit UI loop helper...")
        cv2.namedWindow("Barrel Detections", cv2.WINDOW_AUTOSIZE)
        
        try:
            while self.running:
                try:
                    # Non-blocking fetch from UI Queue
                    frame_to_show = self.ui_queue.get_nowait()
                    cv2.imshow("Barrel Detections", frame_to_show)
                except queue.Empty:
                    pass

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[Vision] 'q' pressed. Shutting down system...")
                    self.stop()
                    break
                
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"[Vision UI Error] {e}")
        finally:
            cv2.destroyAllWindows()

    async def wait_for_arming_health(self):
        """Polls PX4 telemetry state until pre-flight checks are cleared."""
        print("[Navigation] Waiting for autopilot health checks to pass...")
        async for health in self.drone.drone.telemetry.health():
            if health.is_home_position_ok and health.is_local_position_ok:
                print("[Navigation] Pre-flight checks passed! Position locks obtained.")
                break
            print("[Navigation] Still waiting for position tracking alignment...")
            await asyncio.sleep(1.0)

    async def run(self):
        print("\n🔍 RUNNING CORNER-GUARDED INERTIAL RECOVERY NAVIGATION WITH INTEGRATED DETECTOR\n")

        # -------------------------------------------------------------
        # STEP 1: CONNECT & VERIFY HEALTH BEFORE ARMING
        # -------------------------------------------------------------
        print("[Navigation] Connecting to drone flight controller...")
        await self.drone.connect()
        await asyncio.sleep(2)
        
        print("[Navigation] Starting position monitor.")
        self.monitor_task = asyncio.create_task(position_monitor_task(self.drone, self.position_state, asyncio.Event()))
        
        # Wait until PX4 allows arming
        await self.wait_for_arming_health()

        print("[Navigation] Sending Arm and Takeoff command...")
        max_retries = 5
        for attempt in range(max_retries):
            try:
                await self.drone.arm_and_takeoff()
                print("[Navigation] Takeoff command accepted successfully.")
                break
            except ActionError as e:
                print(f"[Navigation] Arming attempt {attempt + 1}/{max_retries} Denied: {e}")
                if attempt == max_retries - 1:
                    print("❌ Critical: Autopilot persistently rejected arming sequence. Aborting.")
                    self.stop()
                    return
                await asyncio.sleep(2)
        
        await asyncio.sleep(1.0)
        await self.update_pose()
        self.target_yaw_deg = self.pose["yaw_deg"]

        # -------------------------------------------------------------
        # STEP 2: SPIN UP VISION HEAVY THREADS ONCE SAFELY AIRBORNE
        # -------------------------------------------------------------
        print("[Vision] Initializing backend detector pipeline...")
        target_save_directory = "../detections"
        self.vision_worker = DetectorWorker(
            self.detection_queue, 
            self.ui_queue, 
            self.vision_state, 
            self.detector,
            save_dir=target_save_directory
        )
        self.vision_worker.start()
        
        print("[Vision] Connecting to Gazebo camera network streams...")
        self.rgb_receiver = RGBReceiver(self.rgb_topic, self.vision_state, self.detection_queue)

        # Spawn the concurrent async OpenCV UI window helper task
        asyncio.create_task(self.run_vision_ui_loop())

        # -------------------------------------------------------------
        # STEP 3: MAIN NAVIGATION LOOP
        # -------------------------------------------------------------
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
                    
                    # 🔄 INERTIAL RECOVERY LOGIC (STRAIGHT BACKWARD)
                    if info['blocked']:
                        if not self.is_recovering:
                            self.recovery_yaw_deg = self.pose["yaw_deg"]
                            self.is_recovering = True
                            print(f"⚠️ DEAD END DETECTED! Latching baseline heading: {self.recovery_yaw_deg:.1f}°")

                        print(f"⚠️ RECOVERY ACTIVE: Backing out straight down corridor axis...")
                        
                        recovery_yaw_rad = np.deg2rad(self.recovery_yaw_deg)
                        v_north = -0.3 * math.cos(recovery_yaw_rad)
                        v_east  = -0.3 * math.sin(recovery_yaw_rad)
                        
                        current_yaw_rad = self.pose["yaw"]
                        vx_body = v_north * math.cos(current_yaw_rad) + v_east * math.sin(current_yaw_rad)
                        vy_body = -v_north * math.sin(current_yaw_rad) + v_east * math.cos(current_yaw_rad)
                        
                        await self.drone.send_velocity(vx_body, vy_body, 0.0, self.target_yaw_deg)
                        
                        self.planner.prev_north = None 
                        self.planner.prev_east = None
                    
                    else:
                        if self.is_recovering:
                            print("🎉 Path cleared! Resuming standard flight setpoints.")
                            self.is_recovering = False

                        if abs(heading_error) > self.large_turn_threshold:
                            print(f"🔄 Large Heading Shift Required ({heading_error:.1f}°). Holding position to pivot.")
                            await self.drone.send_position_setpoint(
                                north=self.pose["north"],
                                east=self.pose["east"],
                                down=self.pose["down"],
                                yaw_deg=self.target_yaw_deg
                            )
                        else:
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
            self.shutdown()

    def stop(self):
        self.running = False

    def shutdown(self):
        print("\n[System] Commencing unified shutdown sequences...")
        self.running = False
        
        try:
            print("[Navigation] Sending zero velocity hover command...")
            asyncio.create_task(self.drone.send_velocity(0, 0, 0, self.pose["yaw_deg"]))
        except Exception:
            pass

        if self.vision_worker is not None:
            print("[Vision] Stopping inference worker thread...")
            self.vision_worker.stop()
            
        print("[System] Exited cleanly.")


# ===================================
# ENTRY POINT
# ===================================
async def main():
    app = IntegratedDroneApp()
    task = asyncio.create_task(app.run())
    try:
        await task
    except KeyboardInterrupt:
        print("\n⌨️ Stopping application via Keyboard Interrupt...")
        app.stop()
        await asyncio.gather(task, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())