# #!/usr/bin/env python3
# import asyncio
# import numpy as np
# import time

# from depth_receiver import DepthReceiver
# from drone_control import Drone
# from AvoidancePlanner import AvoidancePlanner
# from get_position_with_task import SharedState, position_monitor_task

# class DroneNavigation:
#     def __init__(self,
#                  depth_topic="/depth_camera",
#                  loop_hz=20.0):

#         self.loop_hz = loop_hz
#         self.running = True

#         # =========================
#         # GRID HEADING SYSTEM
#         # =========================
#         self.grid_headings = [0, 90, 180, -90]  # N, E, S, W
#         self.current_heading_idx = 0
#         self.target_yaw_deg = self.grid_headings[self.current_heading_idx]
#         self.yaw_tolerance = 5.0

#         # =========================
#         #  NED POSE TRACKING
#         # =========================
#         self.pose = {
#             "north": 0.0,
#             "east": 0.0,
#             "down": -2.0,
#             "yaw": 0.0,
#             "yaw_deg": 0.0
#         }

#         # Camera intrinsics
#         K = np.array([[433.0, 0.0, 320.0],
#                       [0.0, 433.0, 240.0],
#                       [0.0, 0.0, 1.0]])

#         self.receiver = DepthReceiver(depth_topic)

#         self.planner = AvoidancePlanner(
#             K=K,
#             width=640,
#             height=480,
#             safe_distance=4.0,
#             critical_distance=1.5
#         )

#         self.drone = Drone()
#         self.position_state = SharedState()    


#     # =========================
#     #  YAW UTILS
#     # =========================
#     def _yaw_error(self, target, current):
#         error = target - current
#         while error > 180:
#             error -= 360
#         while error < -180:
#             error += 360
#         return error

#     async def update_pose(self):
#         """
#         Get pose from drone from state shared by position monitor task. This is critical to ensure the planner has the latest pose for decision making.
#         """
#         self.pose["north"] = self.position_state.latest_position.north_m
#         self.pose["east"]  = self.position_state.latest_position.east_m
#         self.pose["down"]  = self.position_state.latest_position.down_m
#         self.pose["yaw_deg"]   = self.position_state.latest_yaw
#         self.pose["yaw"] = np.deg2rad(self.pose["yaw_deg"])

#     async def align_to_grid(self):
#         current_yaw = await self.drone.get_yaw()
#         error = self._yaw_error(self.target_yaw_deg, current_yaw)

#         if abs(error) > self.yaw_tolerance:
#             print(f"Aligning to {self.target_yaw_deg}° (err={error:.2f})")
#             await self.drone.rotate_to_yaw(self.target_yaw_deg)

#     # =========================
#     # 🔄 GRID TURNING
#     # =========================
#     async def rotate_next_direction(self):
#         self.current_heading_idx = (self.current_heading_idx + 1) % 4
#         self.target_yaw_deg = self.grid_headings[self.current_heading_idx]

#         print(f"🔄 New heading: {self.target_yaw_deg}°")
#         await self.drone.rotate_to_yaw(self.target_yaw_deg)

#     # =========================
#     # tHE MAIN LOOP WHERE THE PIPELINE COMES TOGETHER
#     # =========================
#     async def run(self):
#         print("\nPOSITION-BASED AUTONOMOUS AvoidanceNAVIGATION\n")

#         await self.drone.connect()
#         await asyncio.sleep(3)
#         print("Starting position monitor.")
#         self.monitor_task = asyncio.create_task(position_monitor_task(self.drone, self.position_state, asyncio.Event()))
#         await self.drone.arm_and_takeoff()
#         # Initial alignment
#         await self.drone.rotate_to_yaw(self.target_yaw_deg)

#         try:
#             while self.running:
#                 t_start = time.monotonic()

#                 # -----------------------------------
#                 # UPDATE POSE (CRITICAL)
#                 # -----------------------------------
#                 await self.update_pose()

#                 depth_frame = self.receiver.get_frame()

#                 # -----------------------------------
#                 # POSITION PLANNER
#                 # -----------------------------------
#                 north, east, down, info = self.planner.compute_position_ned(
#                     depth_frame,
#                     self.pose,
#                     step_size=1.5
#                 )

#                 c = info['clearance']

#                 print(f"Blocked={info['blocked']} | "
#                       f"Target N={north:.2f}, E={east:.2f} | "
#                       f"L={c['left']:.2f} C={c['center']:.2f} R={c['right']:.2f}")

#                 # ===================================
#                 #  BLOCK HANDLING
#                 # ===================================
#                 if info['blocked']:
#                     await self.drone.send_velocity(0, 0, 0, self.target_yaw_deg)
#                     await self.rotate_next_direction()
#                 else:
#                     # Ensure alignment before motion
#                     await self.align_to_grid()

#                     # -----------------------------------
#                     #  SEND POSITION SETPOINT
#                     # -----------------------------------
#                     await self.drone.send_position_setpoint(
#                         north=north,
#                         east=east,
#                         down=down,
#                         yaw_deg=self.target_yaw_deg
#                     )

#                 # Maintain loop timing
#                 elapsed = time.monotonic() - t_start
#                 sleep_time = (1.0 / self.loop_hz) - elapsed
#                 if sleep_time > 0:
#                     await asyncio.sleep(sleep_time)

#         except asyncio.CancelledError:
#             print("🛑 Navigation cancelled")

#         finally:
#             await self.drone.send_velocity(0, 0, 0, self.target_yaw_deg)
#             print("Drone hovering safely")

#     def stop(self):
#         self.running = False


# # =========================
# #  ENTRY POINT
# # =========================
# async def main():
#     nav = DroneNavigation()

#     task = asyncio.create_task(nav.run())

#     try:
#         await task
#     except KeyboardInterrupt:
#         print("\n⌨️ Stopping...")
#         nav.stop()
#         await asyncio.gather(task, return_exceptions=True)


# if __name__ == "__main__":
#     asyncio.run(main())




#!/usr/bin/env python3
"""
Minimal obstacle avoidance mission loop.
Reads depth, computes safe movement, sends commands, repeats.
"""

import time
import numpy as np
from enum import Enum

# Import from your existing modules
from drone_control import DroneControl
from AvoidancePlanner import AvoidancePlanner
from depth_receiver import DepthReceiver
# Assuming Person B's telemetry is in shared_state.py or similar
try:
    from shared_state import TelemetryState
except ImportError:
    print("WARNING: shared_state not found, using dummy telemetry")
    class TelemetryState:
        def __init__(self):
            self.position_ned = [0.0, 0.0, 0.0]
            self.heading = 0.0
            self.armed = False
            self.mode = "UNKNOWN"

# Constants
LOOP_RATE_HZ = 10  # 10 Hz mission loop
DEPTH_TIMEOUT_SEC = 1.0
BLOCKED_YAW_RATE = 20.0  # degrees/sec for fallback turn
BLOCKED_SIDESTEP_SPEED = 0.3  # m/s for fallback sidestep
CRUISE_SPEED = 0.5  # m/s forward when path is clear
CRUISE_ALTITUDE = -3.0  # 3m above ground (NED: negative is up)

class MissionState(Enum):
    IDLE = 0
    CRUISING = 1
    AVOIDING = 2
    BLOCKED_TURNING = 3
    BLOCKED_SIDESTEPPING = 4

class ObstacleAvoidanceMission:
    def __init__(self):
        print("[AVOID] Initializing mission...")
        
        # Initialize components
        self.drone = DroneControl()
        self.planner = AvoidancePlanner()
        self.depth = DepthReceiver()
        self.telemetry = TelemetryState()
        
        # Mission state
        self.state = MissionState.IDLE
        self.last_depth_time = 0
        self.blocked_counter = 0
        self.fallback_timer = 0
        
        print("[AVOID] Mission initialized")
    
    def get_depth_frame(self):
        """Get latest depth frame from receiver."""
        try:
            frame = self.depth.get_latest_frame()
            if frame is not None:
                self.last_depth_time = time.time()
                return frame
            else:
                # Check timeout
                if time.time() - self.last_depth_time > DEPTH_TIMEOUT_SEC:
                    print("[AVOID] WARNING: Depth frame timeout")
                return None
        except Exception as e:
            print(f"[AVOID] ERROR getting depth frame: {e}")
            return None
    
    def compute_movement(self, depth_frame):
        """
        Compute safe movement from depth frame.
        Returns: (vx, vy, vz, yaw_rate, is_blocked)
        """
        try:
            # Use planner to compute safe velocity
            result = self.planner.compute_safe_velocity(
                depth_frame,
                current_heading=self.telemetry.heading,
                target_speed=CRUISE_SPEED
            )
            
            # Assuming planner returns dict with:
            # {'vx': float, 'vy': float, 'yaw_rate': float, 'status': str}
            if result['status'] in ['BLOCKED', 'NO_PATH', 'INVALID']:
                print(f"[AVOID] Path blocked: {result['status']}")
                return 0.0, 0.0, 0.0, 0.0, True
            
            # Add vertical component (maintain altitude)
            vz = self._compute_altitude_correction()
            
            return result['vx'], result['vy'], vz, result['yaw_rate'], False
            
        except Exception as e:
            print(f"[AVOID] ERROR in planner: {e}")
            return 0.0, 0.0, 0.0, 0.0, True
    
    def _compute_altitude_correction(self):
        """Simple altitude hold."""
        current_alt = self.telemetry.position_ned
        error = CRUISE_ALTITUDE - current_alt
        # Simple P controller
        vz = np.clip(error * 0.5, -0.5, 0.5)
        return vz
    
    def send_velocity_command(self, vx, vy, vz, yaw_rate):
        """Send velocity command to drone."""
        try:
            self.drone.send_velocity_ned(vx, vy, vz, yaw_rate)
        except Exception as e:
            print(f"[AVOID] ERROR sending command: {e}")
    
    def handle_blocked_case(self):
        """
        Simple fallback when path is blocked.
        Alternates between turning and sidestepping.
        """
        self.blocked_counter += 1
        
        # Decide fallback strategy every 2 seconds
        if self.blocked_counter % (LOOP_RATE_HZ * 2) == 0:
            # Switch strategy
            if self.state == MissionState.BLOCKED_TURNING:
                self.state = MissionState.BLOCKED_SIDESTEPPING
                print("[AVOID] Blocked → trying sidestep")
            else:
                self.state = MissionState.BLOCKED_TURNING
                print("[AVOID] Blocked → trying turn")
            self.fallback_timer = 0
        
        # Execute fallback
        vz = self._compute_altitude_correction()
        
        if self.state == MissionState.BLOCKED_TURNING:
            # Turn in place
            self.send_velocity_command(0.0, 0.0, vz, BLOCKED_YAW_RATE)
        else:
            # Sidestep right
            self.send_velocity_command(0.0, BLOCKED_SIDESTEP_SPEED, vz, 0.0)
        
        self.fallback_timer += 1
        
        # Reset counter if blocked too long (30 seconds)
        if self.blocked_counter > LOOP_RATE_HZ * 30:
            print("[AVOID] CRITICAL: Blocked for 30s, stopping mission")
            self.state = MissionState.IDLE
            self.blocked_counter = 0
    
    def mission_loop(self):
        """Main mission loop."""
        print("[AVOID] Starting mission loop at {LOOP_RATE_HZ} Hz")
        rate = 1.0 / LOOP_RATE_HZ
        
        try:
            while True:
                loop_start = time.time()
                
                # Check if drone is ready
                if not self.telemetry.armed:
                    if self.state != MissionState.IDLE:
                        print("[AVOID] Drone disarmed, pausing mission")
                        self.state = MissionState.IDLE
                    time.sleep(rate)
                    continue
                
                # Get depth frame
                depth_frame = self.get_depth_frame()
                
                if depth_frame is None:
                    # No depth data - stop and wait
                    print("[AVOID] No depth data, holding position")
                    self.send_velocity_command(0.0, 0.0, 0.0, 0.0)
                    time.sleep(rate)
                    continue
                
                # Compute movement
                vx, vy, vz, yaw_rate, is_blocked = self.compute_movement(depth_frame)
                
                if is_blocked:
                    # Handle blocked case
                    self.handle_blocked_case()
                else:
                    # Clear path - send normal command
                    if self.blocked_counter > 0:
                        print("[AVOID] Path clear again")
                        self.blocked_counter = 0
                    
                    self.state = MissionState.CRUISING if vx > 0.1 else MissionState.AVOIDING
                    self.send_velocity_command(vx, vy, vz, yaw_rate)
                    
                    # Debug print every 2 seconds
                    if int(time.time()) % 2 == 0:
                        print(f"[AVOID] {self.state.name}: vx={vx:.2f}, vy={vy:.2f}, yaw={yaw_rate:.1f}")
                
                # Maintain loop rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, rate - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n[AVOID] Mission interrupted by user")
        except Exception as e:
            print(f"[AVOID] CRITICAL ERROR in mission loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Stop drone
            print("[AVOID] Stopping drone...")
            self.send_velocity_command(0.0, 0.0, 0.0, 0.0)

def main():
    mission = ObstacleAvoidanceMission()
    mission.mission_loop()

if __name__ == "__main__":
    main()
