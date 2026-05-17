#!/usr/bin/env python3
import asyncio
import numpy as np
import time
import math

from depth_receiver import DepthReceiver
from drone_control import Drone
from AvoidancePlanner import AvoidancePlanner
from get_position_with_task import SharedState, position_monitor_task

class DroneNavigation:
    def __init__(self, depth_topic="/depth_camera", loop_hz=10.0):
        self.loop_hz = loop_hz
        self.running = True

        # ===================================
        # 🎯 CONTINUOUS HORIZON TRACKING
        # ===================================
        self.target_yaw_deg = 0.0
        self.yaw_tolerance = 4.0          
        self.large_turn_threshold = 15.0  

        # ===================================
        #  NED POSE TRACKING
        # ===================================
        self.pose = {
            "north": 0.0, "east": 0.0, "down": -2.0,
            "yaw": 0.0, "yaw_deg": 0.0
        }

        # ===================================
        # 🔄 STATE TRACKING FOR LOCKED RECOVERY
        # ===================================
        self.is_recovering = False
        self.recovery_yaw_deg = 0.0

        # Camera intrinsics
        K = np.array([[433.0, 0.0, 320.0],
                      [0.0, 433.0, 240.0],
                      [0.0, 0.0, 1.0]])

        self.receiver = DepthReceiver(depth_topic)
        self.planner = AvoidancePlanner(K=K, width=640, height=480, safe_distance=4.0, critical_distance=1.5)
        self.drone = Drone()
        self.position_state = SharedState()    
        self.monitor_task = None
        
        self.last_command_time = 0.0
        self.command_interval = 0.1  

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

    async def run(self):
        print("\n🔍 RUNNING CORNER-GUARDED INERTIAL RECOVERY NAVIGATION\n")

        await self.drone.connect()
        await asyncio.sleep(3)
        print("Starting position monitor.")
        self.monitor_task = asyncio.create_task(position_monitor_task(self.drone, self.position_state, asyncio.Event()))
        await self.drone.arm_and_takeoff()
        
        await asyncio.sleep(0.5)
        await self.update_pose()
        self.target_yaw_deg = self.pose["yaw_deg"]

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
                            # 🎯 LATCH RECOVERY HEADING:
                            # Capture the exact orientation the drone had right before stopping.
                            self.recovery_yaw_deg = self.pose["yaw_deg"]
                            self.is_recovering = True
                            print(f"⚠️ DEAD END DETECTED! Latching baseline heading: {self.recovery_yaw_deg:.1f}°")

                        print(f"⚠️ RECOVERY ACTIVE: Backing out straight down corridor axis...")
                        
                        # Calculate static backward vector components relative to the latched corridor angle
                        recovery_yaw_rad = np.deg2rad(self.recovery_yaw_deg)
                        v_north = -0.3 * math.cos(recovery_yaw_rad)
                        v_east  = -0.3 * math.sin(recovery_yaw_rad)
                        
                        # 🎯 Convert the calculated static global vector into a body command 
                        # so that spinning the camera doesn't distort the straight trajectory path.
                        current_yaw_rad = self.pose["yaw"]
                        vx_body = v_north * math.cos(current_yaw_rad) + v_east * math.sin(current_yaw_rad)
                        vy_body = -v_north * math.sin(current_yaw_rad) + v_east * math.cos(current_yaw_rad)
                        
                        await self.drone.send_velocity(vx_body, vy_body, 0.0, self.target_yaw_deg)
                        
                        # Flush smoothing lag memories to prevent rapid snaps on recovery exit
                        self.planner.prev_north = None 
                        self.planner.prev_east = None
                    
                    else:
                        # Clear recovery flag once the front clearance opens up again
                        if self.is_recovering:
                            print("🎉 Path cleared! Resuming standard flight setpoints.")
                            self.is_recovering = False

                        if abs(heading_error) > self.large_turn_threshold:
                            # Large rotation required: Pivot on spot using a unified setpoint call
                            print(f"🔄 Large Heading Shift Required ({heading_error:.1f}°). Holding position to pivot.")
                            await self.drone.send_position_setpoint(
                                north=self.pose["north"],
                                east=self.pose["east"],
                                down=self.pose["down"],
                                yaw_deg=self.target_yaw_deg
                            )
                        else:
                            # Standard tracking flight mode: forward setpoint + continuous fine yaw tracking
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


# ===================================
# ENTRY POINT
# ===================================
async def main():
    nav = DroneNavigation()
    task = asyncio.create_task(nav.run())
    try:
        await task
    except KeyboardInterrupt:
        print("\n⌨️ Stopping...")
        nav.stop()
        await asyncio.gather(task, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())