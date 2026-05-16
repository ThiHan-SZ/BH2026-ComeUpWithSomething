import asyncio
import time
import queue

from mavsdk import System

from shared_state import SharedState
from Detector import Detector, DepthReceiver, RGBReceiver, DetectorWorker

RGB_TOPIC = "/world/roboverse/model/x500_depth_0/link/camera_link/sensor/IMX214/image"
DEPTH_TOPIC = "/depth_camera"

async def main():
    shared_state = SharedState()
    detection_queue = queue.Queue(maxsize=5)

    detector = Detector()
    depth_receiver = DepthReceiver(DEPTH_TOPIC, shared_state)
    rgb_receiver = RGBReceiver(RGB_TOPIC, shared_state, detection_queue)

    detector_worker = DetectorWorker(detection_queue, shared_state, detector)
    detector_worker.start()

    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("✓ Drone connected")
            break

    print("Arming drone...")
    await drone.action.arm()
    print("✓ Drone armed")

    print("Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(3)
    print("✓ Drone in air")

    print("\n--- Telemetry, Depth & Detection Data ---")
    start_time = time.time()
    telemetry_duration = 15

    async for position in drone.telemetry.position():
        elapsed = time.time() - start_time
        if elapsed > telemetry_duration:
            break

        state = shared_state.get_state()
        depth_data = state.get("depth")
        vision_data = state.get("vision")
        detection_data = state.get("planner")

        print(
            f"\n[{elapsed:.1f}s] Position - "
            f"Lat: {position.latitude_deg:.6f}, "
            f"Lon: {position.longitude_deg:.6f}, "
            f"Alt: {position.absolute_altitude_m:.2f}m"
        )

        if depth_data and depth_data["min_depth"] is not None:
            print(
                f"        Depth - "
                f"Min: {depth_data['min_depth']:.3f}m, "
                f"Max: {depth_data['max_depth']:.3f}m, "
                f"Mean: {depth_data['mean_depth']:.3f}m, "
                f"ValidPixels: {depth_data['valid_pixels']}"
            )
        else:
            print("        Depth - No valid data")


        if vision_data:
            print(
                f"        Vision - "
                f"Frame: {vision_data['shape']}, "
                f"FrameID: {vision_data['frame_id']}"
            )
        else:
            print("        Vision - No data")

        if detection_data:
            print(
                f"        Detection - "
                f"Status: {detection_data['status']}, "
                f"Objects: {detection_data['num_detections']}, "
                f"Detections: {detection_data['detections']}"
            )
        else:
            print("        Detection - Waiting for results...")

        print(f"        Queue - Size: {detection_queue.qsize()}/5")

        await asyncio.sleep(0.5)

    print("\n--- Landing ---")
    await drone.action.land()

    async for in_air in drone.telemetry.in_air():
        if not in_air:
            print("✓ Drone landed")
            break

    detector_worker.stop()
    detector_worker.join(timeout=2)

    print("Mission complete!")

if __name__ == "__main__":
    asyncio.run(main())
