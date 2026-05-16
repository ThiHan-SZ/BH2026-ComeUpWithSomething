# main.py
import asyncio
import time
from mavsdk import System

async def main():
    # Initialize drone
    drone = System()
    
    # Connect to drone (adjust connection string for your setup)
    await drone.connect(system_address="udp://:14540")
    
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("✓ Drone connected")
            break
    
    # Arm drone
    print("Arming drone...")
    await drone.action.arm()
    print("✓ Drone armed")
    
    # Takeoff
    print("Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(3)
    print("✓ Drone in air")
    
    # Start telemetry task - print NED position for 10-20 seconds
    print("\n--- Telemetry: NED Position ---")
    start_time = time.time()
    telemetry_duration = 15  # seconds (10-20 range)
    
    async def print_telemetry():
        async for position in drone.telemetry.position():
            elapsed = time.time() - start_time
            if elapsed > telemetry_duration:
                break
            
            # Extract NED position (relative to home)
            print(f"Time: {elapsed:.1f}s | Position - "
                  f"Lat: {position.latitude_deg:.6f}, "
                  f"Lon: {position.longitude_deg:.6f}, "
                  f"Alt: {position.absolute_altitude_m:.2f}m")
            
            await asyncio.sleep(0.5)  # Print every 0.5 seconds
    
    # Run telemetry task
    await print_telemetry()
    
    print("\n--- Landing ---")
    # Land drone
    await drone.action.land()
    
    # Wait for landing to complete
    while True:
        async for in_air in drone.telemetry.in_air():
            if not in_air:
                print("✓ Drone landed")
                break
        break
    
    print("Mission complete!")

if __name__ == "__main__":
    asyncio.run(main())