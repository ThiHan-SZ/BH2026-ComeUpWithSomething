import asyncio
from mavsdk import System, telemetry

async def print_flight_mode():
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!\n")
            break

    last_mode = None
    current_mode = None
    
    print("📡 Monitoring flight mode (Ctrl+C to exit)...\n")
    
    try:
        async for flight_mode in drone.telemetry.flight_mode():
            current_mode = flight_mode  # Track current mode
            
            # Only print changes
            if flight_mode != last_mode:
                print(f"FlightMode: {flight_mode}")
                last_mode = flight_mode
            
            await asyncio.sleep(0)
            
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n\n👋 Exiting...")
        
        # ✅ CLEANUP: Exit OFFBOARD if currently in it
        if current_mode == telemetry.FlightMode.OFFBOARD:
            print("🔧 Detected OFFBOARD mode - cleaning up...")
            try:
                await drone.offboard.stop()
                await asyncio.sleep(0.5)
                await drone.action.hold()
                print("✅ Switched to HOLD mode")
            except Exception as e:
                print(f"⚠️  Cleanup note: {e}")
        
        print("✅ Cleanup complete\n")

if __name__ == "__main__":
    try:
        asyncio.run(print_flight_mode())
    except KeyboardInterrupt:
        print()  # Clean newline