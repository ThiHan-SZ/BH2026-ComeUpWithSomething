# main.py
from structure import telemetry, depth, vision, planner
from shared_state import SharedState

def main():
    shared_state = SharedState()

    # Placeholder: Initialize telemetry interface
    def init_telemetry():
        # TODO: Connect to telemetry sensor or data source
        # Sample update
        shared_state.update_telemetry({"gps": (0,0), "speed": 0})

    # Placeholder: Initialize depth interface
    def init_depth():
        # TODO: Connect to depth sensor
        shared_state.update_depth({"depth_map": None})

    # Placeholder: Initialize vision interface
    def init_vision():
        # TODO: Connect to vision system
        shared_state.update_vision({"image": None})

    # Placeholder: Initialize planner
    def init_planner():
        # TODO: Use shared state data to plan
        shared_state.update_planner({"next_move": None})

    # Initialize all components
    init_telemetry()
    init_depth()
    init_vision()
    init_planner()

    # Example main loop - update and process
    while True:
        # Example: update telemetry
        init_telemetry()
        
        # Example: update depth
        init_depth()
        
        # Example: update vision
        init_vision()

        # Example: run planner with updated data
        init_planner()

        # Access shared state for decision or output
        state_snapshot = shared_state.get_state()
        print(state_snapshot)

        # Add sleep or loop exit condition as needed
        break  # just run once for demonstration

if __name__ == "__main__":
    main()

