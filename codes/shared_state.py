# shared_state.py
import threading

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.telemetry_data = None
        self.depth_data = None
        self.vision_data = None
        self.planner_data = None

    def update_telemetry(self, data):
        with self.lock:
            self.telemetry_data = data

    def update_depth(self, data):
        with self.lock:
            self.depth_data = data

    def update_vision(self, data):
        with self.lock:
            self.vision_data = data

    def update_planner(self, data):
        with self.lock:
            self.planner_data = data

    def get_state(self):
        with self.lock:
            return {
                'telemetry': self.telemetry_data,
                'depth': self.depth_data,
                'vision': self.vision_data,
                'planner': self.planner_data
            }
