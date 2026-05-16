# WallFollower.py

class WallFollower:
    """
    Right-hand wall follower.
    Uses left/center/right clearance values from VelocityPlanner.

    Decision priority (evaluated top to bottom every tick):
      1. Emergency: forward critically close → stop + turn left
      2. Dead end: all sides blocked → U-turn
      3. Right wall lost (too far) → turn right to find wall
      4. Right wall too close → nudge left
      5. Forward clear + right in range → go straight
      6. Forward blocked → turn left
    """

    def __init__(self,
                 wall_follow_dist=2.0,    # target distance to keep from right wall (m)
                 wall_band=0.6,           # +/- tolerance around target (m)
                 critical_dist=1.0,       # emergency stop distance (m)
                 safe_dist=2.5,           # forward clearance = safe to proceed (m)
                 max_speed=0.8,           # forward cruise speed (m/s)
                 turn_speed=0.4):         # lateral/turn speed (m/s)
        self.wall_dist     = wall_follow_dist
        self.band          = wall_band
        self.critical      = critical_dist
        self.safe          = safe_dist
        self.max_speed     = max_speed
        self.turn_speed    = turn_speed

    def compute(self, left: float, center: float, right: float) -> dict:
        """
        Returns:
          vx   — forward speed in camera frame  (+ = forward)
          vy   — lateral speed in camera frame  (+ = right, - = left)
          state — string label for logging/debug
        """

        # ── 1. Emergency: about to hit something forward ───────────────────
        if center < self.critical:
            return self._cmd(0.0, -self.turn_speed, "EMERGENCY_LEFT")

        # ── 2. Dead end: blocked on all three sides ────────────────────────
        all_blocked = (center < self.safe and
                       left   < self.safe and
                       right  < self.safe)
        if all_blocked:
            return self._cmd(-self.max_speed * 0.5, -self.turn_speed, "UTURN")

        # ── 3. Right wall lost — open space on right, turn to find it ─────
        if right > self.wall_dist + self.band:
            # Slide right while still moving forward slowly
            return self._cmd(self.max_speed * 0.5, self.turn_speed, "FIND_WALL_RIGHT")

        # ── 4. Too close to right wall — nudge left ────────────────────────
        if right < self.wall_dist - self.band:
            return self._cmd(self.max_speed * 0.7, -self.turn_speed * 0.5, "NUDGE_LEFT")

        # ── 5. Forward blocked — turn left ────────────────────────────────
        if center < self.safe:
            return self._cmd(0.0, -self.turn_speed, "TURN_LEFT")

        # ── 6. Nominal: forward clear, right wall in range → cruise ───────
        return self._cmd(self.max_speed, 0.0, "FORWARD")

    @staticmethod
    def _cmd(vx, vy, state):
        return {"vx": vx, "vy": vy, "state": state}